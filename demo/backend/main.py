"""Interactive attention-steering demo backend.

FastAPI server: load a model (Qwen3-VL or Gemma-4 family), send it an image
+ prompt + a mouse-drawn bounding box, and get back baseline vs. attention-
steered generations. Reuses vis_head's existing discovery/steering machinery
(vis_head.vir, vis_head.steering, vis_head.demo_adapters) -- no new model
logic beyond the family adapters in demo_adapters.py.

Run: uvicorn main:app --host 0.0.0.0 --port 8000  (from demo/backend/, with
the virheads conda env active and the repo root importable).
"""
from __future__ import annotations

import base64
import gc
import io
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from vis_head.demo_adapters import (
    bbox_to_token_positions, gemma_family, image_token_range, load_model,
    model_dims, prepare_inputs,
)
from vis_head.vir import collect_last_query_attentions, rank_heads_by_score
from vis_head.steering import (
    group_heads_by_layer, intervention_positions, make_static_attention_mask_hook,
    register_mask_hooks, remove_handles,
)

DEVICE = "cuda:0"

MODELS = {
    "qwen3_vl_instruct": {"label": "Qwen3-VL-8B-Instruct", "model_id": "Qwen/Qwen3-VL-8B-Instruct"},
    "qwen3_vl_thinking": {"label": "Qwen3-VL-8B-Thinking", "model_id": "Qwen/Qwen3-VL-8B-Thinking"},
    "gemma4_e4b_it": {"label": "Gemma-4-E4B-it", "model_id": "google/gemma-4-E4B-it"},
    "gemma3n_e4b_it": {"label": "Gemma-3n-E4B-it", "model_id": "google/gemma-3n-E4B-it"},
    "gemma3n_e2b_it": {"label": "Gemma-3n-E2B-it", "model_id": "google/gemma-3n-E2B-it"},
}

PRECOMPUTED_HEADS_DIR = REPO_ROOT / "demo" / "backend" / "precomputed_heads"

app = FastAPI(title="Vis-Head Interactive Steering Demo")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.exception_handler(Exception)
async def _json_error_handler(request, exc):
    # Any uncaught exception would otherwise fall through to FastAPI/Starlette's
    # default HTML error page, which the frontend's `res.json()` can't parse
    # (surfaces there as "Unexpected token '<'..." or similar). Always return
    # JSON with the actual error message instead.
    import traceback
    traceback.print_exc()
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})

EXAMPLES_DIR = REPO_ROOT / "demo" / "examples"
EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/examples", StaticFiles(directory=str(EXAMPLES_DIR)), name="examples")


class _Session:
    def __init__(self):
        self.model_key = None
        self.model = None
        self.processor = None
        self.family = None
        self.n_layers = 0
        self.n_heads = 0

    def unload(self):
        if self.model is not None:
            del self.model
            del self.processor
            gc.collect()
            torch.cuda.empty_cache()
        self.model_key = self.model = self.processor = self.family = None
        self.n_layers = self.n_heads = 0


SESSION = _Session()


def _decode_image(image_b64: str) -> Image.Image:
    raw = base64.b64decode(image_b64.split(",")[-1])
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _get_letter_token_ids(tokenizer, letters):
    ids = {}
    for letter in letters:
        candidates = set()
        for cand in (letter, f" {letter}"):
            enc = tokenizer.encode(cand, add_special_tokens=False)
            if len(enc) == 1:
                candidates.add(enc[0])
        ids[letter] = sorted(candidates)
    return ids


class LoadRequest(BaseModel):
    model_key: str


class DiscoverRequest(BaseModel):
    image: str          # base64 data URL
    prompt: str
    bbox: list[float]   # [x0, y0, x1, y1] normalized to the ORIGINAL image
    top_k: int = 15


class GenerateRequest(BaseModel):
    image: str
    prompt: str
    bbox: list[float]
    heads: list[list[int]]   # [[layer, head], ...]
    max_new_tokens: int = 80
    mode: str = "boost_suppress"   # "boost_suppress" | "suppress_image" | "suppress_all" | "max_suppress_all"


class McqRequest(BaseModel):
    image: str
    bbox: list[float]
    heads: list[list[int]]
    options: list[str]       # candidate object names, e.g. cell_names from example metadata
    correct_index: int       # which option is ground truth for this bbox
    mode: str = "boost_suppress"
    max_new_tokens: int = 6  # instant-answer models; "thinking"/CoT checkpoints need far more


@app.get("/api/models")
def list_models():
    out = []
    for k, v in MODELS.items():
        has_precomputed = (PRECOMPUTED_HEADS_DIR / f"{k}.json").exists()
        out.append({"key": k, "label": v["label"], "has_precomputed_heads": has_precomputed})
    return out


@app.get("/api/precomputed_heads/{model_key}")
def get_precomputed_heads(model_key: str):
    """Real head rankings averaged over N=300 discovery samples (or N=300 x
    150-token CoT traces for reasoning models), computed offline via
    dump_precomputed_heads.py -- NOT the noisy single-sample live discovery
    that /api/discover does. Use these as the "validated preset" instead of
    (or as a sanity check against) a fresh live discovery call."""
    path = PRECOMPUTED_HEADS_DIR / f"{model_key}.json"
    if not path.exists():
        raise HTTPException(404, f"No precomputed heads for {model_key!r}.")
    import json as _json
    return _json.loads(path.read_text())


@app.get("/api/examples")
def list_examples():
    import json as _json
    files = sorted(p.name for p in EXAMPLES_DIR.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    out = []
    for f in files:
        entry = {"name": f, "url": f"/examples/{f}"}
        meta_path = EXAMPLES_DIR / f"{f}.meta.json"
        if meta_path.exists():
            entry["mcq"] = _json.loads(meta_path.read_text())
        out.append(entry)
    return out


@app.post("/api/load")
def load(req: LoadRequest):
    if req.model_key not in MODELS:
        raise HTTPException(404, f"Unknown model_key {req.model_key!r}")
    if SESSION.model_key == req.model_key:
        return {"status": "already_loaded", "n_layers": SESSION.n_layers, "n_heads": SESSION.n_heads}
    SESSION.unload()
    model_id = MODELS[req.model_key]["model_id"]
    family = gemma_family(model_id)
    model, processor = load_model(model_id, device=DEVICE)
    n_layers, n_heads = model_dims(family, model)
    SESSION.model_key, SESSION.model, SESSION.processor, SESSION.family = req.model_key, model, processor, family
    SESSION.n_layers, SESSION.n_heads = n_layers, n_heads
    return {"status": "loaded", "n_layers": n_layers, "n_heads": n_heads, "family": family}


@app.post("/api/discover")
def discover(req: DiscoverRequest):
    if SESSION.model is None:
        raise HTTPException(400, "No model loaded. Call /api/load first.")
    image = _decode_image(req.image)
    inputs = prepare_inputs(SESSION.family, SESSION.processor, image, req.prompt, DEVICE)
    img_start, img_end = image_token_range(SESSION.family, inputs, SESSION.processor)
    target, _ = bbox_to_token_positions(SESSION.family, inputs, SESSION.model, img_start, img_end, tuple(req.bbox))
    if not target:
        raise HTTPException(400, "Selected region contains no image tokens.")

    attn = collect_last_query_attentions(SESSION.model, inputs)  # (L, H, T)
    target_rel = np.array(target) - img_start
    # aggregate_region_attention needs find_image_token_range internally
    # (architecture-specific); img_start/img_end are already known here, so
    # sum the raw target-token attention directly instead.
    image_attn = attn[:, :, img_start:img_end]
    scores = image_attn[:, :, target_rel].sum(axis=-1)  # (L, H)

    ranked = rank_heads_by_score(scores)
    top = ranked[: req.top_k]
    return {"heads": [{"layer": r["layer"], "head": r["head"], "score": r["score"]} for r in top]}


def _run_generate(inputs, prompt_length, heads_by_layer, max_new_tokens):
    handles = []
    if heads_by_layer:
        hook_by_layer = {
            l: make_static_attention_mask_hook(
                head_indices=hh, suppress_positions=_CURRENT_SUPPRESS, boost_positions=_CURRENT_BOOST,
                n_query_heads=SESSION.n_heads, device=DEVICE, decode_only=False, pad_with_suppress=_CURRENT_PAD,
            )
            for l, hh in heads_by_layer.items()
        }
        handles = register_mask_hooks(SESSION.model, hook_by_layer)
    try:
        with torch.no_grad():
            out = SESSION.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    finally:
        remove_handles(handles)
    text = SESSION.processor.tokenizer.decode(out[0, prompt_length:], skip_special_tokens=True)
    return text.strip()


_CURRENT_SUPPRESS: list[int] = []
_CURRENT_BOOST: list[int] = []
_CURRENT_PAD: bool = False


@app.post("/api/generate")
def generate(req: GenerateRequest):
    global _CURRENT_SUPPRESS, _CURRENT_BOOST, _CURRENT_PAD
    if SESSION.model is None:
        raise HTTPException(400, "No model loaded. Call /api/load first.")
    image = _decode_image(req.image)
    inputs = prepare_inputs(SESSION.family, SESSION.processor, image, req.prompt, DEVICE)
    prompt_length = int(inputs["input_ids"].shape[1])
    img_start, img_end = image_token_range(SESSION.family, inputs, SESSION.processor)
    target, other = bbox_to_token_positions(SESSION.family, inputs, SESSION.model, img_start, img_end, tuple(req.bbox))
    if not target:
        raise HTTPException(400, "Selected region contains no image tokens.")

    suppress_positions, boost_positions, pad_with_suppress = intervention_positions(
        mode=req.mode, target_positions=target, other_image_positions=other,
        img_start=img_start, img_end=img_end, prompt_length=prompt_length,
    )
    _CURRENT_SUPPRESS, _CURRENT_BOOST, _CURRENT_PAD = suppress_positions, boost_positions, pad_with_suppress

    baseline_text = _run_generate(inputs, prompt_length, None, req.max_new_tokens)

    heads_by_layer = group_heads_by_layer([(l, h) for l, h in req.heads])
    steered_text = _run_generate(inputs, prompt_length, heads_by_layer, req.max_new_tokens)

    return {
        "baseline": baseline_text,
        "steered": steered_text,
        "n_target_tokens": len(target),
        "n_other_tokens": len(other),
    }


OPTION_LETTERS = ["A", "B", "C", "D", "E", "F"]


def _run_mcq_generate(inputs, prompt_length, heads_by_layer, letter_token_ids, all_letter_ids_flat, id_to_letter,
                       max_new_tokens=6):
    """Same pattern used throughout this project's notebooks: greedy generate,
    scan generated tokens for the LAST one naming an option letter (handles
    both instant-answer and reasoning models), read real logits at that step
    for a genuine probability vector over options."""
    handles = []
    if heads_by_layer:
        hook_by_layer = {
            l: make_static_attention_mask_hook(
                head_indices=hh, suppress_positions=_CURRENT_SUPPRESS, boost_positions=_CURRENT_BOOST,
                n_query_heads=SESSION.n_heads, device=DEVICE, decode_only=False, pad_with_suppress=_CURRENT_PAD,
            )
            for l, hh in heads_by_layer.items()
        }
        handles = register_mask_hooks(SESSION.model, hook_by_layer)
    try:
        with torch.no_grad():
            out = SESSION.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                output_scores=True, return_dict_in_generate=True,
            )
    finally:
        remove_handles(handles)
    gen_ids = out.sequences[0, prompt_length:].tolist()
    answer_step = None
    for step in range(len(gen_ids) - 1, -1, -1):
        if gen_ids[step] in all_letter_ids_flat:
            answer_step = step
            break
    if answer_step is None:
        return None, None
    predicted = id_to_letter[gen_ids[answer_step]]
    step_logits = out.scores[answer_step][0].float()
    letter_logits = [step_logits[letter_token_ids[l]].max().item() for l in letter_token_ids]
    probs = torch.softmax(torch.tensor(letter_logits), dim=0).tolist()
    return predicted, probs


@app.post("/api/generate_mcq")
def generate_mcq(req: McqRequest):
    """MCQ mode: matches the paradigm this project's rigorous experiments use
    to demonstrate large, unambiguous steering effects (forced choice among
    real candidate object names, single-token answer) -- much stronger signal
    than free-form description, at the cost of needing known candidate names
    for the image (only available for the example composite-grid images,
    which carry per-region ground truth)."""
    global _CURRENT_SUPPRESS, _CURRENT_BOOST, _CURRENT_PAD
    if SESSION.model is None:
        raise HTTPException(400, "No model loaded. Call /api/load first.")
    if not (0 <= req.correct_index < len(req.options)):
        raise HTTPException(400, "correct_index out of range for options.")
    letters = OPTION_LETTERS[: len(req.options)]
    option_lines = "  ".join(f"{l}) {name}" for l, name in zip(letters, req.options))
    prompt = f"Which of the following is shown in this image? {option_lines}. Answer with only the letter."
    correct_letter = letters[req.correct_index]

    image = _decode_image(req.image)
    inputs = prepare_inputs(SESSION.family, SESSION.processor, image, prompt, DEVICE)
    prompt_length = int(inputs["input_ids"].shape[1])
    img_start, img_end = image_token_range(SESSION.family, inputs, SESSION.processor)
    target, other = bbox_to_token_positions(SESSION.family, inputs, SESSION.model, img_start, img_end, tuple(req.bbox))
    if not target:
        raise HTTPException(400, "Selected region contains no image tokens.")

    suppress_positions, boost_positions, pad_with_suppress = intervention_positions(
        mode=req.mode, target_positions=target, other_image_positions=other,
        img_start=img_start, img_end=img_end, prompt_length=prompt_length,
    )
    _CURRENT_SUPPRESS, _CURRENT_BOOST, _CURRENT_PAD = suppress_positions, boost_positions, pad_with_suppress

    letter_token_ids = _get_letter_token_ids(SESSION.processor.tokenizer, letters)
    all_letter_ids_flat = set(i for ids in letter_token_ids.values() for i in ids)
    id_to_letter = {i: l for l, ids in letter_token_ids.items() for i in ids}

    # "Thinking"/CoT checkpoints emit a long reasoning trace before committing
    # to the answer letter -- the default budget (tuned for instant-answer
    # models) isn't enough to ever reach it. Auto-bump unless the caller
    # explicitly asked for something other than the field default.
    effective_max_new_tokens = req.max_new_tokens
    if req.max_new_tokens == 6 and "thinking" in (SESSION.model_key or "").lower():
        effective_max_new_tokens = 400

    baseline_letter, baseline_probs = _run_mcq_generate(
        inputs, prompt_length, None, letter_token_ids, all_letter_ids_flat, id_to_letter,
        max_new_tokens=effective_max_new_tokens)

    heads_by_layer = group_heads_by_layer([(l, h) for l, h in req.heads])
    steered_letter, steered_probs = _run_mcq_generate(
        inputs, prompt_length, heads_by_layer, letter_token_ids, all_letter_ids_flat, id_to_letter,
        max_new_tokens=effective_max_new_tokens)

    return {
        "prompt": prompt,
        "options": dict(zip(letters, req.options)),
        "correct_letter": correct_letter,
        "baseline_letter": baseline_letter,
        "baseline_probs": dict(zip(letters, baseline_probs)) if baseline_probs else None,
        "steered_letter": steered_letter,
        "steered_probs": dict(zip(letters, steered_probs)) if steered_probs else None,
        "n_target_tokens": len(target),
        "n_other_tokens": len(other),
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "loaded_model": SESSION.model_key, "gpu": torch.cuda.is_available()}
