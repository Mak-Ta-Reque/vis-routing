"""Random-head ablation control: is base's higher causal effect specific to gaze
heads, or is base just generally more fragile to any attention perturbation?

Reuses cached vis_head_scores.npy (no need to redo the 200-comic discovery pass).
Same comic/target-panel sample sequence is used for both the gaze-head and
random-head conditions (both draw from the same seeded RNG inside
causal_ablation_effects), so within each model the two conditions are paired —
enabling a Wilcoxon signed-rank test, not just an unpaired comparison.
"""
import sys
from pathlib import Path

REPO_ROOT = Path("/mnt/abka03/Projects/vis-head")
sys.path.insert(0, str(REPO_ROOT))

import gc
import json
import numpy as np
import torch
from scipy import stats
from tqdm.auto import tqdm

from vis_head.common import DEFAULT_COMICS_ROOT, DEFAULT_N_PANELS, DEFAULT_SEED, dump_json
from vis_head.data import build_strip, list_comic_dirs
from vis_head.gaze import panel_query_prompt, rank_heads_by_score
from vis_head.judge import semantic_similarity
from vis_head.modeling import decode_generated_text, find_image_token_range, load_model_and_processor, model_dims, run_generation
from vis_head.regions import assign_panels_to_tokens, region_positions_from_ids
from vis_head.steering import group_heads_by_layer, make_static_attention_mask_hook, register_mask_hooks, remove_handles

DEVICE = "cuda:0"
SEED = DEFAULT_SEED
N_PANELS = DEFAULT_N_PANELS
N_CAUSAL_SAMPLES = 40
CAUSAL_MAX_NEW_TOKENS = 40
HEAD_BUDGETS = [50, 15, 5]
EXCLUDE_TOP_N = 200   # heads excluded from the random pool, so "random" really means non-vis-head
RANDOM_CONTROL_SEED = SEED + 100

MODELS = {
    "base": "Qwen/Qwen2-VL-7B",
    "instruct": "Qwen/Qwen2-VL-7B-Instruct",
    "agentic": "ByteDance-Seed/UI-TARS-7B-DPO",
}

COMICS_ROOT = Path(DEFAULT_COMICS_ROOT)
comic_dirs = list_comic_dirs(COMICS_ROOT, n_panels=N_PANELS)[:200]

# --- load cached scores, rebuild the combined ranking exactly as the notebook did ---
scores_by_tag = {}
for tag in MODELS:
    scores_by_tag[tag] = np.load(REPO_ROOT / "logs" / f"vis_head_discovery_qwen2vl_{tag}" / "vis_head_scores.npy")
n_layers, n_heads = scores_by_tag["base"].shape
combined_score = np.mean([scores_by_tag[t].astype(np.float64) for t in MODELS], axis=0)
combined_ranked = rank_heads_by_score(combined_score)

excluded = set((r["layer"], r["head"]) for r in combined_ranked[:EXCLUDE_TOP_N])
all_heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]
candidate_random = [hh for hh in all_heads if hh not in excluded]
print(f"Random-control pool: {len(candidate_random)} heads (excluding top {EXCLUDE_TOP_N} by combined gaze score)")

rand_rng = np.random.RandomState(RANDOM_CONTROL_SEED)
random_heads_by_budget = {}
for budget in HEAD_BUDGETS:
    idx = rand_rng.choice(len(candidate_random), size=budget, replace=False)
    random_heads_by_budget[budget] = [candidate_random[i] for i in idx]
    print(f"  budget {budget}: sampled random heads {random_heads_by_budget[budget]}")


def causal_ablation_effects(model, processor, comic_dirs, n_panels, n_samples, heads_by_layer, n_query_heads, device):
    from transformers import AutoProcessor
    _template_processor = AutoProcessor.from_pretrained(MODELS["instruct"])

    def render_chat_text(prompt):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        rendered = _template_processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
        return rendered[0] if isinstance(rendered, list) else rendered

    def prepare_inputs_any(processor, image, prompt, device):
        text = render_chat_text(prompt)
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        return inputs.to(device)

    rng = np.random.RandomState(DEFAULT_SEED)
    results = []
    for comic_dir in tqdm(comic_dirs[:n_samples], desc="Causal ablation", leave=False):
        strip = build_strip(comic_dir, n_panels=n_panels)
        target_panel = int(rng.randint(n_panels))
        prompt = panel_query_prompt(target_panel + 1, n_panels=n_panels)
        try:
            inputs = prepare_inputs_any(processor, strip.strip, prompt, device)
            img_start, img_end = find_image_token_range(inputs, processor)
            region_ids, _, _ = assign_panels_to_tokens(
                image_grid_thw=inputs["image_grid_thw"], panel_widths=strip.panel_widths,
                spatial_merge=model_dims(model)[2],
            )
            panel_positions = region_positions_from_ids(img_start=img_start, region_ids=region_ids, n_regions=n_panels)
            target_positions = panel_positions[target_panel]
            prompt_length = int(inputs["input_ids"].shape[1])

            baseline_sequences = run_generation(model=model, inputs=inputs, max_new_tokens=CAUSAL_MAX_NEW_TOKENS)
            baseline_text = decode_generated_text(processor, baseline_sequences, prompt_length)

            hook_by_layer = {
                layer_idx: make_static_attention_mask_hook(
                    head_indices=heads, suppress_positions=target_positions, boost_positions=[],
                    n_query_heads=n_query_heads, device=device, decode_only=False, pad_with_suppress=False,
                )
                for layer_idx, heads in heads_by_layer.items()
            }
            ablate_inputs = prepare_inputs_any(processor, strip.strip, prompt, device)
            handles = register_mask_hooks(model, hook_by_layer)
            try:
                ablated_sequences = run_generation(model=model, inputs=ablate_inputs, max_new_tokens=CAUSAL_MAX_NEW_TOKENS)
            finally:
                remove_handles(handles)
            ablated_text = decode_generated_text(processor, ablated_sequences, prompt_length)

            similarity = semantic_similarity(ablated_text, baseline_text, device="cpu")
            results.append({
                "changed": similarity < 0.85, "effect": 1.0 - similarity,
                "baseline_text": baseline_text, "ablated_text": ablated_text,
            })
        except Exception as exc:
            print(f"Skipping {strip.name}: {exc}")
            continue
    return results


gaze_results = {}
random_results = {}
for tag, model_id in MODELS.items():
    print(f"\n=== Loading {model_id} ({tag}) ===")
    model, processor = load_model_and_processor(model_id=model_id, device=DEVICE)
    n_query_heads = model_dims(model)[1]
    gaze_results[tag] = {}
    random_results[tag] = {}
    for budget in HEAD_BUDGETS:
        vis_head = group_heads_by_layer([(r["layer"], r["head"]) for r in combined_ranked[:budget]])
        random_heads = group_heads_by_layer(random_heads_by_budget[budget])
        print(f"--- budget {budget} ---")
        gaze_results[tag][budget] = causal_ablation_effects(
            model, processor, comic_dirs, N_PANELS, N_CAUSAL_SAMPLES, vis_head, n_query_heads, DEVICE)
        random_results[tag][budget] = causal_ablation_effects(
            model, processor, comic_dirs, N_PANELS, N_CAUSAL_SAMPLES, random_heads, n_query_heads, DEVICE)
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

print("\n\n=== RESULTS ===")
print(f"{'heads':>6s}  {'model':>9s}  {'gaze effect':>12s}  {'random effect':>14s}  {'gap':>8s}  {'paired Wilcoxon p':>18s}")
summary_rows = []
for budget in HEAD_BUDGETS:
    for tag in MODELS:
        gaze_effects = [r["effect"] for r in gaze_results[tag][budget]]
        random_effects = [r["effect"] for r in random_results[tag][budget]]
        n = min(len(gaze_effects), len(random_effects))
        gaze_effects, random_effects = gaze_effects[:n], random_effects[:n]
        gap = float(np.mean(gaze_effects) - np.mean(random_effects))
        w = stats.wilcoxon(gaze_effects, random_effects)
        print(f"{budget:6d}  {tag:>9s}  {np.mean(gaze_effects):12.3f}  {np.mean(random_effects):14.3f}  {gap:8.3f}  {w.pvalue:18.3e}")
        summary_rows.append({
            "budget": budget, "tag": tag,
            "gaze_effect_mean": float(np.mean(gaze_effects)),
            "random_effect_mean": float(np.mean(random_effects)),
            "gap": gap, "wilcoxon_p": float(w.pvalue),
        })

dump_json(REPO_ROOT / "logs" / "causal_control_check" / "summary.json", summary_rows)

# a few example transcripts at the smallest (most interpretable) budget, per model
print("\n\n=== EXAMPLE TRANSCRIPTS (budget=5) ===")
for tag in MODELS:
    print(f"\n--- {tag} ---")
    for cond_name, results in (("gaze", gaze_results[tag][5]), ("random", random_results[tag][5])):
        r = results[0]
        print(f"  [{cond_name}] baseline: {r['baseline_text'][:150]!r}")
        print(f"  [{cond_name}] ablated : {r['ablated_text'][:150]!r}")

print("\nDONE")
