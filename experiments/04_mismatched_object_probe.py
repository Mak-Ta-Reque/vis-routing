"""Mismatched-object probe: ask "Find the cat." on a grid that contains NO cat.

Compares three conditions, all scored the same way (since "mismatched" has no
valid ground-truth target cell, we can't use target-based scoring for any of
them here — instead: for each head, how strongly does it concentrate its
image-attention onto its single most-attended cell, right or wrong):

  matched     - "Find the persian cat." on a grid where a cat IS present.
  mismatched  - "Find the cat." on a grid where NO cat (of any breed) is present.
  positional  - "(row, col)" bare-coordinate prompt (never needs an object at all).

Question: does "mismatched" look like "matched" (object-reference heads fire
regardless of whether grounding succeeds) or like "positional" (once there's
no visual referent, engagement falls back to whatever generic "look somewhere"
heads also handle position queries)?
"""
import sys
from pathlib import Path

REPO_ROOT = Path("/mnt/abka03/Projects/vis-head")
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image
from scipy import stats
from tqdm.auto import tqdm

from vis_head.common import DEFAULT_MODEL_ID, DEFAULT_SEED, dump_json
from vis_head.vir import collect_last_query_attentions, rank_heads_by_score
from vis_head.imagenet_grid import DEFAULT_IMAGENET_ROOT, PROMPT_TEMPLATES, bare_row_col_prompt, list_val_class_dirs, load_class_names
from vis_head.modeling import load_model_and_processor, model_dims, prepare_inputs
from vis_head.regions import assign_grid_cells_to_tokens

MODEL_ID = DEFAULT_MODEL_ID
DEVICE = "cuda:0"
SEED = DEFAULT_SEED
ROWS, COLS = 2, 2
N_CELLS = ROWS * COLS
CELL_SIZE = 256
N_SAMPLES = 40
CAT_WNIDS = {"n02124075", "n02123394", "n02123159", "n02123597"}   # true cat breeds only
TARGET_CAT_WNID = "n02123394"   # persian cat, for the "matched" condition

model, processor = load_model_and_processor(model_id=MODEL_ID, device=DEVICE)
n_layers, n_heads, spatial_merge = model_dims(model)
print(f"{n_layers} layers x {n_heads} heads")

imagenet_class_dirs = {d.name: d for d in list_val_class_dirs(DEFAULT_IMAGENET_ROOT)}
imagenet_class_names = load_class_names(DEFAULT_IMAGENET_ROOT)
all_wnids = list(imagenet_class_dirs.keys())
non_cat_wnids = [w for w in all_wnids if w not in CAT_WNIDS]


def _open_rgb_square(path: Path, size: int) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
        w, h = image.size
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
        image = image.crop((left, top, left + side, top + side))
        return image.resize((size, size), Image.LANCZOS)


def build_grid(rng: np.random.RandomState, wnid_pool, forced_wnid: str | None):
    """forced_wnid, if given, occupies a random cell; the rest are drawn from
    wnid_pool. If forced_wnid is None, all N_CELLS cells are drawn from wnid_pool
    (i.e. guaranteed absent, for the mismatched condition)."""
    if forced_wnid is not None:
        target_cell = int(rng.randint(N_CELLS))
        cell_wnids = []
        distractors = []
        while len(distractors) < N_CELLS - 1:
            c = wnid_pool[int(rng.randint(len(wnid_pool)))]
            if c != forced_wnid and c not in distractors:
                distractors.append(c)
        di = 0
        for i in range(N_CELLS):
            if i == target_cell:
                cell_wnids.append(forced_wnid)
            else:
                cell_wnids.append(distractors[di]); di += 1
    else:
        target_cell = None
        cell_wnids = []
        while len(cell_wnids) < N_CELLS:
            c = wnid_pool[int(rng.randint(len(wnid_pool)))]
            if c not in cell_wnids:
                cell_wnids.append(c)

    cell_images = []
    for wnid in cell_wnids:
        class_dir = imagenet_class_dirs[wnid]
        images = sorted(p for p in class_dir.iterdir() if p.suffix.upper() in (".JPEG", ".JPG", ".PNG"))
        image_path = images[int(rng.randint(len(images)))]
        cell_images.append(_open_rgb_square(image_path, CELL_SIZE))

    grid_image = Image.new("RGB", (COLS * CELL_SIZE, ROWS * CELL_SIZE), (255, 255, 255))
    for idx, img in enumerate(cell_images):
        r, c = divmod(idx, COLS)
        grid_image.paste(img, (c * CELL_SIZE, r * CELL_SIZE))
    return grid_image, target_cell, cell_wnids


def peak_concentration_scores(prompt_fn, condition: str, seed: int) -> np.ndarray:
    """mean over samples of (max-attended-cell / total attention over cells),
    per head — how strongly a head commits to SOME single cell, right or wrong."""
    rng = np.random.RandomState(seed)
    peak_sum = np.zeros((n_layers, n_heads), dtype=np.float64)
    valid = 0
    for _ in tqdm(range(N_SAMPLES), desc=f"[{condition}]", leave=False):
        grid_image, prompt = prompt_fn(rng)
        try:
            inputs = prepare_inputs(processor, grid_image, prompt, DEVICE)
            region_ids, _ = assign_grid_cells_to_tokens(
                image_grid_thw=inputs["image_grid_thw"], rows=ROWS, cols=COLS, spatial_merge=spatial_merge)
            attn_at_query = collect_last_query_attentions(model, inputs)   # (n_layers, n_heads, seq_len)
            img_start = 0  # aggregate_region_attention finds its own range via inputs/processor normally;
            from vis_head.modeling import find_image_token_range
            img_start, img_end = find_image_token_range(inputs, processor)
            image_attention = attn_at_query[:, :, img_start:img_end].astype(np.float64)
            usable = min(image_attention.shape[-1], region_ids.shape[0])
            image_attention = image_attention[:, :, :usable]
            region_ids_usable = region_ids[:usable]
            onehot = np.zeros((usable, N_CELLS), dtype=np.float64)
            onehot[np.arange(usable), region_ids_usable] = 1.0
            region_attn = np.einsum("lht,tr->lhr", image_attention, onehot)   # (n_layers, n_heads, N_CELLS)
            total = region_attn.sum(axis=-1)
            peak = region_attn.max(axis=-1)
            peak_frac = peak / np.maximum(total, 1e-8)
            peak_sum += peak_frac
            valid += 1
        except Exception as exc:
            print(f"Skipping: {exc}")
    print(f"  [{condition:>10s}] valid={valid}/{N_SAMPLES}  mean peak-concentration={peak_sum.mean() / max(valid, 1):.4f}")
    return (peak_sum / max(valid, 1)).astype(np.float32)


def matched_prompt_fn(rng):
    grid_image, target_cell, wnids = build_grid(rng, all_wnids, TARGET_CAT_WNID)
    name = imagenet_class_names[TARGET_CAT_WNID]
    return grid_image, PROMPT_TEMPLATES["find"].format(name=name)


def mismatched_prompt_fn(rng):
    grid_image, target_cell, wnids = build_grid(rng, non_cat_wnids, None)
    return grid_image, PROMPT_TEMPLATES["find"].format(name="cat")


def positional_prompt_fn(rng):
    grid_image, target_cell, wnids = build_grid(rng, all_wnids, None)
    target_cell = int(rng.randint(N_CELLS)) if target_cell is None else target_cell
    return grid_image, bare_row_col_prompt(target_cell + 1, COLS)


scores = {}
scores["matched"] = peak_concentration_scores(matched_prompt_fn, "matched", SEED + 700)
scores["mismatched"] = peak_concentration_scores(mismatched_prompt_fn, "mismatched", SEED + 701)
scores["positional"] = peak_concentration_scores(positional_prompt_fn, "positional", SEED + 702)

print("\n=== Pairwise comparison (peak-concentration ranking) ===")
names = list(scores.keys())
for i, a in enumerate(names):
    ranked_a = rank_heads_by_score(scores[a])
    for b in names[i + 1:]:
        ranked_b = rank_heads_by_score(scores[b])
        for K in (10, 50, 100):
            top_a = set((r["layer"], r["head"]) for r in ranked_a[:K])
            top_b = set((r["layer"], r["head"]) for r in ranked_b[:K])
            overlap = len(top_a & top_b) / K
            print(f"  {a:>10s} vs {b:<10s}  top-{K:<4d} overlap: {overlap:.3f}")
        rho = stats.spearmanr(scores[a].reshape(-1), scores[b].reshape(-1)).correlation
        print(f"  {a:>10s} vs {b:<10s}  Spearman rho: {rho:.3f}")
        print()

dump_json(REPO_ROOT / "logs" / "mismatch_check" / "summary.json", {
    "mean_peak_concentration": {k: float(v.mean()) for k, v in scores.items()},
})
print("DONE")
