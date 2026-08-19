"""Bare, non-instructional referring prompts vs. verb-phrased instructions,
on the same ImageNet-grid samples used in imagenet_grid_vis_heads.ipynb Part 6.

Tests whether vis heads need a task framing ("Find the X", "Where is the X?")
or fire on the bare reference alone: just the object name, just the cell
number, or just the (row, col) coordinate — no verb, no instruction.
"""
import sys
from pathlib import Path

REPO_ROOT = Path("/mnt/abka03/Projects/vis-head")
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from scipy import stats
from tqdm.auto import tqdm

from vis_head.common import DEFAULT_MODEL_ID, DEFAULT_SEED, dump_json
from vis_head.gaze import aggregate_region_attention, collect_last_query_attentions, rank_heads_by_score
from vis_head.imagenet_grid import (
    DEFAULT_IMAGENET_ROOT, PROMPT_TEMPLATES, bare_cell_prompt, bare_row_col_prompt,
    list_val_class_dirs, load_class_names, ordinal_prompt, sample_grid,
)
from vis_head.modeling import load_model_and_processor, model_dims, prepare_inputs
from vis_head.regions import assign_grid_cells_to_tokens

MODEL_ID = DEFAULT_MODEL_ID
DEVICE = "cuda:0"
SEED = DEFAULT_SEED
ROWS, COLS = 2, 2
N_CELLS = ROWS * COLS
CELL_SIZE = 256
N_VERB_GRIDS = 30

model, processor = load_model_and_processor(model_id=MODEL_ID, device=DEVICE)
n_layers, n_heads, spatial_merge = model_dims(model)
print(f"{n_layers} layers x {n_heads} heads")

imagenet_class_dirs = list_val_class_dirs(DEFAULT_IMAGENET_ROOT)
imagenet_class_names = load_class_names(DEFAULT_IMAGENET_ROOT)

# Same seed/grid-sampling sequence as the notebook's Part 6, so these grids and
# targets are identical to the ones the verb-phrased templates were tested on.
verb_rng = np.random.RandomState(SEED + 2)
fixed_grids = []
for _ in range(N_VERB_GRIDS):
    grid = sample_grid(rows=ROWS, cols=COLS, cell_size=CELL_SIZE, rng=verb_rng,
                        class_dirs=imagenet_class_dirs, class_names=imagenet_class_names)
    target_cell = int(verb_rng.randint(N_CELLS))
    fixed_grids.append((grid, target_cell))


def discover_for_prompt_fn(prompt_fn, label: str) -> np.ndarray:
    raw_sum = np.zeros((n_layers, n_heads), dtype=np.float64)
    valid = 0
    for grid, target_cell in tqdm(fixed_grids, desc=f"[{label}]", leave=False):
        prompt = prompt_fn(grid, target_cell)
        try:
            inputs = prepare_inputs(processor, grid.grid, prompt, DEVICE)
            region_ids, _ = assign_grid_cells_to_tokens(
                image_grid_thw=inputs["image_grid_thw"], rows=ROWS, cols=COLS, spatial_merge=spatial_merge)
            attn_at_query = collect_last_query_attentions(model, inputs)
            region_attention = aggregate_region_attention(
                attn_at_query=attn_at_query, inputs=inputs, processor=processor, region_ids=region_ids, n_regions=N_CELLS)
            raw_sum += region_attention[:, :, target_cell]
            valid += 1
        except Exception as exc:
            print(f"Skipping grid: {exc}")
    print(f"  [{label:>18s}] valid={valid}/{N_VERB_GRIDS}  mean score={raw_sum.sum() / max(valid, 1) / (n_layers * n_heads):.5f}")
    return (raw_sum / max(valid, 1)).astype(np.float32)


prompt_variants = {}
for template in PROMPT_TEMPLATES:
    prompt_variants[template] = discover_for_prompt_fn(
        lambda grid, target_cell, t=template: PROMPT_TEMPLATES[t].format(name=grid.cell_names[target_cell]), template)

prompt_variants["ordinal_sentence"] = discover_for_prompt_fn(
    lambda grid, target_cell: ordinal_prompt(target_cell + 1, N_CELLS), "ordinal_sentence")
prompt_variants["bare_cell_number"] = discover_for_prompt_fn(
    lambda grid, target_cell: bare_cell_prompt(target_cell + 1), "bare_cell_number")
prompt_variants["bare_row_col"] = discover_for_prompt_fn(
    lambda grid, target_cell: bare_row_col_prompt(target_cell + 1, COLS), "bare_row_col")

names = list(prompt_variants.keys())
K = 50
print("\n=== Pairwise top-50 overlap ===")
header = "  ".join(f"{n:>18s}" for n in names)
print(f"{'':>18s}  {header}")
overlap_rows = {}
for a in names:
    ranked_a = rank_heads_by_score(prompt_variants[a])
    top_a = set((r["layer"], r["head"]) for r in ranked_a[:K])
    row = []
    for b in names:
        ranked_b = rank_heads_by_score(prompt_variants[b])
        top_b = set((r["layer"], r["head"]) for r in ranked_b[:K])
        row.append(len(top_a & top_b) / K)
    overlap_rows[a] = row
    row_str = "  ".join(f"{v:18.2f}" for v in row)
    print(f"{a:>18s}  {row_str}")

print("\n=== Pairwise Spearman rho ===")
print(f"{'':>18s}  {header}")
rho_rows = {}
for a in names:
    row = []
    for b in names:
        row.append(stats.spearmanr(prompt_variants[a].reshape(-1), prompt_variants[b].reshape(-1)).correlation)
    rho_rows[a] = row
    row_str = "  ".join(f"{v:18.3f}" for v in row)
    print(f"{a:>18s}  {row_str}")

named_object = ["find", "locate", "where_is", "point_to", "identify"]
bare_named = ["bare_name"]
positional = ["ordinal_sentence", "bare_cell_number", "bare_row_col"]


def mean_off_diag(group_a, group_b):
    vals = []
    for a in group_a:
        for b in group_b:
            if a == b:
                continue
            vals.append(overlap_rows[a][names.index(b)])
    return float(np.mean(vals)) if vals else float("nan")


print("\n=== Summary ===")
print(f"Named-object VERB phrasings (find/locate/where_is/point_to/identify), mean pairwise top-{K} overlap: "
      f"{mean_off_diag(named_object, named_object):.3f}")
print(f"Named-object verb phrasings  vs  bare object name only: {mean_off_diag(named_object, bare_named):.3f}")
print(f"Named-object verb phrasings  vs  positional (ordinal sentence + bare cell/coord): {mean_off_diag(named_object, positional):.3f}")
print(f"Bare object name             vs  positional prompts: {mean_off_diag(bare_named, positional):.3f}")
print(f"Ordinal sentence             vs  bare cell number:    {overlap_rows['ordinal_sentence'][names.index('bare_cell_number')]:.3f}")
print(f"Ordinal sentence             vs  bare (row, col):     {overlap_rows['ordinal_sentence'][names.index('bare_row_col')]:.3f}")
print(f"Bare cell number             vs  bare (row, col):     {overlap_rows['bare_cell_number'][names.index('bare_row_col')]:.3f}")

dump_json(REPO_ROOT / "logs" / "bare_prompt_check" / "overlap_matrix.json", {
    "names": names, "overlap": overlap_rows, "rho": rho_rows,
})
print("\nDONE")
