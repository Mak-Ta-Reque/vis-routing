"""Is the named-object / positional vir-head split actually about DIGIT
TOKENS (many tokenizers give numbers special handling), or genuinely about
spatial-vs-semantic reference type?

Four conditions on the same fixed ImageNet-grid samples:
  object_no_digit     - "Find the {name}."                          (anchor: named-object, no digit)
  position_digit      - "(row, col)"                                (anchor: positional, has digits)
  position_no_digit   - "top-left" / "bottom-right" / ...            (positional meaning, NO digits)
  object_with_digit   - "Find the {name} in image 1."                (named-object meaning, irrelevant digit)

If digits are the real driver: position_no_digit should shift toward the
object cluster, and object_with_digit should shift toward the positional
cluster.
If reference TYPE is the real driver (as hypothesized): position_no_digit
should still cluster with position_digit, and object_with_digit should still
cluster with object_no_digit, despite the digit swap.
"""
import sys
from pathlib import Path

REPO_ROOT = Path("/mnt/abka03/Projects/vis-head")
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from scipy import stats
from tqdm.auto import tqdm

from vis_head.common import DEFAULT_MODEL_ID, DEFAULT_SEED, dump_json
from vis_head.vir import aggregate_region_attention, collect_last_query_attentions, rank_heads_by_score
from vis_head.imagenet_grid import (
    DEFAULT_IMAGENET_ROOT, PROMPT_TEMPLATES, bare_row_col_prompt, list_val_class_dirs,
    load_class_names, position_words_prompt, sample_grid,
)
from vis_head.modeling import load_model_and_processor, model_dims, prepare_inputs
from vis_head.regions import assign_grid_cells_to_tokens

MODEL_ID = DEFAULT_MODEL_ID
DEVICE = "cuda:0"
SEED = DEFAULT_SEED
ROWS, COLS = 2, 2
N_CELLS = ROWS * COLS
CELL_SIZE = 256
N_GRIDS = 30

model, processor = load_model_and_processor(model_id=MODEL_ID, device=DEVICE)
n_layers, n_heads, spatial_merge = model_dims(model)
print(f"{n_layers} layers x {n_heads} heads")

imagenet_class_dirs = list_val_class_dirs(DEFAULT_IMAGENET_ROOT)
imagenet_class_names = load_class_names(DEFAULT_IMAGENET_ROOT)

# Same seed/grid-sampling sequence as the earlier verb/bare-prompt tests.
verb_rng = np.random.RandomState(SEED + 2)
fixed_grids = []
for _ in range(N_GRIDS):
    grid = sample_grid(rows=ROWS, cols=COLS, cell_size=CELL_SIZE, rng=verb_rng,
                        class_dirs=imagenet_class_dirs, class_names=imagenet_class_names)
    target_cell = int(verb_rng.randint(N_CELLS))
    fixed_grids.append((grid, target_cell))

prompt_fns = {
    "object_no_digit": lambda grid, tc: PROMPT_TEMPLATES["find"].format(name=grid.cell_names[tc]),
    "position_digit": lambda grid, tc: bare_row_col_prompt(tc + 1, COLS),
    "position_no_digit": lambda grid, tc: position_words_prompt(tc + 1, ROWS, COLS),
    "object_with_digit": lambda grid, tc: f"Find the {grid.cell_names[tc]} in image 1.",
}

print("\nExample prompts (sample 0):")
g0, t0 = fixed_grids[0]
for name, fn in prompt_fns.items():
    print(f"  [{name:>18s}] {fn(g0, t0)!r}")


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
    print(f"  [{label:>18s}] valid={valid}/{N_GRIDS}  mean score={raw_sum.mean() / max(valid, 1):.5f}")
    return (raw_sum / max(valid, 1)).astype(np.float32)


scores = {name: discover_for_prompt_fn(fn, name) for name, fn in prompt_fns.items()}

names = list(scores.keys())
K = 50
print(f"\n=== Pairwise top-{K} overlap / Spearman rho ===")
for i, a in enumerate(names):
    ranked_a = rank_heads_by_score(scores[a])
    top_a = set((r["layer"], r["head"]) for r in ranked_a[:K])
    for b in names[i + 1:]:
        ranked_b = rank_heads_by_score(scores[b])
        top_b = set((r["layer"], r["head"]) for r in ranked_b[:K])
        overlap = len(top_a & top_b) / K
        rho = stats.spearmanr(scores[a].reshape(-1), scores[b].reshape(-1)).correlation
        print(f"  {a:>18s} vs {b:<18s}  overlap={overlap:.3f}  rho={rho:.3f}")

print("\n=== Verdict ===")
def ov(a, b):
    ranked_a = rank_heads_by_score(scores[a]); ranked_b = rank_heads_by_score(scores[b])
    top_a = set((r["layer"], r["head"]) for r in ranked_a[:K]); top_b = set((r["layer"], r["head"]) for r in ranked_b[:K])
    return len(top_a & top_b) / K

print(f"position_no_digit vs its TYPE match (position_digit):     {ov('position_no_digit', 'position_digit'):.3f}")
print(f"position_no_digit vs its DIGIT-presence match (object_no_digit... wait, no digit either -- compare to object_with_digit): {ov('position_no_digit', 'object_with_digit'):.3f}")
print(f"object_with_digit vs its TYPE match (object_no_digit):     {ov('object_with_digit', 'object_no_digit'):.3f}")
print(f"object_with_digit vs its DIGIT-presence match (position_digit): {ov('object_with_digit', 'position_digit'):.3f}")

dump_json(REPO_ROOT / "logs" / "digit_confound_check" / "summary.json", {
    "mean_scores": {k: float(v.mean()) for k, v in scores.items()},
})
print("\nDONE")
