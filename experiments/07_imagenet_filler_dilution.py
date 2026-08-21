"""Closes the filler-discrepancy gap: add a short "Look carefully at this
picture." preamble to the ImageNet-grid prompt (mirroring the comics
prompt's own opening, "Look carefully at this six-panel comic strip.")
instead of the earlier "Look at the picture." filler, and check vis-head-score
identity + causal effect against the bare "Find the {name}." baseline.
"""
import sys
from pathlib import Path

REPO_ROOT = Path("/mnt/abka03/Projects/vis-head")
sys.path.insert(0, str(REPO_ROOT))

import json
import numpy as np
from scipy import stats
from tqdm.auto import tqdm

from vis_head.common import DEFAULT_MODEL_ID, DEFAULT_SEED, dump_json
from vis_head.vir import aggregate_region_attention, collect_last_query_attentions, rank_heads_by_score
from vis_head.imagenet_grid import DEFAULT_IMAGENET_ROOT, PROMPT_TEMPLATES, list_val_class_dirs, load_class_names, sample_grid
from vis_head.judge import bootstrap_ci, semantic_similarity
from vis_head.modeling import decode_generated_text, find_image_token_range, load_model_and_processor, model_dims, prepare_inputs, run_generation
from vis_head.regions import assign_grid_cells_to_tokens, region_positions_from_ids
from vis_head.steering import group_heads_by_layer, make_static_attention_mask_hook, register_mask_hooks, remove_handles

MODEL_ID = DEFAULT_MODEL_ID
DEVICE = "cuda:0"
SEED = DEFAULT_SEED
ROWS, COLS = 2, 2
N_CELLS = ROWS * COLS
CELL_SIZE = 256
N_GRIDS = 30
N_CAUSAL_HEADS = 15
CAUSAL_MAX_NEW_TOKENS = 40


def original_prompt(name: str) -> str:
    return PROMPT_TEMPLATES["find"].format(name=name)


def look_carefully_prompt(name: str) -> str:
    return f"Look carefully at this picture. Find the {name}."


model, processor = load_model_and_processor(model_id=MODEL_ID, device=DEVICE)
n_layers, n_heads, spatial_merge = model_dims(model)
print(f"{n_layers} layers x {n_heads} heads")

imagenet_class_dirs = list_val_class_dirs(DEFAULT_IMAGENET_ROOT)
imagenet_class_names = load_class_names(DEFAULT_IMAGENET_ROOT)

verb_rng = np.random.RandomState(SEED + 2)   # same fixed-grid sequence as all earlier prompt-comparison tests
fixed_grids = []
for _ in range(N_GRIDS):
    grid = sample_grid(rows=ROWS, cols=COLS, cell_size=CELL_SIZE, rng=verb_rng,
                        class_dirs=imagenet_class_dirs, class_names=imagenet_class_names)
    target_cell = int(verb_rng.randint(N_CELLS))
    fixed_grids.append((grid, target_cell))

prompt_fns = {"original": original_prompt, "look_carefully": look_carefully_prompt}

print("\nExample prompts (sample 0):")
g0, t0 = fixed_grids[0]
for name, fn in prompt_fns.items():
    print(f"  [{name:>15s}] {fn(g0.cell_names[t0])!r}")


# ---------------------------- Part 1: vir score ----------------------------
def discover_for_prompt_fn(prompt_fn, label: str) -> np.ndarray:
    raw_sum = np.zeros((n_layers, n_heads), dtype=np.float64)
    valid = 0
    for grid, target_cell in tqdm(fixed_grids, desc=f"[{label}]", leave=False):
        prompt = prompt_fn(grid.cell_names[target_cell])
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
    print(f"  [{label:>15s}] valid={valid}/{N_GRIDS}  mean score={raw_sum.mean() / max(valid, 1):.5f}")
    return (raw_sum / max(valid, 1)).astype(np.float32)


scores = {name: discover_for_prompt_fn(fn, name) for name, fn in prompt_fns.items()}
ranked_o, ranked_lc = rank_heads_by_score(scores["original"]), rank_heads_by_score(scores["look_carefully"])
print("\n=== Vir-score comparison ===")
for K in (10, 50, 100):
    top_o = set((r["layer"], r["head"]) for r in ranked_o[:K])
    top_lc = set((r["layer"], r["head"]) for r in ranked_lc[:K])
    print(f"  top-{K:<4d} overlap: {len(top_o & top_lc) / K:.3f}")
rho = stats.spearmanr(scores["original"].reshape(-1), scores["look_carefully"].reshape(-1))
print(f"  Spearman rho: {rho.correlation:.3f}  (p={rho.pvalue:.3e})")
print(f"  Mean score: original={scores['original'].mean():.5f}  look_carefully={scores['look_carefully'].mean():.5f}")


# ---------------------------- Part 2: causal effect ----------------------------
print("\n=== Causal-effect check (ablate top imagenet vis heads under each phrasing) ===")
imagenet_ranking = json.loads((REPO_ROOT / "logs" / "vis_head_discovery_compare_datasets_imagenet" / "vis_head_ranking.json").read_text())
top_heads = [(r["layer"], r["head"]) for r in imagenet_ranking[:N_CAUSAL_HEADS]]
heads_by_layer = group_heads_by_layer(top_heads)
print(f"Ablating {N_CAUSAL_HEADS} pre-identified ImageNet-grid vis heads across {len(heads_by_layer)} layers.")

causal_results = {name: [] for name in prompt_fns}
for grid, target_cell in tqdm(fixed_grids, desc="Causal ablation"):
    for name, fn in prompt_fns.items():
        prompt = fn(grid.cell_names[target_cell])
        try:
            inputs = prepare_inputs(processor, grid.grid, prompt, DEVICE)
            img_start, img_end = find_image_token_range(inputs, processor)
            region_ids, _ = assign_grid_cells_to_tokens(
                image_grid_thw=inputs["image_grid_thw"], rows=ROWS, cols=COLS, spatial_merge=spatial_merge)
            positions = region_positions_from_ids(img_start=img_start, region_ids=region_ids, n_regions=N_CELLS)
            target_positions = positions[target_cell]
            prompt_length = int(inputs["input_ids"].shape[1])

            baseline_sequences = run_generation(model=model, inputs=inputs, max_new_tokens=CAUSAL_MAX_NEW_TOKENS)
            baseline_text = decode_generated_text(processor, baseline_sequences, prompt_length)

            hook_by_layer = {
                layer_idx: make_static_attention_mask_hook(
                    head_indices=heads, suppress_positions=target_positions, boost_positions=[],
                    n_query_heads=n_heads, device=DEVICE, decode_only=False, pad_with_suppress=False,
                )
                for layer_idx, heads in heads_by_layer.items()
            }
            ablate_inputs = prepare_inputs(processor, grid.grid, prompt, DEVICE)
            handles = register_mask_hooks(model, hook_by_layer)
            try:
                ablated_sequences = run_generation(model=model, inputs=ablate_inputs, max_new_tokens=CAUSAL_MAX_NEW_TOKENS)
            finally:
                remove_handles(handles)
            ablated_text = decode_generated_text(processor, ablated_sequences, prompt_length)

            similarity = semantic_similarity(ablated_text, baseline_text, device="cpu")
            causal_results[name].append({"changed": similarity < 0.85, "effect": 1.0 - similarity})
        except Exception as exc:
            print(f"Skipping grid [{name}]: {exc}")

print(f"\n{'phrasing':>15s}  {'n':>4s}  {'change rate':>12s}  {'95% CI':>16s}  {'mean effect':>12s}")
for name in prompt_fns:
    res = causal_results[name]
    ci = bootstrap_ci([r["changed"] for r in res])
    mean_effect = np.mean([r["effect"] for r in res])
    print(f"{name:>15s}  {ci['n']:4d}  {ci['accuracy']:12.3f}  [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]  {mean_effect:12.3f}")

n = min(len(causal_results["original"]), len(causal_results["look_carefully"]))
ea = [r["effect"] for r in causal_results["original"][:n]]
eb = [r["effect"] for r in causal_results["look_carefully"][:n]]
w = stats.wilcoxon(ea, eb)
print(f"Paired Wilcoxon (original vs look_carefully causal effect): statistic={w.statistic:.1f}  p={w.pvalue:.3e}")

dump_json(REPO_ROOT / "logs" / "imagenet_look_carefully_check" / "summary.json", {
    "vis_head_score_means": {k: float(v.mean()) for k, v in scores.items()},
    "causal_mean_effect": {k: float(np.mean([r["effect"] for r in v])) for k, v in causal_results.items()},
})
print("\nDONE")
