"""Does prompt word order / filler text matter for vis heads?

Two tests, same ImageNet-grid samples used in the earlier verb/bare-prompt
check (identical seed, so directly comparable):

1. Vir score: original "Find the {name}." vs. a word-shuffled scramble of the
   same words vs. a filler-padded version ("Look at the picture. Find the
   {name}.") — does the head ranking change if the instruction is ungrammatical
   or padded with irrelevant text?
2. Causal effect: ablate the (already-identified) top ImageNet-grid vis heads
   and measure how much the model's answer changes, under each of the three
   phrasings — does steerability survive scrambled/padded phrasing?
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


def shuffled_prompt(name: str, sample_seed: int) -> str:
    """Word-order scramble of "Find the {name}." — same words, random order,
    deterministic per sample so results are reproducible."""
    words = f"Find the {name}.".rstrip(".").split(" ")
    rng = np.random.RandomState(sample_seed)
    perm = rng.permutation(len(words))
    return " ".join(words[i] for i in perm) + "."


def filler_prompt(name: str) -> str:
    return f"Look at the picture. Find the {name}."


model, processor = load_model_and_processor(model_id=MODEL_ID, device=DEVICE)
n_layers, n_heads, spatial_merge = model_dims(model)
print(f"{n_layers} layers x {n_heads} heads")

imagenet_class_dirs = list_val_class_dirs(DEFAULT_IMAGENET_ROOT)
imagenet_class_names = load_class_names(DEFAULT_IMAGENET_ROOT)

verb_rng = np.random.RandomState(SEED + 2)
fixed_grids = []
for _ in range(N_GRIDS):
    grid = sample_grid(rows=ROWS, cols=COLS, cell_size=CELL_SIZE, rng=verb_rng,
                        class_dirs=imagenet_class_dirs, class_names=imagenet_class_names)
    target_cell = int(verb_rng.randint(N_CELLS))
    fixed_grids.append((grid, target_cell))

prompt_fns = {
    "original": lambda grid, target_cell, i: PROMPT_TEMPLATES["find"].format(name=grid.cell_names[target_cell]),
    "shuffled": lambda grid, target_cell, i: shuffled_prompt(grid.cell_names[target_cell], SEED + 1000 + i),
    "filler": lambda grid, target_cell, i: filler_prompt(grid.cell_names[target_cell]),
}

print("\nExample prompts (sample 0):")
g0, t0 = fixed_grids[0]
for name, fn in prompt_fns.items():
    print(f"  [{name:>9s}] {fn(g0, t0, 0)!r}")


# ---------------------------- Part 1: vir score ----------------------------
def discover_for_prompt_fn(prompt_fn, label: str) -> np.ndarray:
    raw_sum = np.zeros((n_layers, n_heads), dtype=np.float64)
    valid = 0
    for i, (grid, target_cell) in enumerate(tqdm(fixed_grids, desc=f"[{label}]", leave=False)):
        prompt = prompt_fn(grid, target_cell, i)
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
    print(f"  [{label:>9s}] valid={valid}/{N_GRIDS}  mean score={raw_sum.mean() / max(valid, 1):.5f}")
    return (raw_sum / max(valid, 1)).astype(np.float32)


scores = {name: discover_for_prompt_fn(fn, name) for name, fn in prompt_fns.items()}

print("\n=== Vir-score comparison ===")
names = list(scores.keys())
K = 50
for a in names:
    ranked_a = rank_heads_by_score(scores[a])
    top_a = set((r["layer"], r["head"]) for r in ranked_a[:K])
    for b in names:
        if b <= a:
            continue
        ranked_b = rank_heads_by_score(scores[b])
        top_b = set((r["layer"], r["head"]) for r in ranked_b[:K])
        overlap = len(top_a & top_b) / K
        rho = stats.spearmanr(scores[a].reshape(-1), scores[b].reshape(-1)).correlation
        print(f"  {a:>9s} vs {b:<9s}: top-{K} overlap={overlap:.3f}  Spearman rho={rho:.3f}  "
              f"mean score {scores[a].mean():.5f} vs {scores[b].mean():.5f}")


# ---------------------------- Part 2: causal effect ----------------------------
print("\n=== Causal-effect check (ablate top imagenet vis heads under each phrasing) ===")
imagenet_ranking = json.loads((REPO_ROOT / "logs" / "vis_head_discovery_compare_datasets_imagenet" / "vis_head_ranking.json").read_text())
top_heads = [(r["layer"], r["head"]) for r in imagenet_ranking[:N_CAUSAL_HEADS]]
heads_by_layer = group_heads_by_layer(top_heads)
print(f"Ablating {N_CAUSAL_HEADS} pre-identified ImageNet-grid vis heads across {len(heads_by_layer)} layers.")

causal_results = {name: [] for name in prompt_fns}
for i, (grid, target_cell) in enumerate(tqdm(fixed_grids, desc="Causal ablation")):
    for name, fn in prompt_fns.items():
        prompt = fn(grid, target_cell, i)
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
            print(f"Skipping grid {i} [{name}]: {exc}")

print(f"\n{'phrasing':>9s}  {'n':>4s}  {'change rate':>12s}  {'95% CI':>16s}  {'mean effect':>12s}")
for name in prompt_fns:
    res = causal_results[name]
    ci = bootstrap_ci([r["changed"] for r in res])
    mean_effect = np.mean([r["effect"] for r in res])
    print(f"{name:>9s}  {ci['n']:4d}  {ci['accuracy']:12.3f}  [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]  {mean_effect:12.3f}")

for a, b in [("original", "shuffled"), ("original", "filler")]:
    ea = [r["effect"] for r in causal_results[a]]
    eb = [r["effect"] for r in causal_results[b]]
    n = min(len(ea), len(eb))
    w = stats.wilcoxon(ea[:n], eb[:n])
    print(f"Paired Wilcoxon ({a} vs {b} causal effect): statistic={w.statistic:.1f}  p={w.pvalue:.3e}")

dump_json(REPO_ROOT / "logs" / "word_order_check" / "summary.json", {
    "vis_head_score_means": {k: float(v.mean()) for k, v in scores.items()},
    "causal_mean_effect": {k: float(np.mean([r["effect"] for r in v])) for k, v in causal_results.items()},
})
print("\nDONE")
