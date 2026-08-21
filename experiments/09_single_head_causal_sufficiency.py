"""Per-head causal SUFFICIENCY score: for each candidate head, boost it alone
toward panel A vs. panel B (using a NEUTRAL prompt that never names a panel,
so the boosted head is the only signal determining what gets described) and
measure how much the output actually changes between the two targets.

This is not a proxy for causal importance -- it IS a causal measurement
(direct intervention + outcome), just restricted to a candidate pool (testing
all 1152 heads individually is 1152x the cost of testing the top-40) and
scored with the semantic-similarity effect size used throughout this project.

Compares this single-head sufficiency ranking against the original raw-
attention ranking, and cross-validates: does the sufficiency ranking's top-15
also show a larger GROUP ablation-necessity effect (the standard test used
everywhere else in this project) than raw attention's top-15?
"""
import sys
from pathlib import Path

REPO_ROOT = Path("/mnt/abka03/Projects/vis-head")
sys.path.insert(0, str(REPO_ROOT))

import json
import numpy as np
from scipy import stats
from tqdm.auto import tqdm

from vis_head.common import DEFAULT_COMICS_ROOT, DEFAULT_MODEL_ID, DEFAULT_N_PANELS, DEFAULT_SEED, dump_json
from vis_head.data import build_strip, list_comic_dirs
from vis_head.vir import panel_query_prompt, rank_heads_by_score
from vis_head.judge import bootstrap_ci, semantic_similarity
from vis_head.modeling import decode_generated_text, find_image_token_range, load_model_and_processor, model_dims, prepare_inputs, run_generation
from vis_head.regions import assign_panels_to_tokens, region_positions_from_ids
from vis_head.steering import make_static_attention_mask_hook, register_mask_hooks, remove_handles

MODEL_ID = DEFAULT_MODEL_ID
DEVICE = "cuda:0"
SEED = DEFAULT_SEED
N_PANELS = DEFAULT_N_PANELS
N_CANDIDATE_HEADS = 40    # pool tested individually (from the raw top-100)
N_SUFFICIENCY_SAMPLES = 10
CAUSAL_HEAD_BUDGET = 15
N_CAUSAL_SAMPLES = 30
MAX_NEW_TOKENS = 40
NEUTRAL_PROMPT = "Describe what is happening in this comic strip. Answer briefly."

model, processor = load_model_and_processor(model_id=MODEL_ID, device=DEVICE)
n_layers, n_heads, spatial_merge = model_dims(model)
print(f"{n_layers} layers x {n_heads} heads")

comic_dirs = list_comic_dirs(Path(DEFAULT_COMICS_ROOT), n_panels=N_PANELS)
raw_ranking = json.loads((REPO_ROOT / "logs" / "vis_head_discovery_compare_datasets_comics" / "vis_head_ranking.json").read_text())
candidate_heads = [(r["layer"], r["head"]) for r in raw_ranking[:N_CANDIDATE_HEADS]]
print(f"Testing {len(candidate_heads)} candidate heads (raw top-{N_CANDIDATE_HEADS}) for single-head sufficiency.")

suff_rng = np.random.RandomState(SEED)
sufficiency_samples = []   # (comic_dir, panel_a, panel_b)
for comic_dir in comic_dirs[:N_SUFFICIENCY_SAMPLES]:
    a, b = suff_rng.choice(N_PANELS, size=2, replace=False)
    sufficiency_samples.append((comic_dir, int(a), int(b)))

sufficiency_scores = {}
for layer_idx, head_idx in tqdm(candidate_heads, desc="Single-head sufficiency"):
    effects = []
    for comic_dir, panel_a, panel_b in sufficiency_samples:
        strip = build_strip(comic_dir, n_panels=N_PANELS)
        try:
            texts = {}
            for panel in (panel_a, panel_b):
                inputs = prepare_inputs(processor, strip.strip, NEUTRAL_PROMPT, DEVICE)
                img_start, img_end = find_image_token_range(inputs, processor)
                region_ids, _, _ = assign_panels_to_tokens(
                    image_grid_thw=inputs["image_grid_thw"], panel_widths=strip.panel_widths, spatial_merge=spatial_merge)
                positions = region_positions_from_ids(img_start=img_start, region_ids=region_ids, n_regions=N_PANELS)
                target_positions = positions[panel]
                other_positions = [p for i in range(N_PANELS) if i != panel for p in positions[i]]
                prompt_length = int(inputs["input_ids"].shape[1])

                hook = make_static_attention_mask_hook(
                    head_indices=[head_idx], suppress_positions=other_positions, boost_positions=target_positions,
                    n_query_heads=n_heads, device=DEVICE, decode_only=False, pad_with_suppress=False)
                handles = register_mask_hooks(model, {layer_idx: hook})
                try:
                    sequences = run_generation(model=model, inputs=inputs, max_new_tokens=MAX_NEW_TOKENS)
                finally:
                    remove_handles(handles)
                texts[panel] = decode_generated_text(processor, sequences, prompt_length)

            similarity = semantic_similarity(texts[panel_a], texts[panel_b], device="cpu")
            effects.append(1.0 - similarity)
        except Exception as exc:
            print(f"Skipping {strip.name} for head ({layer_idx},{head_idx}): {exc}")
    sufficiency_scores[(layer_idx, head_idx)] = float(np.mean(effects)) if effects else 0.0

sorted_by_sufficiency = sorted(sufficiency_scores.items(), key=lambda kv: kv[1], reverse=True)
print("\n=== Top-15 heads by single-head sufficiency (boosting alone flips described panel) ===")
for (l, h), s in sorted_by_sufficiency[:15]:
    print(f"  layer={l:2d} head={h:2d}  sufficiency={s:.3f}")

raw_rank_of = {(r["layer"], r["head"]): i for i, r in enumerate(raw_ranking)}
print("\nWhere do the top sufficiency heads rank in the ORIGINAL raw-attention ranking?")
for (l, h), s in sorted_by_sufficiency[:15]:
    print(f"  ({l},{h})  sufficiency={s:.3f}  raw_attention_rank={raw_rank_of.get((l, h), '???')}")

suff_top15 = [k for k, _ in sorted_by_sufficiency[:CAUSAL_HEAD_BUDGET]]
raw_top15 = candidate_heads[:CAUSAL_HEAD_BUDGET]
overlap = len(set(suff_top15) & set(raw_top15))
print(f"\nTop-15 overlap between sufficiency ranking and raw-attention ranking (within the top-{N_CANDIDATE_HEADS} pool): {overlap}/15")

# ---------------------------- cross-validate with group ablation-necessity ----------------------------
print("\n=== Cross-validation: group ablation-necessity effect of each top-15 set ===")


def group_causal_effects(heads_list, label):
    from vis_head.steering import group_heads_by_layer
    heads_by_layer = group_heads_by_layer(heads_list)
    rng = np.random.RandomState(SEED)
    results = []
    for comic_dir in tqdm(comic_dirs[:N_CAUSAL_SAMPLES], desc=f"Group ablation [{label}]", leave=False):
        strip = build_strip(comic_dir, n_panels=N_PANELS)
        target_panel = int(rng.randint(N_PANELS))
        prompt = panel_query_prompt(target_panel + 1, n_panels=N_PANELS)
        try:
            inputs = prepare_inputs(processor, strip.strip, prompt, DEVICE)
            img_start, img_end = find_image_token_range(inputs, processor)
            region_ids, _, _ = assign_panels_to_tokens(
                image_grid_thw=inputs["image_grid_thw"], panel_widths=strip.panel_widths, spatial_merge=spatial_merge)
            positions = region_positions_from_ids(img_start=img_start, region_ids=region_ids, n_regions=N_PANELS)
            target_positions = positions[target_panel]
            prompt_length = int(inputs["input_ids"].shape[1])

            baseline_sequences = run_generation(model=model, inputs=inputs, max_new_tokens=MAX_NEW_TOKENS)
            baseline_text = decode_generated_text(processor, baseline_sequences, prompt_length)

            hook_by_layer = {
                layer_idx: make_static_attention_mask_hook(
                    head_indices=heads, suppress_positions=target_positions, boost_positions=[],
                    n_query_heads=n_heads, device=DEVICE, decode_only=False, pad_with_suppress=False)
                for layer_idx, heads in heads_by_layer.items()
            }
            ablate_inputs = prepare_inputs(processor, strip.strip, prompt, DEVICE)
            handles = register_mask_hooks(model, hook_by_layer)
            try:
                ablated_sequences = run_generation(model=model, inputs=ablate_inputs, max_new_tokens=MAX_NEW_TOKENS)
            finally:
                remove_handles(handles)
            ablated_text = decode_generated_text(processor, ablated_sequences, prompt_length)
            similarity = semantic_similarity(ablated_text, baseline_text, device="cpu")
            results.append({"changed": similarity < 0.85, "effect": 1.0 - similarity})
        except Exception as exc:
            print(f"Skipping {strip.name}: {exc}")
    return results


res_raw = group_causal_effects(raw_top15, "raw")
res_suff = group_causal_effects(suff_top15, "sufficiency")

print(f"\n{'ranking':>12s}  {'n':>4s}  {'change rate':>12s}  {'95% CI':>16s}  {'mean effect':>12s}")
for label, res in (("raw", res_raw), ("sufficiency", res_suff)):
    ci = bootstrap_ci([r["changed"] for r in res])
    print(f"{label:>12s}  {ci['n']:4d}  {ci['accuracy']:12.3f}  [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]  "
          f"{np.mean([r['effect'] for r in res]):12.3f}")

n = min(len(res_raw), len(res_suff))
w = stats.wilcoxon([r["effect"] for r in res_raw][:n], [r["effect"] for r in res_suff][:n])
print(f"Paired Wilcoxon (raw-top15 vs sufficiency-top15 group ablation effect): statistic={w.statistic:.1f}  p={w.pvalue:.3e}")

dump_json(REPO_ROOT / "logs" / "single_head_sufficiency" / "summary.json", {
    "sufficiency_scores": {f"{l}_{h}": s for (l, h), s in sufficiency_scores.items()},
    "top15_overlap_with_raw": overlap,
    "group_causal_raw": float(np.mean([r["effect"] for r in res_raw])),
    "group_causal_sufficiency": float(np.mean([r["effect"] for r in res_suff])),
})
print("\nDONE")
