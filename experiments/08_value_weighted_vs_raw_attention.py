"""Value-weighted vs. raw-attention gaze scoring, head to head.

Same comics, same panel queries, both metrics computed in the same pass set:
  raw    - attention weight on the queried panel's image tokens (current method)
  vw     - that attention weighted by each image token's value-vector norm

Then: how much do the two head sets differ, and — the question that motivated
this — does the value-weighted ranking pick heads with a LARGER causal
ablation effect than the raw ranking? If yes, raw attention was mismeasuring
and value-weighting partly explains the attention-vs-causal dissociation. If
the causal effects come out the same, the dissociation is NOT a value-norm
artifact and the redundancy explanation (H3) survives.
"""
import sys
from pathlib import Path

REPO_ROOT = Path("/mnt/abka03/Projects/vis-head")
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from scipy import stats
from tqdm.auto import tqdm

from vis_head.common import DEFAULT_COMICS_ROOT, DEFAULT_MODEL_ID, DEFAULT_N_PANELS, DEFAULT_SEED, dump_json
from vis_head.data import build_strip, list_comic_dirs
from vis_head.gaze import (
    aggregate_region_attention, collect_last_query_attentions,
    collect_last_query_attentions_value_weighted, panel_query_prompt, rank_heads_by_score,
)
from vis_head.judge import bootstrap_ci, semantic_similarity
from vis_head.modeling import (
    decode_generated_text, find_image_token_range, load_model_and_processor,
    model_dims, prepare_inputs, run_generation,
)
from vis_head.regions import assign_panels_to_tokens, region_positions_from_ids
from vis_head.steering import group_heads_by_layer, make_static_attention_mask_hook, register_mask_hooks, remove_handles

MODEL_ID = DEFAULT_MODEL_ID
DEVICE = "cuda:0"
SEED = DEFAULT_SEED
N_PANELS = DEFAULT_N_PANELS
N_DISCOVERY = 100
N_CAUSAL_SAMPLES = 30
CAUSAL_HEAD_BUDGET = 15
CAUSAL_MAX_NEW_TOKENS = 40

model, processor = load_model_and_processor(model_id=MODEL_ID, device=DEVICE)
n_layers, n_heads, spatial_merge = model_dims(model)
print(f"{n_layers} layers x {n_heads} heads")

comic_dirs = list_comic_dirs(Path(DEFAULT_COMICS_ROOT), n_panels=N_PANELS)
discovery_dirs = comic_dirs[:N_DISCOVERY]

# ---------------------------- discovery, both metrics ----------------------------
raw_sum = np.zeros((n_layers, n_heads), dtype=np.float64)
vw_sum = np.zeros((n_layers, n_heads), dtype=np.float64)
valid = 0
for comic_dir in tqdm(discovery_dirs, desc="Discovery (raw + value-weighted)"):
    strip = build_strip(comic_dir, n_panels=N_PANELS)
    try:
        per_prompt_raw, per_prompt_vw = [], []
        region_ids = None
        for panel_index in range(1, N_PANELS + 1):
            prompt = panel_query_prompt(panel_index, n_panels=N_PANELS)
            inputs = prepare_inputs(processor, strip.strip, prompt, DEVICE)
            if region_ids is None:
                region_ids, _, _ = assign_panels_to_tokens(
                    image_grid_thw=inputs["image_grid_thw"], panel_widths=strip.panel_widths,
                    spatial_merge=spatial_merge)
            attn_raw = collect_last_query_attentions(model, inputs)
            attn_vw = collect_last_query_attentions_value_weighted(model, inputs, processor)
            per_prompt_raw.append(aggregate_region_attention(
                attn_at_query=attn_raw, inputs=inputs, processor=processor,
                region_ids=region_ids, n_regions=N_PANELS))
            per_prompt_vw.append(aggregate_region_attention(
                attn_at_query=attn_vw, inputs=inputs, processor=processor,
                region_ids=region_ids, n_regions=N_PANELS))
        # diagonal: attention on the queried panel
        for p in range(N_PANELS):
            raw_sum += per_prompt_raw[p][:, :, p]
            vw_sum += per_prompt_vw[p][:, :, p]
        valid += 1
    except Exception as exc:
        print(f"Skipping {strip.name}: {exc}")

raw_scores = (raw_sum / max(valid, 1) / N_PANELS).astype(np.float32)
vw_scores = (vw_sum / max(valid, 1) / N_PANELS).astype(np.float32)
print(f"\nvalid strips: {valid}/{len(discovery_dirs)}")
print(f"raw   mean={raw_scores.mean():.5f}  max={raw_scores.max():.5f}")
print(f"vw    mean={vw_scores.mean():.5f}  max={vw_scores.max():.5f}")

raw_ranked = rank_heads_by_score(raw_scores)
vw_ranked = rank_heads_by_score(vw_scores)

print("\n=== How different are the two head sets? ===")
for K in (10, 15, 50, 100):
    top_raw = set((r["layer"], r["head"]) for r in raw_ranked[:K])
    top_vw = set((r["layer"], r["head"]) for r in vw_ranked[:K])
    ov = len(top_raw & top_vw) / K
    print(f"  top-{K:<4d} overlap: {ov:.3f}  ({len(top_raw & top_vw)}/{K})   "
          f"raw-only={sorted(top_raw - top_vw)[:3]}...  vw-only={sorted(top_vw - top_raw)[:3]}...")
rho = stats.spearmanr(raw_scores.reshape(-1), vw_scores.reshape(-1))
print(f"  Spearman rho: {rho.correlation:.3f}  (p={rho.pvalue:.3e})")

print(f"\nTop-10 heads by each metric:")
print(f"  raw: {[(r['layer'], r['head']) for r in raw_ranked[:10]]}")
print(f"  vw : {[(r['layer'], r['head']) for r in vw_ranked[:10]]}")

# ---------------------------- causal check on each metric's top heads ----------------------------
print(f"\n=== Causal effect of each metric's top-{CAUSAL_HEAD_BUDGET} heads ===")


def causal_effects(heads_by_layer, label):
    rng = np.random.RandomState(SEED)
    results = []
    for comic_dir in tqdm(comic_dirs[:N_CAUSAL_SAMPLES], desc=f"Causal [{label}]", leave=False):
        strip = build_strip(comic_dir, n_panels=N_PANELS)
        target_panel = int(rng.randint(N_PANELS))
        prompt = panel_query_prompt(target_panel + 1, n_panels=N_PANELS)
        try:
            inputs = prepare_inputs(processor, strip.strip, prompt, DEVICE)
            img_start, img_end = find_image_token_range(inputs, processor)
            region_ids, _, _ = assign_panels_to_tokens(
                image_grid_thw=inputs["image_grid_thw"], panel_widths=strip.panel_widths,
                spatial_merge=spatial_merge)
            positions = region_positions_from_ids(img_start=img_start, region_ids=region_ids, n_regions=N_PANELS)
            target_positions = positions[target_panel]
            prompt_length = int(inputs["input_ids"].shape[1])

            baseline_sequences = run_generation(model=model, inputs=inputs, max_new_tokens=CAUSAL_MAX_NEW_TOKENS)
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
                ablated_sequences = run_generation(model=model, inputs=ablate_inputs, max_new_tokens=CAUSAL_MAX_NEW_TOKENS)
            finally:
                remove_handles(handles)
            ablated_text = decode_generated_text(processor, ablated_sequences, prompt_length)
            similarity = semantic_similarity(ablated_text, baseline_text, device="cpu")
            results.append({"changed": similarity < 0.85, "effect": 1.0 - similarity})
        except Exception as exc:
            print(f"Skipping {strip.name}: {exc}")
    return results


raw_heads = group_heads_by_layer([(r["layer"], r["head"]) for r in raw_ranked[:CAUSAL_HEAD_BUDGET]])
vw_heads = group_heads_by_layer([(r["layer"], r["head"]) for r in vw_ranked[:CAUSAL_HEAD_BUDGET]])
res_raw = causal_effects(raw_heads, "raw")
res_vw = causal_effects(vw_heads, "vw")

print(f"\n{'metric':>8s}  {'n':>4s}  {'change rate':>12s}  {'95% CI':>16s}  {'mean effect':>12s}")
for label, res in (("raw", res_raw), ("vw", res_vw)):
    ci = bootstrap_ci([r["changed"] for r in res])
    print(f"{label:>8s}  {ci['n']:4d}  {ci['accuracy']:12.3f}  [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]  "
          f"{np.mean([r['effect'] for r in res]):12.3f}")

n = min(len(res_raw), len(res_vw))
w = stats.wilcoxon([r["effect"] for r in res_raw][:n], [r["effect"] for r in res_vw][:n])
print(f"Paired Wilcoxon (raw vs vw causal effect): statistic={w.statistic:.1f}  p={w.pvalue:.3e}")

dump_json(REPO_ROOT / "logs" / "value_weighted_compare" / "summary.json", {
    "raw_mean": float(raw_scores.mean()), "vw_mean": float(vw_scores.mean()),
    "spearman_rho": float(rho.correlation),
    "causal_raw": float(np.mean([r["effect"] for r in res_raw])),
    "causal_vw": float(np.mean([r["effect"] for r in res_vw])),
})
np.save(REPO_ROOT / "logs" / "value_weighted_compare" / "raw_scores.npy", raw_scores)
np.save(REPO_ROOT / "logs" / "value_weighted_compare" / "vw_scores.npy", vw_scores)
print("\nDONE")
