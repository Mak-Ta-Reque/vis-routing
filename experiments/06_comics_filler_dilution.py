"""Same filler-dilution test as the ImageNet-grid one, applied to the
original comics panel-pointing prompt:

  original - "Look carefully at this six-panel comic strip. What is
              happening in the 3rd panel from the left? Answer briefly."
  padded   - the same, wrapped in extra unnecessary sentences before and
              after (irrelevant content, doesn't change the question).

Checks both vir score (attention identity) and causal-ablation effect
(does padding weaken how much the answer depends on the top comics vir
heads, the way it did for the ImageNet-grid "Look at the picture." filler?).
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
from vis_head.vir import aggregate_region_attention, collect_last_query_attentions, ordinal, rank_heads_by_score
from vis_head.judge import bootstrap_ci, semantic_similarity
from vis_head.modeling import decode_generated_text, find_image_token_range, load_model_and_processor, model_dims, prepare_inputs, run_generation
from vis_head.regions import assign_panels_to_tokens, region_positions_from_ids
from vis_head.steering import group_heads_by_layer, make_static_attention_mask_hook, register_mask_hooks, remove_handles

MODEL_ID = DEFAULT_MODEL_ID
DEVICE = "cuda:0"
SEED = DEFAULT_SEED
N_PANELS = DEFAULT_N_PANELS
N_SAMPLES = 30
N_CAUSAL_HEADS = 15
CAUSAL_MAX_NEW_TOKENS = 40

_NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}


def original_prompt(panel_index: int, n_panels: int) -> str:
    count = _NUMBER_WORDS.get(n_panels, str(n_panels))
    return (
        f"Look carefully at this {count}-panel comic strip. "
        f"What is happening in the {ordinal(panel_index)} panel from the left? "
        "Answer briefly."
    )


def padded_prompt(panel_index: int, n_panels: int) -> str:
    core = original_prompt(panel_index, n_panels)
    preamble = "This is a wonderful comic strip that many readers around the world enjoy in their free time. "
    postscript = " Please take your time and think carefully before answering, and remember that comics can be a great way to relax."
    return preamble + core + postscript


model, processor = load_model_and_processor(model_id=MODEL_ID, device=DEVICE)
n_layers, n_heads, spatial_merge = model_dims(model)
print(f"{n_layers} layers x {n_heads} heads")

comic_dirs = list_comic_dirs(Path(DEFAULT_COMICS_ROOT), n_panels=N_PANELS)[:N_SAMPLES]

print("\nExample prompts (strip 0, panel 3):")
print(f"  [original] {original_prompt(3, N_PANELS)!r}")
print(f"  [padded  ] {padded_prompt(3, N_PANELS)!r}")

prompt_fns = {"original": original_prompt, "padded": padded_prompt}


# ---------------------------- Part 1: vir score ----------------------------
def discover_for_prompt_fn(prompt_fn, label: str) -> np.ndarray:
    rng = np.random.RandomState(SEED)
    raw_sum = np.zeros((n_layers, n_heads), dtype=np.float64)
    valid = 0
    for comic_dir in tqdm(comic_dirs, desc=f"[{label}]", leave=False):
        strip = build_strip(comic_dir, n_panels=N_PANELS)
        target_panel = int(rng.randint(N_PANELS))
        prompt = prompt_fn(target_panel + 1, N_PANELS)
        try:
            inputs = prepare_inputs(processor, strip.strip, prompt, DEVICE)
            region_ids, _, _ = assign_panels_to_tokens(
                image_grid_thw=inputs["image_grid_thw"], panel_widths=strip.panel_widths, spatial_merge=spatial_merge)
            attn_at_query = collect_last_query_attentions(model, inputs)
            region_attention = aggregate_region_attention(
                attn_at_query=attn_at_query, inputs=inputs, processor=processor, region_ids=region_ids, n_regions=N_PANELS)
            raw_sum += region_attention[:, :, target_panel]
            valid += 1
        except Exception as exc:
            print(f"Skipping {strip.name}: {exc}")
    print(f"  [{label:>9s}] valid={valid}/{N_SAMPLES}  mean score={raw_sum.mean() / max(valid, 1):.5f}")
    return (raw_sum / max(valid, 1)).astype(np.float32)


scores = {name: discover_for_prompt_fn(fn, name) for name, fn in prompt_fns.items()}
ranked_o, ranked_p = rank_heads_by_score(scores["original"]), rank_heads_by_score(scores["padded"])
print("\n=== Vir-score comparison ===")
for K in (10, 50, 100):
    top_o = set((r["layer"], r["head"]) for r in ranked_o[:K])
    top_p = set((r["layer"], r["head"]) for r in ranked_p[:K])
    print(f"  top-{K:<4d} overlap: {len(top_o & top_p) / K:.3f}")
rho = stats.spearmanr(scores["original"].reshape(-1), scores["padded"].reshape(-1))
print(f"  Spearman rho: {rho.correlation:.3f}  (p={rho.pvalue:.3e})")
print(f"  Mean score: original={scores['original'].mean():.5f}  padded={scores['padded'].mean():.5f}")


# ---------------------------- Part 2: causal effect ----------------------------
print("\n=== Causal-effect check (ablate top comics vis heads under each phrasing) ===")
comics_ranking = json.loads((REPO_ROOT / "logs" / "vis_head_discovery_compare_datasets_comics" / "vis_head_ranking.json").read_text())
top_heads = [(r["layer"], r["head"]) for r in comics_ranking[:N_CAUSAL_HEADS]]
heads_by_layer = group_heads_by_layer(top_heads)
print(f"Ablating {N_CAUSAL_HEADS} pre-identified comics vis heads across {len(heads_by_layer)} layers.")

causal_results = {name: [] for name in prompt_fns}
rng = np.random.RandomState(SEED)
for comic_dir in tqdm(comic_dirs, desc="Causal ablation"):
    strip = build_strip(comic_dir, n_panels=N_PANELS)
    target_panel = int(rng.randint(N_PANELS))
    for name, fn in prompt_fns.items():
        prompt = fn(target_panel + 1, N_PANELS)
        try:
            inputs = prepare_inputs(processor, strip.strip, prompt, DEVICE)
            img_start, img_end = find_image_token_range(inputs, processor)
            region_ids, _, _ = assign_panels_to_tokens(
                image_grid_thw=inputs["image_grid_thw"], panel_widths=strip.panel_widths, spatial_merge=spatial_merge)
            positions = region_positions_from_ids(img_start=img_start, region_ids=region_ids, n_regions=N_PANELS)
            target_positions = positions[target_panel]
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
            ablate_inputs = prepare_inputs(processor, strip.strip, prompt, DEVICE)
            handles = register_mask_hooks(model, hook_by_layer)
            try:
                ablated_sequences = run_generation(model=model, inputs=ablate_inputs, max_new_tokens=CAUSAL_MAX_NEW_TOKENS)
            finally:
                remove_handles(handles)
            ablated_text = decode_generated_text(processor, ablated_sequences, prompt_length)

            similarity = semantic_similarity(ablated_text, baseline_text, device="cpu")
            causal_results[name].append({"changed": similarity < 0.85, "effect": 1.0 - similarity})
        except Exception as exc:
            print(f"Skipping {strip.name} [{name}]: {exc}")

print(f"\n{'phrasing':>9s}  {'n':>4s}  {'change rate':>12s}  {'95% CI':>16s}  {'mean effect':>12s}")
for name in prompt_fns:
    res = causal_results[name]
    ci = bootstrap_ci([r["changed"] for r in res])
    mean_effect = np.mean([r["effect"] for r in res])
    print(f"{name:>9s}  {ci['n']:4d}  {ci['accuracy']:12.3f}  [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]  {mean_effect:12.3f}")

n = min(len(causal_results["original"]), len(causal_results["padded"]))
ea = [r["effect"] for r in causal_results["original"][:n]]
eb = [r["effect"] for r in causal_results["padded"][:n]]
w = stats.wilcoxon(ea, eb)
print(f"Paired Wilcoxon (original vs padded causal effect): statistic={w.statistic:.1f}  p={w.pvalue:.3e}")

dump_json(REPO_ROOT / "logs" / "comics_filler_check" / "summary.json", {
    "vis_head_score_means": {k: float(v.mean()) for k, v in scores.items()},
    "causal_mean_effect": {k: float(np.mean([r["effect"] for r in v])) for k, v in causal_results.items()},
})
print("\nDONE")
