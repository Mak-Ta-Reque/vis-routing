# Vis-Head Investigation: Experiment Index

A reference index for every experiment run in this investigation, in the order they
were run, each with its motivation, key finding, and the exact file that reproduces
it. This is a catalog, not a script — it does **not** execute anything itself. Run the
referenced notebook/script directly to reproduce a given experiment.

**Layout:**
- `*.ipynb` in the repo root — the durable, polished notebooks (dataset builders,
  main comparisons, the final MCQ causal-intervention notebook).
- `experiments/NN_*.py` — standalone scripts for the smaller, targeted follow-up
  probes (confound checks, robustness tests). These are plain scripts, not notebooks,
  meant to be run with `python experiments/NN_name.py` from the repo root; each prints
  its results and saves numeric summaries under `logs/`.
- `vis_head/*.py` — the reusable library code all of the above imports.

**Prerequisites common to everything below:** a GPU, the repo's conda env
(`virheads`), and — for anything touching COCO or ImageNet — the raw data at the
paths configured in `vis_head/coco.py` / `vis_head/imagenet_grid.py`
(`VIR_COCO_ROOT`, `VIR_IMAGENET_ROOT` env vars if you need to override them).

## 1. Dataset builders

| # | What | Key output | File |
|---|---|---|---|
| 1.1 | Build the COCO Vis-Head dataset — one sample per unambiguous COCO object, image + `"Find the {category}."`, ground truth = segmentation mapped onto the VLM's visual-patch grid | `data/coco_vis_head/{metadata,ground_truth}.jsonl` | `build_coco_vis_head_dataset.ipynb` |
| 1.2 | ImageNet-grid module — tiles distinct-category ImageNet images into an `rows x cols` grid, one object per cell, re-randomized every sample; verb/bare/ordinal/positional prompt templates | — (library, used by 1.3 and everything downstream) | `vis_head/imagenet_grid.py` |
| 1.3 | ImageNet-grid dataset walkthrough + visualization | — | `imagenet_grid_vis_heads.ipynb` (Parts 1-2) |

`06_build_coco_vis_head_dataset.py` is the CLI/script version of 1.1, for regenerating
the dataset outside a notebook.

## 2. Cross-dataset Vis-Head comparison (comics vs. COCO vs. ImageNet-grid)

| # | What | Key finding | File |
|---|---|---|---|
| 2.1 | Raw + area-normalized visual-routing score, comics vs. COCO | Raw score favors comics (5x), but region size differs 1.3-4x between datasets; normalizing narrows but doesn't eliminate the gap | `compare_comics_vs_coco_vis_heads.ipynb` Parts 1-3 |
| 2.2 | Causal-effect sweep (ablation), comics vs. COCO, semantic-similarity effect size | COCO shows a **larger** causal effect than comics despite comics' higher raw attention score — first sign attention concentration != causal importance | `compare_comics_vs_coco_vis_heads.ipynb` Part 4 |
| 2.3 | Three-way discovery: comics vs. COCO vs. ImageNet-grid (all cached to `logs/vis_head_discovery_compare_datasets_{comics,coco,imagenet}/`) | COCO and ImageNet-grid agree with each other far more (67% top-100 overlap) than either agrees with comics (~40-50%) — named-object tasks cluster together, separate from comics' positional task | `imagenet_grid_vis_heads.ipynb` Parts 2-3 |
| 2.4 | Three-way causal-effect sweep + object-generalization (split-half) + verb-generalization | Object identity: near-irrelevant (96% overlap, disjoint object sets). Verb phrasing: near-irrelevant (80% overlap). Reference **type** (named-object vs. positional): only ~45-60% overlap — the real dividing line | `imagenet_grid_vis_heads.ipynb` Parts 4-6 |
| 2.5 | 400-sample discovery with a comics-style verbose ("Look carefully...") prompt on COCO and ImageNet, vs. cached short-prompt rankings | Head set shifts more than any wording-only change tested; ImageNet's raw score nearly catches up to comics' (0.047 vs 0.048), COCO's stays well behind | `experiments/11_long_prompt_discovery_qwen3vl.py` |

## 3. Cross-model comparison (base vs. instruct vs. agentic)

| # | What | Key finding | File |
|---|---|---|---|
| 3.1 | Raw visual-routing score, base vs. instruct vs. agentic (`ByteDance-Seed/UI-TARS-7B-DPO`, a GUI-agent fine-tune of Qwen2-VL-7B) | Instruct and agentic score far higher than base; instruct and agentic are nearly identical to each other (93% top-100 overlap) despite opposite training objectives | `compare_base_vs_instruct_vis_heads.ipynb` Parts 1-2 |
| 3.2 | Confound check: total image-attention (engagement) vs. concentration (accuracy given engagement) | Most of instruct's raw-score gain is general engagement (3.18x), not targeting; concentration gap is real but modest (1.24x mean), sharper in top heads (~2x) | `compare_base_vs_instruct_vis_heads.ipynb` (concentration section) |
| 3.3 | Causal-effect sweep, base vs. instruct vs. agentic, head-budget sweep [50,15,5] | Base's outputs depend **more** on the same heads than instruct/agentic's do, despite base's blunter attention — the reverse of what raw scores predict | `compare_base_vs_instruct_vis_heads.ipynb` (causal-effect section) |
| 3.4 | Random-head ablation control — is base's causal advantage genuinely Vis-Head-specific, or just general fragility? | Base is much more fragile to ANY head ablation (3-10x higher random-head effect); once you subtract that baseline, base's *Vis-Head-specific* excess is comparable to or smaller than instruct/agentic's — the original "base is more causal" claim was largely a fragility artifact | `experiments/01_random_head_ablation_control.py` |

## 4. What makes the discovery *metric* itself good or bad

| # | What | Key finding | File |
|---|---|---|---|
| 4.1 | `word_jaccard_similarity` -> `semantic_similarity` (NLI-based, DeBERTa-xlarge-MNLI) causal-effect metric | Lexical-overlap effect size underestimated true effect; switching to entailment-based similarity made the base > instruct/agentic causal finding *more* significant, not less | `vis_head/judge.py` (`semantic_similarity`); re-run via `compare_base_vs_instruct_vis_heads.ipynb` / `compare_comics_vs_coco_vis_heads.ipynb` |
| 4.2 | Value-weighted attention (`attn x \|\|value\|\|`, Kobayashi et al. 2020) vs. raw attention, head-to-head causal test | Value-weighting relocates "top heads" to late layers (value norm grows with depth, not function) and picks heads with **3.6x smaller** causal effect than raw attention — raw attention wins decisively | `experiments/08_value_weighted_vs_raw_attention.py` |
| 4.3 | Gradient-based attribution patching attempt | **Infeasible as implemented** — full-graph backward through the 8B VLM's vision tower + 36 LM layers OOM'd on a 24GB GPU even with gradient checkpointing; would need vision-tower/LM forward-pass splitting to fix properly. Not resolved; abandoned in favor of 4.4 | `vis_head/vir.py::collect_attribution_scores` (present but never successfully run end-to-end) |
| 4.4 | Single-head causal **sufficiency** (boost one head alone toward panel A vs. B, no location text) vs. raw attention, cross-validated against group ablation-**necessity** | Sufficiency-selected heads underperform raw-attention heads on the necessity test (3.6x smaller effect) — sufficiency and necessity are different, only partially correlated causal properties (H6); raw attention (necessity-aligned) was the right choice for this project's "what does the model naturally rely on" question all along | `experiments/09_single_head_causal_sufficiency.py` |

## 5. Prompt-robustness and confound-isolation probes (mostly on ImageNet-grid)

| # | What | Key finding | File |
|---|---|---|---|
| 5.1 | Verb phrasing (find/locate/where_is/point_to/identify) + bare object name | All cluster tightly (0.70-0.92 overlap) — wording doesn't matter within the named-object reference type | `imagenet_grid_vis_heads.ipynb` Part 6 |
| 5.2 | Word-order shuffle ("Find the chickadee." -> "chickadee the Find.") | 0.88 overlap, rho=0.99 with the original — syntax doesn't matter either | `experiments/10_word_order_shuffle.py` |
| 5.3 | Filler/irrelevant text injection ("Look at the picture." / "Look carefully at this picture.") | Doesn't change *which* heads engage (0.84 overlap) but *does* significantly weaken their **causal** effect (~26-29% drop, p<0.02) on ImageNet — attention identity and causal reliance dissociate again, now as a property of the prompt itself | `experiments/07_imagenet_filler_dilution.py` |
| 5.4 | Same filler test on the **comics** prompt (comics already has its own verbose framing) | No significant causal-effect drop from added filler — the ImageNet effect doesn't replicate on comics, a genuine dataset/task-type-dependent boundary, not a universal filler penalty | `experiments/06_comics_filler_dilution.py` |
| 5.5 | Object identity swap: always "golden retriever" vs. always "persian cat," same setup otherwise | 91-92% top-k overlap, rho=0.992, nearly identical mean scores — object identity essentially irrelevant, confirmed directly (not just inferred from a random split) | `experiments/03_object_identity_swap.py` |
| 5.6 | Mismatched-object probe: ask "Find the cat." on a grid built to guarantee no cat is present | The heads that engage on the impossible query still resemble the *matched*-object heads (0.86 overlap) more than the *positional* heads (0.66-0.70) — object-reference heads fire on *attempting* semantic grounding, not on its success | `experiments/04_mismatched_object_probe.py` |
| 5.7 | Digit-token confound control: bare position with NO digits ("top-left") vs. object query WITH an irrelevant digit ("...in image 1.") | Every prediction of "it's about reference type" confirmed; every prediction of "it's about digit tokens" contradicted — the named-object/positional split is genuinely about reference type, not a tokenizer quirk | `experiments/05_digit_confound_control.py` |

## 6. Does comics' steering advantage survive controlling for reference type?

| # | What | Key finding | File |
|---|---|---|---|
| 6.1 | Open-ended-caption steering demo: discover heads on COCO with Qwen2-VL-7B-Instruct, steer 10 multi-object COCO images toward the *other* (non-obvious) object's box, judge against ground truth | 5/10 — but inspection revealed several "matches" were cases the baseline already got right (dominant-object bbox), and one clean failure showed attention moved correctly (per heatmap) but content generation still failed for a tiny/ambiguous region | `qwen2vl_coco_steer_demo.ipynb` |
| 6.2 | Same 10 images, steered instead with the **comics**-discovered ranking (Qwen2-VL-7B-Instruct) | 8/10 — comics heads noticeably outperformed the COCO-discovered ones, on COCO's own task | `experiments/14_steer_comics_heads_on_coco.py` |
| 6.3 | Same 10 images, COCO/ImageNet heads discovered with a comics-style **verbose** prompt this time (controls for prompt style, not just wording) | Still 6/10 and 6/10 vs. comics' 8/10 — comics wins even with prompt style matched | `experiments/12_steer_coco_longprompt_heads.py`, `experiments/13_steer_imagenet_longprompt_heads.py` |
| 6.4 | Scaled up to **500** multi-object COCO images, same 3 head sets, bootstrap CIs + paired Wilcoxon | comics 74.6%, ImageNet-lp 74.0%, COCO-lp 69.8%. comics beats COCO (p=8.6e-07) and ImageNet (p=4.1e-04) significantly — but the comics-vs-ImageNet gap is much smaller than the n=10 result suggested (small-sample noise) | `experiments/15_steer_3way_500images.py` |
| 6.5 | **The decisive control**: ImageNet-grid discovered with comics' own *ordinal* (positional, no object name) prompt structure, at 2x2 and 4x4, same 500 images | Match rate identical to 3 decimals (comics 0.746 = ordinal_2x2 0.746); comics vs. ordinal_2x2 p=0.459 (**not significant**) — comics' apparent advantage was entirely the reference-type confound, not domain or discovery quality. ordinal_4x4 is slightly worse than ordinal_2x2 (p=0.013) — grid density, not comics-specific magic | `vis_head/imagenet_grid.py::comics_style_ordinal_prompt`; `experiments/16_steer_ordinal_matched_2x2_4x4.py` |

## 7. Controlled causal intervention: forced-choice pointing game (final, most rigorous)

| # | What | Key finding | File |
|---|---|---|---|
| 7.1 | MCQ pointing-game design: grid + neutral (location-free) prompt + 4 options drawn from objects actually in the grid; baseline vs. attention-steered accuracy, across 2x2/3x3/4x4 | Steering roughly **doubles** accuracy over chance-level baseline at every grid size (2x2: 29%->57%, McNemar p=1.8e-06); accuracy degrades monotonically with grid density (57%->50%->47%), replicating the 2x2-vs-4x4 finding from section 6.5 in a completely independent, ground-truth-verifiable paradigm | `mcq_causal_intervention.ipynb` Parts 1-4 |
| 7.2 | Added a second, **language-only** baseline: explicitly tell the model the target's row/column in words, no attention intervention | Location-cue baseline (39-42%) beats no-cue (26-29%) but still loses to attention-steering (47-57%) at every grid size, significantly at 2x2 (p=0.025) — directly manipulating attention is a stronger causal lever than instructing the model in language, even when the instruction is unambiguous | `mcq_causal_intervention.ipynb` Part 3 (3-condition version) |
| 7.3 | Pointing-game qualitative demo: search for a sample where the location-cue baseline fails but steering succeeds, visualize both full prompts + the attention heatmap | Found and reproduced deterministically: "row 2, column 1, lab coat" — location-cue answers "otter" (wrong), steering answers "lab coat" (correct), heatmap shows a precise, correctly-localized hotspot | `mcq_causal_intervention.ipynb` (pointing-game demo cell); `experiments/17_pointing_game_demo_reproduce.py` (standalone reproduction, cheaper than a full notebook re-run) |

**Theoretical framing** (motivation from the Pointing Game protocol, formal ATE/
McNemar formulation) is written up in `mcq_causal_intervention.ipynb`'s Part 0/Part 1
markdown cells.

## Quick reference: where things are cached

- `logs/vis_head_discovery_qwen2vl_{base,instruct,agentic}/` — comics-discovered rankings per checkpoint (section 3).
- `logs/vis_head_discovery_compare_datasets_{comics,coco,imagenet}/` — short-prompt 3-way rankings (section 2.3), each with raw + normalized `vis_head_scores.npy` and `vis_head_ranking.json`.
- `logs/vis_head_discovery_compare_datasets_{coco,imagenet}_longprompt/` — 400-sample verbose-prompt rankings, Qwen3-VL-8B-Instruct (section 2.5).
- `logs/vis_head_discovery_qwen2vl_instruct_{coco,imagenet}_longprompt/` — same, but Qwen2-VL-7B-Instruct, used for the steering comparisons in section 6.
- `logs/vis_head_discovery_qwen2vl_instruct_imagenet_ordinal_{2x2,4x4}/` — the reference-type-controlled rankings (section 6.5).
- `logs/steer_demo_*/`, `logs/mcq_causal_intervention/`, `logs/*_check/` — per-experiment JSON summaries for everything in sections 4-7.
- `data/coco_vis_head/` — the built COCO Vis-Head dataset (section 1.1).

**Model checkpoints used throughout:** `Qwen/Qwen3-VL-8B-Instruct` (default, `vis_head/common.py::DEFAULT_MODEL_ID`) for the main comics/COCO/ImageNet-grid comparisons (sections 2, 3, 5); `Qwen/Qwen2-VL-7B-Instruct` for everything steering-related in sections 6-7 (needed base + instruct + agentic variants to exist for the same architecture); `Qwen/Qwen2-VL-7B` (base) and `ByteDance-Seed/UI-TARS-7B-DPO` (agentic) additionally in section 3.
