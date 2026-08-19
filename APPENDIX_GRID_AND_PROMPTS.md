  # Appendix: ImageNet-Grid Dataset Construction and Prompt Strategies

  This appendix documents (1) how the ImageNet-grid dataset used for Vis-Head
  discovery and causal intervention is constructed, (2) the prompt strategies used
  for Vis-Head *routing/discovery* (attention-score measurement), and (3) the three
  prompt/intervention conditions used for *causal* evaluation — no-cue baseline,
  language location-cue, and attention steering — and how they are compared.

  Implementation: `vis_head/imagenet_grid.py` (grid construction, prompt templates),
  `vis_head/regions.py` (token-to-cell assignment), `vis_head/steering.py`
  (attention-hook intervention), used throughout `mcq_causal_intervention.ipynb`,
  `prompt_phrasing_vis_head_vs_causal.ipynb`, `top2_phrasings_across_model_stages.ipynb`,
  `head_count_sweep_qwen2vl.ipynb`, and `linear_probe_causal_intervention.ipynb`.

  ---

  ## 1. Grid dataset construction

  ### 1.1 Motivation

  Two earlier discovery datasets in this project — hand-drawn comic strips and COCO
  photographs — each have a confound that makes causal attribution ambiguous:

  - **Comics**: panel content is fixed per strip. The same four panels are reused
    across every sample that strip appears in, so "does the model attend to panel 3"
    is entangled with "what happens to be drawn in panel 3."
  - **COCO**: real photographs have occlusion, clutter, and ambiguous object
    boundaries — a bounding box for "person" may overlap a bounding box for
    "bicycle," so attention landing in that region does not cleanly separate which
    object is being attended to.

  The ImageNet-grid dataset removes both confounds. Each sample tiles `rows x cols`
  **single-object** ImageNet validation images into one grid, with:

  - **No occlusion, no clutter** — each cell contains exactly one unambiguous object,
    center-cropped to a square and resized, on a plain white background between
    cells.
  - **Independent per-sample randomization** — which object lands in which cell is
    redrawn every sample, decoupling "attention to a region" from "identity of the
    object that happens to be there." This is what makes the dataset suitable for
    testing object-generality (holding the prompt fixed, varying the object) and
    prompt-generality (holding the object fixed, varying the instruction wording)
    independently.

  ### 1.2 Construction procedure

  For a `rows x cols` grid (default `2x2`) with cell size `256px` and a `6px` gap
  between cells (`vis_head/imagenet_grid.py::sample_grid`):

  1. **Sample `rows*cols` distinct ImageNet classes** without replacement from the
    validation split's class directories (`rng.choice(len(class_dirs), size=n_cells,
    replace=False)`). Distinctness is enforced at the class-directory (WordNet
    synset ID) level, guaranteeing no two cells in a grid share a class — this is
    what keeps a discovery prompt like `"Find the {name}."` and an MCQ option list
    unambiguous (no risk of two cells matching the same option, or a distractor
    option accidentally equaling the target).
  2. **Sample one image per chosen class** uniformly at random from that class's
    available images.
  3. **Center-crop each image to a square** (crop to `min(width, height)`, avoiding
    any resize-induced aspect-ratio distortion of the object), then resize to the
    target cell size.
  4. **Tile cells row-major** into one grid image on a white canvas, with a fixed
    pixel gap between cells. Each cell's pixel bounding box
    `(x0, y0, x1, y1)` is recorded (`ImageNetGrid.cell_bboxes`), row-major, enabling
    both visualization (`draw_cell_labels`) and pixel-coordinate location prompts.
  5. The resulting `ImageNetGrid` dataclass carries the assembled grid image, each
    cell's source image, WordNet ID, human-readable class name, and pixel bounding
    box — everything needed to build both discovery prompts (name the target
    object) and MCQ prompts (list candidate names, know which cell/bbox is ground
    truth) without re-deriving anything from the image.

  ### 1.3 Token-to-cell mapping

  Vision-token sequences from Qwen-VL-family models are mapped back onto the `rows x
  cols` grid geometry via `vis_head/regions.py::assign_grid_cells_to_tokens`, using
  the model's own `image_grid_thw` (patch grid shape) and `spatial_merge_size` to
  convert each cell's pixel bounding box into a range of token positions. This
  mapping is what makes it possible to (a) aggregate attention scores per grid cell
  during discovery, and (b) target the correct token positions during attention
  steering. For architectures without Qwen-VL's dynamic `image_grid_thw` (e.g.
  LLaVA's fixed 24x24 patch grid), a synthetic `image_grid_thw`-equivalent tensor is
  constructed so the same mapping function is reused unmodified.

  ---

  ## 2. Prompt strategies for Vis-Head discovery/routing

  Discovery measures, for a given prompt and target cell, how much attention mass
  each `(layer, head)` places on the target cell's tokens at the final query
  position, averaged over many samples (`vis_head/gaze.py::aggregate_region_attention`,
  `collect_last_query_attentions`). The **discovery prompt directly names the
  target object** — there is no MCQ, no letter to parse, no generation-format
  consideration. Its only job is to give the model a reason to route attention to a
  specific region.

  ### 2.1 Verb/instruction-phrasing sweep (`PROMPT_TEMPLATES`)

  Six phrasings of the same underlying reference, varying only the instruction verb,
  to test whether discovered heads are tied to specific wording or generalize across
  verbs:

  | tag | template |
  |---|---|
  | `find` | `"Find the {name}."` |
  | `locate` | `"Locate the {name}."` |
  | `where_is` | `"Where is the {name}?"` |
  | `point_to` | `"Point to the {name}."` |
  | `identify` | `"Identify the {name}."` |
  | `bare_name` | `"{name}"` (no instruction/verb at all — the referring expression alone) |

  `find` matches the phrasing used for the COCO gaze dataset
  (`vis_head/coco.py`), for cross-dataset comparability. `bare_name` isolates
  whether a task framing is even necessary or whether the reference alone is
  sufficient to recruit the same heads.

  A broader 10-phrasing sweep (`prompt_phrasing_vis_head_vs_causal.ipynb`) extends
  this with `verbose_framing` (comic-strip-style elaborate framing), `classify`,
  `look_carefully`, and `shuffled`/`bare` word-order controls — used to test
  robustness of the causal effect (not just the discovery score) to surface-level
  linguistic variation.

  ### 2.2 Position-only phrasings (no object name)

  Used to test whether Vis-Heads are driven by the *type of reference* (semantic
  object-name vs. spatial/positional) rather than being tied to particular words:

  - `ordinal_prompt(cell_index, n_cells)` — `"What is in cell {i} of {n}, counting
    left to right, top to bottom?"`
  - `comics_style_ordinal_prompt(cell_index, n_cells)` — a word-for-word mirror of
    the comics discovery prompt's framing/closing ("Look carefully at this
    {count}-cell picture grid. What is happening in the {ordinal} cell... Answer
    briefly."), holding verbosity and structure fixed while only the reference type
    varies.
  - `bare_row_col_prompt(cell_index, cols)` — bare `(row, col)` coordinate, e.g.
    `"(1, 1)"` — no instruction, no digits beyond the coordinate itself.
  - `bare_cell_prompt(cell_index)` — bare cell number, e.g. `"4"`.
  - `position_words_prompt(cell_index, rows, cols)` — word-based spatial reference
    with **no digit tokens at all**, e.g. `"top-left"` for a 2x2 grid's first cell
    (defined up to 3x3, using top/middle/bottom x left/center/right). This variant
    specifically controls for whether an observed positional/semantic head split is
    actually about digit-token presence rather than about spatial-vs-semantic
    reference type.

  ---

  ## 3. Three prompt/intervention conditions for causal evaluation

  Causal evaluation asks a different question from discovery: given a 4-way MCQ
  over grid-cell contents, does an intervention change whether the model answers
  correctly? Every causal-intervention notebook in this project (`mcq_causal_intervention.ipynb`,
  `prompt_phrasing_vis_head_vs_causal.ipynb`, `top2_phrasings_across_model_stages.ipynb`,
  `head_count_sweep_qwen2vl.ipynb`, `linear_probe_causal_intervention.ipynb`) uses the
  same three conditions, built from one shared `build_mcq_sample`:

  ```python
  correct_letter = ...              # tracked after shuffling 4 options (1 correct + 3 distractors)
  prompt          = f"Which of the following is shown in this image? {option_lines}. Answer with only the letter."
  location_prompt = mcq_prompt(target_cell + 1, rows, cols, options, letters)   # word-position phrasing
  ```

  ### 3.1 No-cue baseline

  **Prompt**: `sample["prompt"]` — the plain MCQ question above, with **no
  information about where in the image to look**. Run with no attention
  intervention (no hooks registered).

  This measures the model's default competence: given four candidate labels and an
  image containing all four objects in unknown positions, can it identify which one
  the question intends? Since the prompt is genuinely ambiguous about which cell is
  being asked about, this baseline is *expected* to be low — it is the reference
  point both other conditions are compared against, not a strong solution in its own
  right.

  ### 3.2 Language location-cue

  **Prompt**: `sample["location_prompt"]`, built by `vis_head/imagenet_grid.py::mcq_prompt`:

  ```
  "Which of the following is shown in the {word_position} part of this image? {option_lines}. Answer with only the letter."
  ```

  e.g. `"Which of the following is shown in the top-left part of this image? A) cat
  B) dog C) fish D) bird. Answer with only the letter."` Run with **no attention
  intervention** — this is a pure-language condition. It measures whether the model
  can resolve the location purely from an explicit verbal instruction, using its own
  unmodified attention.

  `mcq_prompt` names the location in words (`position_words_prompt`, e.g.
  "top-left") for grids up to 3x3, falling back to an explicit `(row, column)`
  coordinate for larger grids (word-position phrasing is undefined beyond
  top/middle/bottom x left/center/right). This specific phrasing was chosen after
  `mcq_prompt_format_search_v2_located.ipynb` empirically compared word-position,
  row/col-coordinate, and pixel-bounding-box phrasing across four model checkpoints
  (LLaVA-1.5, LLaVA-1.6, Gemma-3-4B-it, Gemma-3-4B-pt) and found word-position
  phrasing scored highest on average (mean accuracy 0.531 vs. 0.24–0.29 for the
  coordinate/bbox alternatives) — natural-language spatial words transfer far better
  across model families than raw coordinates, consistent with these words appearing
  far more often in each model's pretraining distribution than synthetic coordinate
  strings.

  *(An earlier version of this document's prompt design mistakenly used the
  location-cue phrasing as the default for the no-cue baseline and steered
  conditions as well, contaminating the "no cue" baseline with an implicit location
  hint. This has been corrected: `mcq_prompt` is used exclusively for the
  location-cue condition; the no-cue baseline and steered condition share the
  plain, location-free prompt.)*

  ### 3.3 Attention steering

  **Prompt**: `sample["prompt"]` — **identical text to the no-cue baseline**. Run
  **with** attention-hook intervention (`vis_head/steering.py`):
  `intervention_positions` computes which token positions belong to the target cell
  vs. the other cells; `make_static_attention_mask_hook` builds a pre-softmax
  additive bias (`boost_suppress` mode: saturating +∞/−∞ bias in bf16) that forces
  the top-`K` Vis-Heads (by discovery-time attention-score ranking) to attend to the
  target cell's tokens and away from the others; `register_mask_hooks` attaches
  these as forward pre-hooks on each targeted layer's `self_attn` module for the
  duration of generation.

  Because the prompt text is identical to the no-cue baseline, this is a clean
  paired comparison isolating the effect of the attention-hook intervention alone —
  **no textual information about location is present in either condition**; only the
  attention pattern differs.

  ### 3.4 Comparing the three conditions

  The central causal-effect quantities computed from these three conditions
  (`mcnemar_p` in the relevant notebooks) are:

  - **`ATE(steered, baseline) = P(correct | steered) − P(correct | baseline)`** —
    the primary causal claim: does redirecting attention (with no textual hint)
    improve accuracy over doing nothing? Tested via McNemar's paired test on the
    discordant baseline/steered outcome pairs.
  - **`ATE(location, baseline) = P(correct | location-cue) − P(correct | baseline)`**
    — does simply telling the model in words help over doing nothing?
  - **`steered − location`** — the key comparison for interpreting *how* the model
    solves the task: is attention manipulation a stronger or weaker localization
    signal than an explicit verbal instruction? Empirically (Qwen2-VL family,
    N=600, corrected location-cue prompt), language localization is far stronger
    for instruction-tuned checkpoints (instruct: 95.2% location-cue vs. 26.0–27.5%
    steered; agentic: 92.8% vs. 31.0–39.3% steered) — these models are much better
    at following an explicit spatial instruction than at having their attention
    forcibly redirected. The **base** (non-instruction-tuned) checkpoint shows the
    opposite pattern: its location-cue accuracy (5.7%) is *below* its own no-cue
    baseline (19.7%), while steering still improves over baseline (22.5–23.8%) —
    a model with no instruction-following ability cannot exploit a verbal cue at
    all, making attention steering the only intervention that helps it.

  ### 3.5 Representational cross-check: linear probing

  `linear_probe_causal_intervention.ipynb` extends the same three-condition design
  to an internal, non-generation-based measurement: instead of generating text and
  parsing a letter, it extracts the last transformer layer's hidden state at the
  final prompt token (immediately before generation would begin) under each of the
  three conditions, and trains a multinomial logistic-regression probe to predict
  the correct option letter directly from that vector — separately per condition.
  `Probe ATE = probe_accuracy(steered) − probe_accuracy(baseline)`, with macro
  one-vs-rest ROC-AUC reported alongside accuracy (chance = 0.25 accuracy / 0.5
  AUC for a 4-way problem) since accuracy alone can be misleading near chance.

  This asks whether the intervention changes the model's *internal* representation
  of the answer, independent of whether it can also verbalize that answer correctly
  under the MCQ generation format (itself an imperfect measure — see the "answer D"   
  default-bias finding for Qwen2-VL-7B-Instruct, and the near-chance MCQ performance
  of LLaVA/Gemma in the ambiguous, pre-location-fix prompt design). On Qwen3-VL-8B-Instruct
  (N=400, top-15 heads), the probe-based and generation-based causal effects agree
  closely (probe ATE +0.395 accuracy / +0.375 AUC vs. generation ATE +0.378
  accuracy), and the baseline condition's probe AUC (0.466) is at chance —
  confirming the no-cue baseline genuinely fails to *represent* the answer
  internally, not merely fails to verbalize a representation it already has.

  ---

  ## 4. CoT-trace Vis-Head extraction (reasoning/CoT-tuned models)

  Every discovery method above measures attention at a **single query
  position**: the last prompt token, immediately before generation begins. For
  a reasoning-tuned checkpoint (`Qwen3-VL-8B-Thinking`), that position precedes
  all of the model's actual reasoning about the image — applying the standard
  method there found unremarkable, Instruct-like scores, yet steering those
  top-ranked heads had **no significant causal effect** (steered accuracy
  0.240 vs. 0.280 baseline, p=0.45).

  `vis_head/gaze.py::collect_cot_trace_region_attention` instead measures
  attention **throughout the full generated reasoning trace**: it registers a
  forward hook on every layer (`vis_head/steering.py::register_attention_trackers`)
  that records each decode step's post-softmax attention row, runs a full
  generation (`max_new_tokens` large enough to cover the reasoning-about-the-image
  phase — 150 tokens in practice), then for each layer/head averages the
  attention mass landing on the target region's tokens across every recorded
  decode step. The result is **area-normalized** — divided by the target
  region's share of total image tokens — so a score of `1.0` means "attends to
  the target region exactly proportional to its size" (chance level); this
  controls for region-size differences the way raw attention mass would not.

  Validated causally against the standard method on the same held-out MCQ
  samples (`cot_vis_head_discovery_qwen3_thinking.ipynb`, N=150): the two
  methods' top-15 heads share **zero overlap**, cluster in different layer
  bands (CoT-trace: layers 15–26; final-query: mostly layers 0–10), and steering
  the CoT-trace heads lifts accuracy to **0.713** (p≈0) — the strongest causal
  steering effect found anywhere in this project, where the final-query
  ranking had found nothing. The standard method was not merely weaker for
  this model; it was identifying heads with no measurable causal role at all,
  because it was looking at the wrong moment in the forward pass.

  ---

  ## 5. Index of causal-intervention notebooks

  Every notebook below reuses the grid construction, discovery prompts, and
  three-condition (baseline / location-cue / steered) design documented above.
  Grouped by what varies.

  ### 5.1 MCQ prompt-format search (established the `mcq_prompt` default, Section 3.2)

  | notebook | what it tests | key result |
  |---|---|---|
  | `mcq_prompt_format_search.ipynb` | 5 location-free MCQ phrasings across LLaVA-1.5, LLaVA-1.6, Gemma-3-4B-it/pt | none beat chance -- the question was ill-posed (never named the target cell) |
  | `mcq_prompt_format_search_v2_located.ipynb` | 6 location-aware phrasings (word-position / row-col / bbox, verbose/short) on the same 4 checkpoints | word-position phrasing wins (mean acc 0.531); adopted as the project-wide `mcq_prompt` default |

  ### 5.2 Qwen2-VL family: training stage x phrasing x head-count x intervention strength

  | notebook | axis varied | N | key result |
  |---|---|---|---|
  | `top2_phrasings_across_model_stages.ipynb` | base / instruct / agentic x {find, identify} | 600 discovery + 600 MCQ | steering beats no-cue baseline everywhere (all p<0.07), but the corrected location-cue crushes steering for instruct (95.2%) and agentic (92.8%); only base (no instruction tuning) can't use the language cue at all, making steering its only usable lever |
  | `head_count_sweep_qwen2vl.ipynb` | number of steered heads: 5/15/30/50/80/150/200 (find phrasing, fixed alpha=10000) | 300 discovery + 300 MCQ per head count | non-monotonic -- no "more heads = better" trend for any stage; instruct collapses to 3% accuracy at k=50 (below chance), only partially recovering at k=200 |
  | `alpha_ablation_qwen2vl.ipynb` | intervention strength (`swap_bias`/alpha): 0 to 10000, log-spaced, fixed k=15 | 300 discovery + 300 MCQ per alpha | mid-range alpha (10-1000) actively hurts, often below alpha=0; only near-saturating alpha (3000-10000) recovers/exceeds no-intervention, and only clearly helps agentic -- steering behaves as a threshold/override effect, not a smooth amplifier |

  ### 5.3 Qwen3-VL family: post-training stage and reasoning-trace discovery

  | notebook | what it compares | key result |
  |---|---|---|
  | `vis_head_across_qwen3vl_stages.ipynb` | Instruct vs. Thinking (8B, matched architecture; no official base/agentic exists for Qwen3-VL) | Instruct: steering works normally (65% vs 28% baseline, p<1e-12); Thinking: no significant steering effect (p=0.45-0.77) using the standard discovery method -- motivated Section 4's CoT-trace method |
  | `cot_vis_head_discovery_qwen3_thinking.ipynb` | standard final-query discovery vs. new CoT-trace discovery (Section 4), same held-out eval set | zero head overlap between methods; CoT-trace ranking lifts steered accuracy to 71.3% (from 24.0% with the final-query ranking) -- resolves the Thinking null result as a discovery-method artifact, not a real immunity to steering |

  ### 5.4 Prompt phrasing (Qwen3-VL-8B-Instruct)

  `prompt_phrasing_vis_head_vs_causal.ipynb` -- expanded from the original 10 action-word
  phrasings to 18, adding a systematic politeness (plain/please/could-you/can-you) x
  sentence-form (imperative/wh-question/yes-no) grid (Section 2.1). Metrics extended from
  plain accuracy to accuracy/macro-F1/macro-AUC (read directly from letter-logit
  probabilities in a single forward pass, replacing generation+regex parsing) plus a
  per-phrasing linear probe (accuracy/F1/AUC) on the final-layer hidden state. Findings:
  `what_shows` ("What shows the {name}?") is the new best phrasing (0.707 steered acc,
  beating `find`/`identify` at 0.653); politeness modifiers never hurt and sometimes help;
  `classify` remains the sharpest attention/causal dissociation in the project (normal
  discovery scores, but steered accuracy statistically indistinguishable from baseline,
  confirmed at both the generation and probe level).

  ### 5.5 Cross-architecture: LLaVA

  `steer_llava_family.ipynb` -- the Vis-Head steering mechanism ported to
  `llava-hf/llava-1.5-7b-hf` (a structurally different VLM: fixed 576-token CLIP grid vs.
  Qwen's dynamic patch grid, Llama-based attention). MCQ format doesn't work for LLaVA
  (chance-level regardless of steering or head count, confirmed separately), so causal
  effect is measured via free-form generation + NLI semantic-similarity margin instead
  (the Section 3.5 probe methodology applied without the MCQ scaffolding). Result:
  mechanism transfers (real per-sample effects, degenerates to gibberish when
  over-steered -- confirming genuine causal pressure), but the aggregate effect
  (delta_mean=+0.026, p=0.227) is weaker and noisier than any Qwen-VL result -- a
  positive but non-significant finding at N=80.

  ### 5.6 Representational cross-check

  `linear_probe_causal_intervention.ipynb` (Section 3.5) -- extended from a
  two-condition (baseline/steered) to the full three-condition design, adding a linear
  probe on the location-cue condition's hidden states. Location-cue probe AUC reaches
  0.999 (vs. steered 0.841, baseline 0.466) -- confirming the "language beats steering"
  finding holds at the representational level, not just the generation-accuracy level.
