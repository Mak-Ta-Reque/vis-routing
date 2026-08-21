# Evidence for the Four Contributions

Mapping this project's experiments onto the four claimed contributions, with the
actual tables and numbers produced. Model is Qwen2-VL-7B family unless noted;
Qwen3-VL-8B-Instruct is used for the dataset-comparison sections. Full reproduction
paths are in `experiments/README.md`.

---

## Contribution 1 — A causal framework combining visual selectivity with intervention-based measurement

**Selectivity component (observational):**

    VisRoutingScore(l,h) = E_(I,pi,t)~D [ sum_{k in T} alpha_{l,h}(q_last, k) ]

mean raw post-softmax attention a head places on the target region's image tokens
when queried about it, averaged over samples. `H = argtop_K VisRoutingScore(l,h)`.

**Intervention component (causal):** an attention-mask edit forcing selected heads'
attention onto/away from a target region (`boost_suppress`), then measuring the
effect on the model's own output — either free-text (semantic-similarity effect
size) or forced-choice (exact accuracy). Formal treatment (`mcq_causal_intervention.ipynb`):

    BPA = P(y_hat = y_t)
    CPA = P(y_hat = y_t | do(attn_H -> T))
    Delta = CPA - BPA          (an average treatment effect, not a correlational readout)

**Why selectivity alone isn't enough — measurement methods tested and cross-validated
against each other:**

| candidate selectivity/intervention metric | top-15 overlap w/ raw attention | causal effect (mean, ablation-necessity) | verdict |
|---|---|---|---|
| **raw attention** (baseline) | — | **0.555** | kept |
| value-weighted attention (attn x \|\|value\|\|, Kobayashi et al. 2020) | 2/15 (13%) | 0.153 (p=6.7e-05 vs raw) | rejected — relocates to late layers by depth, not function |
| single-head causal **sufficiency** (boost alone, forced choice between 2 targets) | 8/15 (53%) | 0.128 (p=8.7e-05 vs raw) | distinct from necessity, not a better *necessity* selector |
| gradient-based attribution patching | — | — | infeasible: OOM on 8B VLM vision-tower backward even with gradient checkpointing |

**Direct causal validation via forced-choice pointing game** (removes free-text
scoring ambiguity entirely — ground truth is a single correct letter):

| grid | no-cue baseline | steered | McNemar p |
|---|---|---|---|
| 2x2 (4 cells) | 0.290 [0.200, 0.380] | **0.570** [0.470, 0.670] | 1.8e-06 |
| 3x3 (9 cells) | 0.280 [0.190, 0.370] | **0.500** [0.400, 0.600] | 3.8e-05 |
| 4x4 (16 cells) | 0.260 [0.180, 0.350] | **0.470** [0.370, 0.570] | 1.2e-04 |

chance = 0.250. Steering roughly **doubles** accuracy over baseline at every grid
size, with the causal claim holding at high significance throughout.

**Random-head control (necessity of the causal claim itself):** ablating a random,
non-Vis-Head (Visual Routing Head) pool produces a much smaller effect than ablating the discovered VIR
heads, at every budget tested (paired Wilcoxon p from 2.5e-02 to 8.2e-06 across
budgets/models) — confirming the causal effect is Vis-Head-specific, not generic fragility
to any intervention.

---

## Contribution 2 — Evolution across pretrained, instruction-tuned, and agentic-tuned VLMs

Three checkpoints of the same base (`Qwen/Qwen2-VL-7B`): **base**, **instruct**
(`Qwen2-VL-7B-Instruct`), **agentic** (`ByteDance-Seed/UI-TARS-7B-DPO`, a GUI-agent
fine-tune of the same base — perception/grounding/action tuning, not chat tuning).

**Raw visual-routing score:**

| top-k | base mean | instruct mean | agentic mean | winner |
|---|---|---|---|---|
| 1 | 0.1176 | 0.5700 | **0.6847** | agentic |
| 10 | 0.0816 | 0.4941 | **0.6100** | agentic |
| 50 | 0.0535 | 0.3707 | **0.4298** | agentic |
| 100 | 0.0433 | 0.2976 | **0.3368** | agentic |

Overall mean: base 0.01275, instruct 0.06441, agentic 0.06504.

**Pairwise significance (Wilcoxon, paired by head):**

| comparison | heads higher | mean diff | p |
|---|---|---|---|
| instruct vs base | 704/784 (89.8%) | +0.05166 | 5.1e-103 |
| agentic vs base | 595/784 (75.9%) | +0.05229 | 1.8e-60 |
| agentic vs instruct | 271/784 (34.6%) | +0.00063 | 5.1e-11 |

**Head-identity overlap:** instruct and agentic converge on nearly the same heads
despite opposite training objectives (chat vs. GUI action), while both diverge from
base:

| pair | top-100 overlap | Spearman rho |
|---|---|---|
| base vs instruct | 40% | 0.804 |
| base vs agentic | 42% | 0.763 |
| **instruct vs agentic** | **93%** | **0.959** |

**Confound check — is the raw-score gain genuine targeting or just more engagement?**

| | base | instruct | agentic |
|---|---|---|---|
| mean total image attention | 0.0717 | 0.2283 (3.18x) | 0.1885 (2.63x) |
| mean concentration (on-target / total) | 0.1715 | 0.2120 | **0.2305** |
| top-100 concentration | 0.2032 | 0.3995 | **0.5061** |

All pairwise concentration Wilcoxon tests significant (p = 1.8e-87 to 1.2e-15).
Agentic shows the least raw-engagement inflation *and* the highest concentration —
the cleanest targeting signal of the three, consistent with GUI-agent training
directly rewarding precise localization.

**Causal-effect sweep (necessity), head budgets [50, 15, 5]:**

| heads | base | instruct | agentic |
|---|---|---|---|
| 50 | 0.681 | 0.509 | 0.542 |
| 15 | **0.624** | 0.316 | 0.371 |
| 5 | **0.451** | 0.186 | 0.261 |

Base > instruct significant at 15/5 heads (p=0.0011, p=0.0002); base > agentic
significant at 15/5 (p=0.0026, p=0.0042) — i.e. raw attention score and causal
necessity **rank the three models in opposite orders**.

**Random-head control resolves the apparent contradiction:** base's raw causal
advantage is largely (not entirely) explained by general fragility — its
random-head-ablation effect (0.29-0.32) is 3-10x higher than instruct/agentic's
(0.03-0.11) at every budget. The Vis-Head-*specific* excess (Vis-Head effect minus random
effect) is comparable across all three models, sometimes even smaller for base:

| heads | base gap | instruct gap | agentic gap |
|---|---|---|---|
| 50 | 0.390 | **0.436** | **0.449** |
| 15 | **0.305** | 0.244 | 0.270 |
| 5 | 0.140 | **0.159** | 0.151 |

---

## Contribution 3 — Invariance to surface form, but organized by reference type (semantic vs. spatial)

**Surface-form invariance (same reference type, different wording):**

| manipulation | top-50 overlap | Spearman rho | interpretation |
|---|---|---|---|
| 5 verb phrasings (find/locate/where_is/point_to/identify) | mean 0.804 pairwise | up to 0.995 | wording irrelevant |
| bare object name only ("chickadee") vs verb phrasings | 0.772 | 0.946-0.983 | instruction framing not needed |
| word-order shuffle ("Find the X." -> "X the Find.") | 0.880 | 0.990 | syntax irrelevant |
| object identity swap (always golden retriever vs. always persian cat, direct test) | top-50: 0.920, top-100: 0.910 | 0.992 | object identity irrelevant |
| object identity, 150-sample random split-half (near-disjoint categories) | top-100: 96% | 0.998 | confirms at scale |

**Reference-type dependence (semantic "find X" vs. spatial "cell N"):**

| pair | top-50 overlap |
|---|---|
| named-object phrasings, mean pairwise | 0.804 |
| named-object vs. **position-only** phrasing ("top-left", no object name) | **0.524** |
| ordinal sentence vs. bare cell number | 0.580 |
| ordinal sentence vs. bare (row, col) | 0.660 |
| bare cell number vs. bare (row, col) | 0.580 |
| comics (pure positional) vs. COCO (named-object), top-100 | 40-50%, rho 0.75-0.80 |
| comics vs. ImageNet-grid (named-object), top-100 | 40%, rho 0.79 |
| **COCO vs. ImageNet-grid (both named-object)**, top-100 | **67%**, rho 0.93 |

Two named-object datasets (COCO, ImageNet-grid) agree with each other far more than
either agrees with the positional dataset (comics) — reference type, not domain, is
the organizing variable.

**Ruling out the confound that this is just a digit/token artifact, not reference
type** — bare position with no digit vs. object query with an irrelevant digit
injected:

| pair (same TYPE, digit toggled) | overlap |
|---|---|
| position_no_digit vs. position_digit | **0.640** |
| object_with_digit vs. object_no_digit | **0.740** |
| pair (same DIGIT presence, TYPE toggled) | overlap |
| position_digit vs. object_with_digit (both have digits) | 0.420 |
| position_no_digit vs. object_with_digit (digit mismatch) | 0.320 |

Matching on reference type beats matching on digit presence in every comparison —
the split is genuinely about what kind of reference is being resolved.

**Mismatched-object probe** ("Find the cat." on a grid guaranteed to contain no
cat, scored by attention peak-concentration since there's no valid ground-truth
cell): the impossible-reference condition still resembles the matched-object
condition (0.86 overlap, top-50) more than the positional condition (0.66-0.70) —
the semantic-reference circuit fires on *attempting* grounding, not on its success.

---

## Contribution 4 — Controlled linguistic framing modulates causal contribution without changing image or task

Same image, same target region, same options/answer — only the prompt's framing
changes.

| condition | attention identity (overlap w/ minimal prompt) | mean causal effect | change from minimal |
|---|---|---|---|
| ImageNet-grid + "Look at the picture." filler | 0.84 (top-100 0.89), rho 0.989 | 0.303 | **-29%**, p=0.011 (significant) |
| ImageNet-grid + "Look carefully at this picture." filler | 0.84-0.89, rho 0.986 | 0.315 | **-26%**, p=0.158 (same direction, not sig. alone) |
| Comics + matched verbose padding (preamble + postscript) | 0.83-0.90 (top-100), rho 0.972 | 0.580 | **+5%**, p=0.171 (**no drop** — opposite pattern) |

In every case, **attention identity is essentially unchanged** by the framing
(0.83-0.90 overlap regardless of dataset) — the same heads engage. But **causal
reliance is not invariant**: two independent ImageNet-grid filler manipulations both
show a real reduction in effect size (consistent direction and magnitude, ~26-29%),
while the same manipulation on comics shows no reduction at all. This demonstrates
the intervention's *causal contribution* is separable from, and can be modulated
independently of, which heads the observational selectivity score identifies — and
that this modulation is itself task-type-dependent (named-object localization is
diluted by irrelevant context in a way pure positional/ordinal reference is not).
