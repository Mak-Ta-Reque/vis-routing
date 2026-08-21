# Vis-Head Experiment Instructions

A dumb-proof, step-by-step guide for running the Visual Information Routing (VIR)
experiments in this repo. Read the TL;DR first, then follow the numbered steps in
order. Nothing here assumes you've seen the codebase before.

---

## 0. TL;DR — what we're doing and why

**Research question.** Vision-language models (VLMs) are believed to route visual
information through specific attention heads when answering a question about an
image. We call this **Visual Information Routing (VIR)**, and the heads responsible
**VIR heads**. Prior work found candidate VIR heads in *instruction-tuned* models
using **one fixed question template**, and identified them via attention-mass
scoring **without a controlled causal test** — i.e. no randomized-head-selection
baseline to check whether the identified heads' effect on the output is actually
different from intervening on an equal number of *random* heads at the same
layers. So it's unclear whether those heads are truly special routers or whether
any similarly-placed head would show a similar effect. This project fixes both
gaps at once: we (a) vary the question template (LFVQ) and training stage instead
of holding both fixed, and (b) always test causal effect against a **layer-matched
random-head control** (§1, §4 Step 6-7), not just against no-intervention baseline.
We don't know, going in:

1. **Whether VIR changes across training phases** — from multimodal pretraining (CPT),
   through supervised instruction tuning (SFT), to further post-training (MPO,
   RL) — call this the **training-stage axis**.
2. **Whether VIR changes with how you phrase the same visual question** — e.g. "find
   the dog" vs. "identify the dog" vs. "what shows the dog?" — call this the
   **linguistic formulation of visual questions (LFVQ) axis**.

**Method, in one paragraph.** For a given model checkpoint and a given question
phrasing, we (a) build synthetic images with a known ground-truth answer region, (b)
run the model once per image and record how much attention each head puts on the
correct region — this gives every (layer, head) a **VIR score**, (c) pick the
top-K highest-scoring heads as "VIR heads", (d) **causally test** them: forcibly
boost attention into the correct region / suppress attention elsewhere, only
through those K heads, and see if answer accuracy goes up — versus a **control**
that intervenes on the same number of heads per layer, just picked at random
instead of by VIR score. If VIR heads reliably beat both baseline (no intervention)
and the random-but-matched control, that's causal evidence they really do route
visual information. We repeat this across training stages and phrasings to answer
the two questions above.

**Models.** Three families, capped at ≤8B parameters:
- **Qwen2.5-VL** and **Qwen3-VL** (Qwen team)
- **Gemma-4** (Google) — has real CPT-only base checkpoints, good for training-stage axis
- **InternVL3.5** (OpenGVLab) — has 4 real training-stage checkpoints per size
  (Pretrained → Instruct → MPO → CascadeRL), the cleanest training-stage ladder we have

**What "finding" means for this project (fill in once results are final):**
- Result summary: `[TODO after full run]`
- Main finding about training stage: `[TODO]`
- Main finding about linguistic formulation: `[TODO]`
- Causal conclusion about VIR development: `[TODO]`

---

## 1. Terminology (read this before touching any notebook)

| Term | Meaning |
|---|---|
| **VIR / VIR head** | Visual Information Routing / an attention head (layer *l*, head *h*) that concentrates attention on the correct answer region of an image when the model is asked a visual question. |
| **LFVQ** | Linguistic Formulation of a Visual Question — the specific wording used to ask about the image (e.g. "find X" vs "what shows X?"). We test several LFVQ phrasings per experiment; VIR heads are re-scored for each phrasing. |
| **Training stage** | Where a checkpoint sits in its training pipeline: pretrained/CPT (no instruction tuning) → instruct/SFT → MPO/RL (further post-training). Compared across checkpoints of the *same model family and size*, never across families. |
| **Grid / cell / region** | Synthetic input image = several ImageNet photos tiled into an R×C grid (we use 2×2 = 4 cells). Each cell shows one object; the question asks about one specific cell (the **target region**). |
| **Discovery** | The unsupervised scoring pass: run N images through the model, record how much attention every head pays to the target region vs. elsewhere, no intervention performed. |
| **Causal / MCQ evaluation** | The supervised test: multiple-choice question ("which of these is shown? A/B/C/D"), model must name the correct cell's object; we compare accuracy with no intervention (**baseline**), with VIR heads steered (**VIR condition**), and with random layer-matched heads steered (**control condition**). |
| **VIR score (scoring methods)** | How a head's (layer,head) "visual-routing-ness" is computed from discovery data. Four supported: |
| — **raw** | Raw attention mass the head puts on the target region, averaged over all discovery samples. Simple, sometimes confounded by heads that are just "generically peaky" everywhere. |
| — **excess mass** | `raw_target_mass − (1/G)·Σ_over_all_regions(mass)`, i.e. mass above what a uniform-attending head would put on the target by chance (G = number of grid cells). Zero for heads that don't discriminate; large only for heads that both look at the image *and* concentrate on the right cell. |
| — **target share (mean-of-ratios)** | For each sample: `target_mass / total_visual_attention_mass`, then averaged across samples. Normalizes out how much a head attends to the image overall. |
| — **target share (pooled-ratio)** | Same idea but sum target mass and sum total mass separately across all samples first, then divide once. Less noisy than mean-of-ratios for low-attention heads. |
| **Top-K / K_FRACTION** | Number of heads selected as "VIR heads". Always expressed as a **fraction of total heads** (`K_FRACTION × n_layers × n_heads`, rounded, minimum 1) — never a fixed head count — so it's comparable across models of different sizes. Typical value: `K_FRACTION = 0.05` (5%), but see §6 for why you may need to sweep this per model. |
| **Layer-matched random control** | A random baseline that selects, from *each layer*, the exact same number of non-VIR heads as VIR selected in that layer. This is stricter than a uniform-random control across all layers, because it isolates "is VIR-selection better than chance *within the same depth profile*" rather than "is intervening on any heads better than nothing." |
| **R_CTRL** | Number of independent random control head-sets drawn (once, reused across all evaluation samples). Gives a distribution of control accuracies to compare VIR against; also used for a permutation p-value with floor `1/(R_CTRL+1)`. |
| **Relative depth (`l/L`)** | Layer index divided by total layer count. Always report depth this way, not as a raw layer index, so results are comparable across models with different depths. |
| **Baseline** | No attention intervention at all — the model's natural, unsteered behavior. |
| **Discovery sample / causal sample** | Discovery images and MCQ evaluation images are drawn from *disjoint* random seeds — never test causal effect on the same images used to find the heads. |

---

## 2. One-time environment setup

You only do this once per machine.

```bash
# 1. clone / cd into the repo
cd /mnt/abka03/Projects/vis-head   # or wherever you cloned it

# 2. create the main conda environment (covers Qwen2.5-VL, Qwen3-VL, Gemma-4)
conda create -n virheads python=3.10
conda activate virheads
pip install -r requirements.txt

# 3. InternVL3.5 needs a SEPARATE environment — its remote code is
#    incompatible with the transformers version everything else needs.
#    Do NOT try to run InternVL notebooks in `virheads`.
conda create -n internvl_env python=3.10
conda activate internvl_env
pip install "transformers==4.57.2" torch pillow numpy tqdm timm einops
conda deactivate
```

**Environment variables to set before running anything** (put these in your
`~/.bashrc` so you never forget):

```bash
# Claude API key — only needed for LLM-judge-based steering scripts
# (03/04/05_steer_*.py). NOT needed for discovery or MCQ causal eval.
export ANTHROPIC_API_KEY="sk-..."

# Where the ImageNet validation set lives on disk (used to build the
# synthetic grid images). Default if unset:
# /mnt/abka03/raw_data_download/imagenet
export VIR_IMAGENET_ROOT="/path/to/your/imagenet/val"

# Optional: redirect the HuggingFace model cache if you don't want
# multi-GB checkpoints landing in your home directory
export HF_HOME="/mnt/your_disk/hf_cache"
```

**Check `VIR_IMAGENET_ROOT` is valid before running anything:**
```bash
ls "$VIR_IMAGENET_ROOT" | head    # should show wnid-named class folders, e.g. n01440764
```
If you don't have ImageNet, ask a labmate for the path or download the val split
(`download_data.py` may already do this — check its `--help`).

**GPU discipline: this is a shared machine.** Before launching ANY experiment:
```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```
If another job is using the GPU, **wait or ask** — don't launch on top of someone
else's run. Notebooks in this repo assume they own the whole GPU (`cuda:0`).

---

## 3. Repo map — what lives where

```
vis_head/                          # the actual library, import from here
  vir.py                          # discovery scoring: raw / excess-mass / target-share, VHS, permutation tests
  steering.py                      # causal intervention: attention-mask hooks, layer-matched controls
  modeling.py                      # model loading + family-specific token/grid plumbing (Qwen, Gemma, InternVL)
  imagenet_grid.py                 # builds the synthetic R×C grid images + MCQ prompts/phrasings
  regions.py                       # maps grid cells -> visual token positions inside the sequence
  demo_adapters.py                 # Gemma-4-specific pooling-grid math (Gemma4VisionPooler)

multistage_vis_head_causality.ipynb   # MAIN notebook: full Task1(grid)/Task2(scoring)/Task3(control)
                                       # pipeline across Qwen2.5-VL, Gemma-4, InternVL3.5, up to 8B
vis_head_across_qwen3vl_stages.ipynb  # Qwen3-VL Instruct vs Thinking comparison (older, fixed K=15)
mulltistage_trained_models.md         # reference table of which HF checkpoint = which training stage,
                                       # for every family (READ THIS before picking checkpoints)
logs/                                  # output CSVs land here, one per checkpoint/notebook run
```

**Do not hand-roll a new grid/scoring/control implementation.** Everything you need
(grid construction, all 4 scoring methods, layer-matched controls, MCQ evaluation
harness) already exists in `vis_head/vir.py` and `vis_head/steering.py`. Import and
reuse it. If you think you need something new, check with the group first — it's
likely one function away from what's already there.

---

## 4. Step-by-step: running one full experiment

This is the loop you repeat for every (model family, training stage, LFVQ phrasing)
combination. Do it once by hand for one checkpoint before automating anything.

### Step 1 — Pick your checkpoint

Open `mulltistage_trained_models.md`. Find your family, find the training stage you
want (Pretrained / Instruct / MPO / CascadeRL, or family-specific equivalents), copy
the exact HuggingFace model ID. **Never guess a checkpoint name** — use the table.

### Step 2 — Pick your notebook / environment

| Family | Notebook to open | Conda env |
|---|---|---|
| Qwen2.5-VL, Qwen3-VL | `multistage_vis_head_causality.ipynb` (Section 1) | `virheads` |
| Gemma-4 | `multistage_vis_head_causality.ipynb` (Section 2) | `virheads` |
| InternVL3.5 | `multistage_vis_head_causality.ipynb` (Section 3) | `internvl_env` |

Launch Jupyter from the correct environment:
```bash
conda activate virheads        # or internvl_env for InternVL3.5
jupyter notebook
```
If your kernel is called "python3" in both environments — that's expected. What
matters is which env's `jupyter`/`python` binary you launched it with. When in
doubt, run `import sys; print(sys.executable)` in the first cell and confirm it
points into the right conda env's `bin/`.

### Step 3 — Set the checkpoint + LFVQ variables

In the notebook's shared config cell, set:

```python
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"   # from mulltistage_trained_models.md, Step 1
DISCOVERY_PROMPT = lambda name: f"What shows the {name}?"   # <-- this IS your LFVQ variable
```

**To run a different LFVQ phrasing**, just change `DISCOVERY_PROMPT`. Common
phrasings already validated in this project (see `top2_phrasings_across_model_stages.ipynb`
and `prompt_phrasing_vis_head_vs_causal.ipynb` for the full sweep):

```python
LFVQ_PHRASINGS = {
    "what_shows":  lambda name: f"What shows the {name}?",
    "find":        lambda name: f"Find the {name}.",
    "identify":    lambda name: f"Identify the {name}.",
    "look_at":     lambda name: f"Look at the {name}.",
    "point_to":    lambda name: f"Point to the {name}.",
}
```
Run the whole discovery→scoring→causal pipeline once per phrasing (a plain `for
tag, DISCOVERY_PROMPT in LFVQ_PHRASINGS.items(): ...` loop around Steps 4-7 below).
**Keep the phrasing fixed within a single discovery+causal run** — never mix
phrasings inside one score computation.

Also set the grid/scoring/control constants (already tuned, don't change without
checking with the group first — see §6 for why):

```python
CANVAS = 448        # px, square composite image
ROWS, COLS = 2, 2    # 2x2 grid = 4 cells (G=4)
K_FRACTION = 0.05    # top-5% of heads selected as "VIR heads" -- SEE CAVEAT in §6
R_CTRL = 20          # number of layer-matched random control draws
N_DISCOVERY = 1000   # images used to compute VIR scores
N_CAUSAL = 200        # images used for the MCQ causal evaluation (disjoint from discovery)
```

### Step 4 — Run discovery (unsupervised scoring pass)

This runs `N_DISCOVERY` grid images through the model once each, no intervention,
and records per-(layer,head) attention mass on the target region. Produces raw
`region_attn` arrays consumed by all 4 scoring methods in Step 5.

Expect this to take from a few minutes (small model, N=300) to ~30-60 min
(8B model, N=1000) on a single modern GPU.

### Step 5 — Compute VIR scores (pick your scoring method)

All four live in `vis_head/vir.py`:

```python
from vis_head.vir import per_sample_head_scores, pooled_target_share, rank_heads_by_score

scores = per_sample_head_scores(region_attn, target_region)
# scores["raw_target_mass"]        -> raw method
# scores["excess_mass"]            -> excess-mass method
# scores["target_share"]           -> mean-of-ratios target-share method
pooled = pooled_target_share(scores["raw_target_mass"].sum(...), scores["total_visual_attention"].sum(...))
# pooled                           -> pooled-ratio target-share method
```

**Run all 4 and keep all 4** where compute allows — don't pick just one up front.
Report top-K overlap between methods (`|A∩B|/K`) so you can see whether they agree.
Pick top-K heads per method:

```python
top_heads = rank_heads_by_score(scores["excess_mass"], k=K)   # or any of the 4 score arrays
```

### Step 6 — Build the layer-matched random control

```python
from vis_head.steering import layer_matched_random_controls, assert_layer_histogram_matches

controls = layer_matched_random_controls(
    selected_heads=top_heads, n_layers=n_layers, n_heads=n_heads,
    n_controls=R_CTRL, rng=np.random.RandomState(SEED),
)
for c in controls:
    assert_layer_histogram_matches(top_heads, c)   # must never fail — if it does, stop and debug
```
Draw controls **once** per (checkpoint, scoring method, LFVQ) combination, then
reuse the same `controls` list across every evaluation sample in Step 7. Never
redraw controls per-sample.

### Step 7 — Causal MCQ evaluation

For each of: baseline (no intervention), VIR heads, and each of the `R_CTRL`
control sets — run the same `N_CAUSAL` MCQ questions (4-way multiple choice: "which
of these is shown?") and record accuracy. This uses `vis_head/steering.py`'s
attention-mask hooks (boost target-region attention, suppress everything else,
only through the specified heads).

Report:
- baseline accuracy
- VIR accuracy, and McNemar p-value vs. baseline
- control mean ± std accuracy (across `R_CTRL` draws)
- permutation p-value of VIR vs. the control distribution, floored at `1/(R_CTRL+1)`

**A result only counts as a real causal VIR effect if VIR beats BOTH baseline AND
clearly separates from the control distribution.** Beating baseline alone is not
enough — see §6, this has already produced misleading conclusions once in this
project.

### Step 8 — Save results

Every per-checkpoint run should append one row to `logs/<experiment_name>_results.csv`
with (at minimum) columns: `model_id, training_stage, lfvq_phrasing, scoring_method,
K, R_CTRL, N_discovery, N_causal, baseline_acc, vir_acc, control_mean_acc,
control_std_acc, p_vs_baseline, p_vs_control`. This is what makes cross-checkpoint /
cross-phrasing comparison possible later — don't skip it, don't reinvent column
names per notebook.

---

## 5. Running the two axes of the study

### Axis A — training-stage comparison

Run Steps 1–8 above **for every training-stage checkpoint of the same model family
and size**, holding LFVQ phrasing fixed (use the best-performing phrasing from Axis
B once you know it, or `"what_shows"` as the default). Put each stage in its own
notebook run (separate cell execution or separate `.ipynb` if you want fully
isolated Jupyter kernels — recommended for InternVL3.5 since each of its 4 stages
is a full checkpoint reload). Then compare VIR-head identity (do the top-K heads
overlap across stages?) and causal effect size (does the VIR-vs-control gap grow,
shrink, or stay flat across CPT → SFT → MPO → RL?).

Recommended checkpoint ladder (see `mulltistage_trained_models.md` for exact IDs):
- **InternVL3.5** (cleanest 4-stage ladder): `-Pretrained` → `-Instruct` → `-MPO` → CascadeRL (no suffix)
- **Gemma-4**: base (`google/gemma-4-E{2,4}B`) vs `-it` instruct variant
- **Qwen2.5-VL / Qwen3-VL**: mostly ship instruct-only; check the table for any base VL checkpoints before assuming you can do this axis for Qwen

### Axis B — linguistic formulation comparison

Run Steps 1–8 **for every LFVQ phrasing in `LFVQ_PHRASINGS`**, holding the
checkpoint (training stage) fixed. Do this for at least one checkpoint per family.
Compare: does VIR head *identity* stay stable across phrasings (compute top-K
overlap `|A∩B|/K` between phrasings), and does causal *effect size* change?

### Combining both axes

Once both axes work independently, run the full cross product: every training
stage × every LFVQ phrasing, for each family/size. This is expensive — budget GPU
time accordingly (see §7 for compute estimates) and check with the group about
compute budget before launching the full grid unsupervised.

**Use a separate notebook (or clearly separated notebook section) per model
family** — do not try to interleave Qwen/Gemma/InternVL cells in one linear
notebook; they need different environments (InternVL) and different grid-token
math, and mixing them makes debugging much harder. Follow the Section 1/2/3
pattern already used in `multistage_vis_head_causality.ipynb`.

---

## 6. Known pitfalls (read before you "discover" a bug that's already known)

1. **`K_FRACTION` as a fixed 5% can dilute the effect for some checkpoints.** We
   found empirically (Qwen3-VL-8B-Instruct) that the top ~15 heads (≈1.3% of all
   heads) carry almost the entire causal effect; expanding to 5% (58 heads) pulled
   in many low-quality heads and cut steering accuracy roughly in half (0.70 → 0.59
   in one A/B test). **Sweep K per checkpoint if your effect looks weak** — don't
   assume 5% is right for every model before checking.
2. **A "significant vs. baseline" result is not enough.** In our 6-checkpoint pilot,
   4/6 checkpoints looked significant against baseline alone, but only 2/6 survived
   comparison against the layer-matched control. Always report both p-values.
3. **Generic prompts ("Describe the image.") produce near-zero visual attention
   in early/mid layers.** Discovery and causal prompts must reference the specific
   object/content you're asking about (e.g. "What shows the {name}?"), not a
   generic instruction, or your VIR scores will be meaningless noise.
4. **Naive "sum attention over all heads, argmax region" tests are confounded by
   positional/recency bias** — a head can spuriously look like it "found" whatever
   region is last in token order, regardless of content. If you're verifying grid
   token ordering, test with the same head across multiple region placements and
   check *consistency*, not a single placement.
5. **InternVL's `generate()` output contains only the newly generated tokens**, not
   prompt+generation like every other family here. If you write custom generation
   code for InternVL, you MUST use `prompt_length=0` when slicing — the shared
   `run_mcq_generate` helper already handles this, don't duplicate the logic.
6. **Gemma-4's grid-to-token mapping is derived mathematically** from
   `Gemma4VisionPooler`'s pooling kernel-index formula (in `demo_adapters.py`,
   `gemma4_output_grid`), not empirically verified head-by-head like Qwen/InternVL
   (Gemma-4's best verification head only hit 3/4 consistency, vs. Qwen/InternVL's
   clean 4/4 heads). Treat this as accepted-but-flagged, not gold-standard-verified.
7. **Discovery and causal-evaluation samples must use disjoint random seeds.**
   Check any new sampling code seeds discovery and evaluation RNGs differently
   (e.g. `SEED+100` vs `SEED+555`) — testing causal effect on the same images used
   to find the heads inflates your result.
8. **Check GPU memory before every launch** (`nvidia-smi --query-gpu=memory.used...`)
   — this is a shared box, an 8B model run left idle-but-loaded will block a
   labmate for hours.

---

## 7. Rough compute budget (single modern GPU, e.g. A100/A6000-class, ≤8B models)

| Step | N | Rough time |
|---|---|---|
| Discovery pass | 300 samples | ~5-10 min |
| Discovery pass | 1000 samples | ~30-60 min |
| Causal eval, baseline+VIR only | 200 samples × 2 conditions | ~5-10 min |
| Causal eval, + R_CTRL=20 controls | 200 samples × 22 conditions | ~1-2 hr |
| Full checkpoint (discovery N=1000 + causal N=200, R_CTRL=20) | — | ~1.5-3 hr |

Multiply by number of (training stage × LFVQ phrasing) combinations you're running.
**Start with R_CTRL=5 and N=300/200 for a fast pilot pass** to sanity-check the
pipeline before committing to a full-resolution run — this is what we did for the
first full pass across all three families.

---

## 8. Who to ask

If something in this doc contradicts what you see in the code, **trust the code**
and flag the discrepancy to the group — this doc can go stale. Start with
`mulltistage_trained_models.md` for checkpoint questions and
`multistage_vis_head_causality.ipynb`'s markdown cells (top of each section) for
the most current, working reference implementation of the full pipeline.
