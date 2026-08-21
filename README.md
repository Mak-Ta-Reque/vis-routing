# Vis-Head: Visual Routing Heads in Vision-Language Models

An investigation into **Visual Routing Heads (VRH)** — attention heads that route a
VLM's generation toward whatever region of an image is currently the subject of
description, and can be causally redirected to make the model describe somewhere
else instead. Starting from an existing head-discovery/steering codebase for the
Qwen3-VL family, this project builds out a substantially larger experimental
program: new datasets, a cross-model developmental study, controlled confound
isolation, causal-metric validation, and a formal forced-choice causal-intervention
evaluation.

## What's original here

- **Two new VIR-discovery datasets** beyond the original comic-strip corpus:
  a COCO-based dataset (`vis_head/coco.py`, `build_coco_vis_head_dataset.ipynb`) and a
  controllable ImageNet-grid dataset (`vis_head/imagenet_grid.py`,
  `imagenet_grid_vis_heads.ipynb`) that independently randomizes object identity and
  reference type per sample.
- **A cross-model developmental study** tracing VRH configuration across pretrained,
  instruction-tuned, and agentic-tuned checkpoints of the same base model
  (`compare_base_vs_instruct_vis_heads.ipynb`) — including a random-head ablation
  control that isolates genuine VRH-specific causal effect from general model
  fragility.
- **A dual-circuit finding**: VRH configuration is close to invariant across surface
  linguistic variation (verb choice, word order, object identity — see
  `imagenet_grid_vis_heads.ipynb` and `experiments/02,03,10`) but splits meaningfully
  by *reference type* — semantic ("find the dog") vs. spatial ("look at cell 3") — with
  the split confirmed to be about reference type and not a token-level artifact
  (`experiments/05_digit_confound_control.py`).
- **A demonstration that linguistic framing modulates causal contribution independent
  of routing-head identity**: irrelevant framing text leaves *which* heads route
  attention unchanged but measurably weakens their *causal* effect on the output, in a
  task-dependent way (`experiments/06,07`).
- **Causal-metric validation**: head-to-head tests showing raw attention outperforms
  value-weighted attention and single-head sufficiency-based selection as a necessity
  metric (`experiments/08,09`), and a semantic (NLI-based), model-independent effect-
  size metric (`vis_head/judge.py::semantic_similarity`) replacing lexical-overlap
  scoring.
- **A formal, forced-choice causal-intervention protocol** (`mcq_causal_intervention.ipynb`)
  — a controlled variant of the Pointing Game with an explicit average-treatment-
  effect formulation, comparing no-cue, language-location-cue, and attention-steered
  conditions with an unambiguous ground truth.

Full experiment-by-experiment index with findings and reproduction paths:
[`experiments/README.md`](experiments/README.md) and
[`00_experiment_index.ipynb`](00_experiment_index.ipynb). Evidence tables organized
by contribution: [`experiments/CONTRIBUTIONS_EVIDENCE.md`](experiments/CONTRIBUTIONS_EVIDENCE.md).

## Setup
```
conda create -n virheads python=3.10
conda activate virheads
pip install -r requirements.txt
```
The steering evaluations use Claude as a judge, so you need to export your
`ANTHROPIC_API_KEY` (ideally to your bashrc). Discovery and the trajectory plots need
no API key. The library package is internally named `vis_head/` for historical
reasons (see Acknowledgments); this doc refers to the heads it finds as Visual
Routing Heads.

## Data
To download the 500-strip comics dataset (500 six-panel strips, per-panel captions):
```
python download_data.py
```
This exports comics to `data/comics/`, one folder per strip. Point any script at your
own comics with `--comics-root` or `VIR_COMICS_ROOT`. For the COCO and ImageNet-grid
datasets, see `build_coco_vis_head_dataset.ipynb` and `imagenet_grid_vis_heads.ipynb`.

## Discovering Visual Routing Heads
```
python 01_discover_vis_heads.py --device cuda:0 --n-samples 500
```
No training, no labels — one forward pass per query. Ranking saved to
`logs/vis_head_discovery/vis_head_ranking.json`; every other script picks it up from
there. See `experiments/README.md` for the COCO/ImageNet-grid/cross-model/agentic
variants of discovery.

## Steering
Steering adds a single pre-softmax bias on the routing heads' attention: boost the
target region's image tokens, suppress the rest.
```
python 03_steer_vqa.py --device cuda:0        # VQA steering, judged protocol
python 04_steer_static_narration.py --device cuda:0   # ambiguous-question steering
python 05_steer_dynamic_narration.py --device cuda:0  # mid-generation target switching
```
For the forced-choice, ground-truth-verifiable version of this evaluation (no LLM
judge needed), see `mcq_causal_intervention.ipynb`.

## Interactive Steering Notebook
Open `interactive_steering.ipynb` (or `interactive_steering_qwen2vl.ipynb` for a
base/instruct/agentic switch, or the COCO/comics `DATASET_SOURCE` flag) from the
repo root: pick a target region from a dropdown, type a switching schedule, or drag a
box over any image and steer the description to it.

## Acknowledgments

This codebase builds on an existing VIR-head discovery/steering implementation for
Qwen3-VL, developed for prior published research on attention-based visual
description in VLMs. That prior work established the base discovery method (raw
attention VIR scoring on comic-panel queries) and the boost/suppress steering
intervention; this project's contribution is everything described above built on top
of it. If citing the underlying discovery/steering mechanism specifically, please
credit the original publication.
