"""Reproduce the exact same pointing-game demo search from
mcq_causal_intervention.ipynb (identical model, head set, seeds, and search
logic) standalone, so the figure can show the full prompt text for both
conditions without re-running the entire 900-generation 3-way evaluation.
"""
import sys
from pathlib import Path

REPO_ROOT = Path("/mnt/abka03/Projects/vis-head")
sys.path.insert(0, str(REPO_ROOT))

import re
import numpy as np
import matplotlib.pyplot as plt

from vis_head.common import DEFAULT_MODEL_ID, DEFAULT_SEED, make_output_paths
from vis_head.vir import load_head_ranking
from vis_head.imagenet_grid import DEFAULT_IMAGENET_ROOT, draw_cell_labels, list_val_class_dirs, load_class_names, sample_grid
from vis_head.modeling import decode_generated_text, find_image_token_range, load_model_and_processor, model_dims, prepare_inputs, run_generation
from vis_head.plots import draw_attention_overlay
from vis_head.regions import assign_grid_cells_to_tokens, region_positions_from_ids
from vis_head.steering import group_heads_by_layer, intervention_positions, make_static_attention_mask_hook, register_mask_hooks, remove_handles

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
DEVICE = "cuda:0"
SEED = DEFAULT_SEED
VIS_HEAD_RANKING_PATH = REPO_ROOT / "logs" / "vis_head_discovery_qwen2vl_instruct" / "vis_head_ranking.json"
TOP_K_HEADS = 30
GRID_SIZES = [2, 3, 4]
CELL_SIZE = 256
N_OPTIONS = 4
MAX_NEW_TOKENS = 6
OPTION_LETTERS = ["A", "B", "C", "D"][:N_OPTIONS]

model, processor = load_model_and_processor(model_id=MODEL_ID, device=DEVICE)
n_layers, n_query_heads, spatial_merge = model_dims(model)

vis_head = load_head_ranking(VIS_HEAD_RANKING_PATH, top_k=TOP_K_HEADS)
vir_by_layer = group_heads_by_layer(vis_head)

imagenet_class_dirs = list_val_class_dirs(DEFAULT_IMAGENET_ROOT)
imagenet_class_names = load_class_names(DEFAULT_IMAGENET_ROOT)


def build_mcq_sample(rows, cols, rng):
    n_cells = rows * cols
    grid = sample_grid(rows=rows, cols=cols, cell_size=CELL_SIZE, rng=rng,
                        class_dirs=imagenet_class_dirs, class_names=imagenet_class_names)
    target_cell = int(rng.randint(n_cells))
    correct_name = grid.cell_names[target_cell]
    other_names = [n for i, n in enumerate(grid.cell_names) if i != target_cell]
    n_distractors = min(N_OPTIONS - 1, len(other_names))
    distractor_idx = rng.choice(len(other_names), size=n_distractors, replace=False)
    distractors = [other_names[i] for i in distractor_idx]
    options = [correct_name] + distractors
    order = rng.permutation(len(options))
    options = [options[i] for i in order]
    correct_letter = OPTION_LETTERS[int(np.where(order == 0)[0][0])]
    option_lines = "  ".join(f"{letter}) {name}" for letter, name in zip(OPTION_LETTERS[:len(options)], options))
    prompt = f"Which of the following is shown in this image? {option_lines}. Answer with only the letter."
    return {"grid": grid, "target_cell": target_cell, "options": options, "correct_letter": correct_letter,
            "prompt": prompt, "rows": rows, "cols": cols}


def add_location_cue(sample):
    rows, cols = sample["rows"], sample["cols"]
    row, col = divmod(sample["target_cell"], cols)
    option_lines = "  ".join(f"{letter}) {name}" for letter, name in
                              zip(OPTION_LETTERS[:len(sample["options"])], sample["options"]))
    prompt = (
        f"Look at the grid cell at row {row + 1}, column {col + 1} "
        f"(counting from the top-left, rows top to bottom, columns left to right). "
        f"Which of the following is shown in that cell? {option_lines}. "
        f"Answer with only the letter."
    )
    new_sample = dict(sample)
    new_sample["prompt"] = prompt
    return new_sample


def extract_letter(text, valid_letters):
    match = re.search(r"\b([" + "".join(valid_letters) + r"])\b", text.upper())
    return match.group(1) if match else None


def run_mcq(sample, steer):
    rows, cols = sample["rows"], sample["cols"]
    n_cells = rows * cols
    inputs = prepare_inputs(processor, sample["grid"].grid, sample["prompt"], DEVICE)
    prompt_length = int(inputs["input_ids"].shape[1])

    handles = []
    if steer:
        img_start, img_end = find_image_token_range(inputs, processor)
        region_ids, _ = assign_grid_cells_to_tokens(
            image_grid_thw=inputs["image_grid_thw"], rows=rows, cols=cols, spatial_merge=spatial_merge)
        positions = region_positions_from_ids(img_start=img_start, region_ids=region_ids, n_regions=n_cells)
        target_positions = positions[sample["target_cell"]]
        other_positions = [p for i in range(n_cells) if i != sample["target_cell"] for p in positions[i]]
        suppress_positions, boost_positions, pad = intervention_positions(
            mode="boost_suppress", target_positions=target_positions, other_image_positions=other_positions,
            img_start=img_start, img_end=img_end, prompt_length=prompt_length)
        hook_by_layer = {
            layer_idx: make_static_attention_mask_hook(
                head_indices=heads, suppress_positions=suppress_positions, boost_positions=boost_positions,
                n_query_heads=n_query_heads, device=DEVICE, decode_only=False, pad_with_suppress=pad)
            for layer_idx, heads in vir_by_layer.items()
        }
        handles = register_mask_hooks(model, hook_by_layer)

    try:
        sequences = run_generation(model=model, inputs=inputs, max_new_tokens=MAX_NEW_TOKENS)
    finally:
        remove_handles(handles)

    text = decode_generated_text(processor, sequences, prompt_length)
    valid_letters = OPTION_LETTERS[: len(sample["options"])]
    predicted = extract_letter(text, valid_letters)
    correct = predicted == sample["correct_letter"]
    return {"text": text, "predicted": predicted, "correct": correct}


def find_language_fails_steering_succeeds(grid_sizes, max_tries=200, seed_offset=12345):
    for k in grid_sizes:
        rng = np.random.RandomState(SEED + seed_offset + k)
        for _ in range(max_tries):
            sample = build_mcq_sample(k, k, rng)
            loc_result = run_mcq(add_location_cue(sample), steer=False)
            if loc_result["correct"]:
                continue
            steer_result = run_mcq(sample, steer=True)
            if steer_result["correct"]:
                return sample, loc_result, steer_result
    return None


found = find_language_fails_steering_succeeds(GRID_SIZES)
assert found is not None, "no example found"
demo_sample, loc_result, steer_result = found
rows, cols = demo_sample["rows"], demo_sample["cols"]
target_cell = demo_sample["target_cell"]
row, col = divmod(target_cell, cols)
location_prompt = add_location_cue(demo_sample)["prompt"]

print("Found example:")
print(f"  grid={rows}x{cols}  target_cell={target_cell} (row {row+1}, col {col+1})")
print(f"  correct answer: {demo_sample['correct_letter']} = {demo_sample['grid'].cell_names[target_cell]!r}")
print(f"  options: {demo_sample['options']}")
print(f"  no-cue prompt: {demo_sample['prompt']!r}")
print(f"  location-cue prompt: {location_prompt!r}")
print(f"  location-cue result: {loc_result}")
print(f"  steered result: {steer_result}")

inputs = prepare_inputs(processor, demo_sample["grid"].grid, demo_sample["prompt"], DEVICE)
img_start, img_end = find_image_token_range(inputs, processor)
region_ids, grid_shape = assign_grid_cells_to_tokens(
    image_grid_thw=inputs["image_grid_thw"], rows=rows, cols=cols, spatial_merge=spatial_merge)
n_cells = rows * cols
cell_mask = (region_ids == target_cell).astype(np.float64).reshape(grid_shape[1], grid_shape[2])

fig, axes = plt.subplots(1, 2, figsize=(12, 7.5))
axes[0].imshow(draw_cell_labels(demo_sample["grid"]))
axes[0].set_title(f"Target: cell {target_cell + 1} (row {row + 1}, col {col + 1}) = {demo_sample['grid'].cell_names[target_cell]!r}\n"
                   f"Options: {demo_sample['options']}")
axes[0].axis("off")

draw_attention_overlay(axes[1], image=demo_sample["grid"].grid, heatmap=cell_mask,
                       title="Attention forced here by the intervention\n(the \"point\" in the pointing game)")

import textwrap
loc_wrapped = "\n    ".join(textwrap.wrap(location_prompt, width=95))
nocue_wrapped = "\n    ".join(textwrap.wrap(demo_sample["prompt"], width=95))

fig.text(0.02, -0.02,
          f"LOCATION-CUE BASELINE (no attention intervention) -- prompt given to the model:\n"
          f"    {loc_wrapped}\n"
          f"  model output: {loc_result['text']!r}  ->  predicted {loc_result['predicted']!r}  "
          f"(correct = {demo_sample['correct_letter']!r})  ->  WRONG\n\n"
          f"ATTENTION-STEERED (vis heads forced onto the target cell) -- prompt given to the model:\n"
          f"    {nocue_wrapped}\n"
          f"  model output: {steer_result['text']!r}  ->  predicted {steer_result['predicted']!r}  "
          f"(correct = {demo_sample['correct_letter']!r})  ->  CORRECT",
          fontsize=9.5, family="monospace", va="top")
plt.tight_layout()
outputs = make_output_paths("mcq_causal_intervention")
plt.savefig(outputs.figures_dir / "pointing_game_language_fails_steering_succeeds.pdf", bbox_inches="tight")
plt.savefig("/tmp/pointing_game_demo_with_prompts.png", bbox_inches="tight", dpi=150)
print("saved figure")
