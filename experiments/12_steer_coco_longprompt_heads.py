"""Long-prompt COCO gaze-head discovery for Qwen2-VL-7B-Instruct (N=400,
"Look carefully at this picture. Find the {category}. Answer briefly."),
then the same steering demo (same 10 multi-object images, same seed, same
top-30 head budget) used for the short-prompt-COCO-heads and comics-heads
runs, for a direct 3-way comparison.
"""
import sys
from pathlib import Path

REPO_ROOT = Path("/mnt/abka03/Projects/vis-head")
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from vis_head.common import DEFAULT_SEED, dump_json, make_output_paths
from vis_head.coco import DEFAULT_COCO_ROOT, DEFAULT_COCO_SPLIT, SelectionConfig, vis_head_score_from_patch_mask, load_coco_index, mask_to_patch_occupancy, segmentation_to_mask, select_target_objects
from vis_head.gaze import collect_last_query_attentions, rank_heads_by_score
from vis_head.judge import semantic_similarity
from vis_head.modeling import decode_generated_text, find_image_token_range, load_model_and_processor, model_dims, prepare_inputs, run_generation
from vis_head.regions import bbox_to_token_positions, get_merged_grid_shape
from vis_head.steering import group_heads_by_layer, intervention_positions, make_static_attention_mask_hook, register_mask_hooks, remove_handles

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
DEVICE = "cuda:0"
SEED = DEFAULT_SEED
N_DISCOVERY_SAMPLES = 400
TOP_K_HEADS = 30
N_DEMO_SAMPLES = 10
CAPTION_PROMPT = "Describe what is in this image. Answer in one short sentence."
MAX_NEW_TOKENS = 30
SIM_THRESHOLD = 0.5


def long_prompt(name: str) -> str:
    return f"Look carefully at this picture. Find the {name}. Answer briefly."


model, processor = load_model_and_processor(model_id=MODEL_ID, device=DEVICE)
n_layers, n_heads, spatial_merge = model_dims(model)
print(f"{n_layers} layers x {n_heads} heads")

coco_index = load_coco_index(coco_root=DEFAULT_COCO_ROOT, split=DEFAULT_COCO_SPLIT)


# ---------------------------- long-prompt discovery, N=400 ----------------------------
coco_config = SelectionConfig(unique_category_only=True, min_area=400.0, max_samples=N_DISCOVERY_SAMPLES, seed=SEED)
coco_targets = select_target_objects(coco_index, coco_config)
print(f"Discovery samples: {len(coco_targets)}")

score_sum = np.zeros((n_layers, n_heads), dtype=np.float64)
valid = 0
for target in tqdm(coco_targets, desc="Gaze discovery [Qwen2-VL-Instruct / COCO / long prompt]"):
    image_path = Path(target.image_path)
    if not image_path.exists():
        continue
    try:
        image = Image.open(image_path).convert("RGB")
        prompt = long_prompt(target.category_name)
        inputs = prepare_inputs(processor, image, prompt, DEVICE)
        img_start, img_end = find_image_token_range(inputs, processor)
        grid_shape = get_merged_grid_shape(inputs["image_grid_thw"], spatial_merge)
        mask = segmentation_to_mask(target.segmentation, target.image_height, target.image_width)
        patch_mask = mask_to_patch_occupancy(mask, grid_shape)
        patch_mask_flat = patch_mask.reshape(-1).tolist()
        attn_at_query = collect_last_query_attentions(model, inputs)
        score_sum += vis_head_score_from_patch_mask(attn_at_query, img_start, img_end, patch_mask_flat)
        valid += 1
    except Exception as exc:
        print(f"Skipping {target.image_path}: {exc}")

vis_head_scores = (score_sum / max(valid, 1)).astype(np.float32)
gaze_ranked = rank_heads_by_score(vis_head_scores)
print(f"valid samples: {valid}/{len(coco_targets)}")
print(f"Top-10 heads: {[(r['layer'], r['head']) for r in gaze_ranked[:10]]}")
print(f"mean raw score: {vis_head_scores.mean():.5f}")

outputs = make_output_paths("vis_head_discovery_qwen2vl_instruct_coco_longprompt")
np.save(outputs.logs_dir / "vis_head_scores.npy", vis_head_scores)
dump_json(outputs.logs_dir / "vis_head_ranking.json", gaze_ranked)

vis_head = [(r["layer"], r["head"]) for r in gaze_ranked[:TOP_K_HEADS]]
gaze_by_layer = group_heads_by_layer(vis_head)
print(f"Using top-{TOP_K_HEADS} heads across {len(gaze_by_layer)} layers for steering.")


# ---------------------------- same 10 demo images as before ----------------------------
def find_multi_object_images(coco_index, n_needed, seed, min_area=1500.0):
    rng = np.random.RandomState(seed)
    image_ids = list(coco_index.annotations_by_image.keys())
    rng.shuffle(image_ids)
    picks = []
    for image_id in image_ids:
        anns = [a for a in coco_index.annotations_by_image[image_id]
                if not a.get("iscrowd", 0) and isinstance(a.get("segmentation"), list)
                and a.get("segmentation") and float(a.get("area", 0)) >= min_area]
        categories = {}
        for a in anns:
            cat = coco_index.categories_by_id.get(a["category_id"])
            if cat and cat not in categories:
                categories[cat] = a
        if len(categories) < 2:
            continue
        cat_names = list(categories.keys())
        rng.shuffle(cat_names)
        original_cat, alternate_cat = cat_names[0], cat_names[1]
        image_info = coco_index.images_by_id[image_id]
        picks.append({
            "image_id": image_id,
            "image_path": str(coco_index.images_root / image_info["file_name"]),
            "original": {"category": original_cat, "ann": categories[original_cat]},
            "alternate": {"category": alternate_cat, "ann": categories[alternate_cat]},
        })
        if len(picks) >= n_needed:
            break
    return picks


demo_samples = find_multi_object_images(coco_index, N_DEMO_SAMPLES, seed=SEED + 1)   # same seed -> same 10 images as before
print(f"\nSame {len(demo_samples)} demo images as the previous two steering runs.")


def caption_with_steering(image, prompt, target_bbox_xyxy, heads_by_layer, n_query_heads):
    inputs = prepare_inputs(processor, image, prompt, DEVICE)
    img_start, img_end = find_image_token_range(inputs, processor)
    grid_shape = get_merged_grid_shape(inputs["image_grid_thw"], spatial_merge)
    _, target_positions = bbox_to_token_positions(target_bbox_xyxy, grid_shape, image.size, img_start)
    other_positions = [p for p in range(img_start, img_end) if p not in set(target_positions)]
    prompt_length = int(inputs["input_ids"].shape[1])

    suppress_positions, boost_positions, pad = intervention_positions(
        mode="boost_suppress", target_positions=target_positions, other_image_positions=other_positions,
        img_start=img_start, img_end=img_end, prompt_length=prompt_length)
    hook_by_layer = {
        layer_idx: make_static_attention_mask_hook(
            head_indices=heads, suppress_positions=suppress_positions, boost_positions=boost_positions,
            n_query_heads=n_query_heads, device=DEVICE, decode_only=False, pad_with_suppress=pad)
        for layer_idx, heads in heads_by_layer.items()
    }
    handles = register_mask_hooks(model, hook_by_layer)
    try:
        sequences = run_generation(model=model, inputs=inputs, max_new_tokens=MAX_NEW_TOKENS)
    finally:
        remove_handles(handles)
    return decode_generated_text(processor, sequences, prompt_length)


def mentions_category(text, category):
    text_l = text.lower()
    words = [w for w in category.lower().replace("-", " ").split() if len(w) > 2]
    return any(w in text_l for w in words) if words else category.lower() in text_l


results = []
for sample in tqdm(demo_samples, desc="Steering demo [COCO long-prompt heads]"):
    image = Image.open(sample["image_path"]).convert("RGB")

    baseline_inputs = prepare_inputs(processor, image, CAPTION_PROMPT, DEVICE)
    baseline_prompt_len = int(baseline_inputs["input_ids"].shape[1])
    baseline_sequences = run_generation(model=model, inputs=baseline_inputs, max_new_tokens=MAX_NEW_TOKENS)
    baseline_text = decode_generated_text(processor, baseline_sequences, baseline_prompt_len)

    x, y, w, h = sample["alternate"]["ann"]["bbox"]
    alt_bbox_xyxy = (x, y, x + w, y + h)
    steered_text = caption_with_steering(image, CAPTION_PROMPT, alt_bbox_xyxy, gaze_by_layer, n_heads)

    alt_cat = sample["alternate"]["category"]
    ref_caption = f"a photo of a {alt_cat}."
    steered_sim = semantic_similarity(steered_text, ref_caption, device="cpu")
    baseline_sim = semantic_similarity(baseline_text, ref_caption, device="cpu")
    matches = mentions_category(steered_text, alt_cat) or steered_sim >= SIM_THRESHOLD

    results.append({
        "image_id": sample["image_id"], "original": sample["original"]["category"], "alternate": alt_cat,
        "baseline_text": baseline_text, "steered_text": steered_text,
        "steered_sim": steered_sim, "baseline_sim": baseline_sim, "match": matches,
    })

print(f"\n{'image_id':>10s}  {'original':>12s}  {'alternate':>12s}  {'match?':>7s}  {'sim(steered,alt)':>17s}  {'sim(base,alt)':>14s}")
for r in results:
    print(f"{r['image_id']:>10d}  {r['original']:>12s}  {r['alternate']:>12s}  "
          f"{'YES' if r['match'] else 'no':>7s}  {r['steered_sim']:17.3f}  {r['baseline_sim']:14.3f}")
    print(f"    baseline: {r['baseline_text']!r}")
    print(f"    steered : {r['steered_text']!r}")

n_match = sum(r["match"] for r in results)
print(f"\nCOCO long-prompt-heads steering redirected the caption to the alternate object in {n_match}/{len(results)} samples ({100*n_match/len(results):.0f}%).")

dump_json(REPO_ROOT / "logs" / "steer_demo_coco_longprompt" / "summary.json", results)
print("\nDONE")
