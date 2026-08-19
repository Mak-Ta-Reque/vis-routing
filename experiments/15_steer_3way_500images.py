"""Scaled-up version of the 3-way steering head-to-head: 500 multi-object
COCO images instead of 10, comparing comics-, COCO-longprompt-, and
ImageNet-longprompt-discovered vis heads (Qwen2-VL-7B-Instruct, top-30
heads each) on the same images -- baseline caption computed once per image
and reused across all three conditions. Bootstrap CIs + paired Wilcoxon on
the continuous similarity scores (more sensitive than the binary match rate
alone, same methodology used throughout this project).
"""
import sys
from pathlib import Path

REPO_ROOT = Path("/mnt/abka03/Projects/vis-head")
sys.path.insert(0, str(REPO_ROOT))

import json
import numpy as np
from PIL import Image
from scipy import stats
from tqdm.auto import tqdm

from vis_head.common import DEFAULT_SEED, dump_json
from vis_head.coco import DEFAULT_COCO_ROOT, DEFAULT_COCO_SPLIT, load_coco_index
from vis_head.judge import bootstrap_ci, semantic_similarity
from vis_head.modeling import decode_generated_text, find_image_token_range, load_model_and_processor, model_dims, prepare_inputs, run_generation
from vis_head.regions import bbox_to_token_positions, get_merged_grid_shape
from vis_head.steering import group_heads_by_layer, intervention_positions, make_static_attention_mask_hook, register_mask_hooks, remove_handles

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
DEVICE = "cuda:0"
SEED = DEFAULT_SEED
TOP_K_HEADS = 30
N_DEMO_SAMPLES = 500
CAPTION_PROMPT = "Describe what is in this image. Answer in one short sentence."
MAX_NEW_TOKENS = 30
SIM_THRESHOLD = 0.5

model, processor = load_model_and_processor(model_id=MODEL_ID, device=DEVICE)
n_layers, n_heads, spatial_merge = model_dims(model)
print(f"{n_layers} layers x {n_heads} heads")

head_sets = {}
for tag, log_dir in [
    ("comics", "vis_head_discovery_qwen2vl_instruct"),
    ("coco_lp", "vis_head_discovery_qwen2vl_instruct_coco_longprompt"),
    ("imagenet_lp", "vis_head_discovery_qwen2vl_instruct_imagenet_longprompt"),
]:
    ranking = json.loads((REPO_ROOT / "logs" / log_dir / "vis_head_ranking.json").read_text())
    heads = [(r["layer"], r["head"]) for r in ranking[:TOP_K_HEADS]]
    head_sets[tag] = group_heads_by_layer(heads)
    print(f"[{tag}] top-{TOP_K_HEADS} heads across {len(head_sets[tag])} layers")

coco_index = load_coco_index(coco_root=DEFAULT_COCO_ROOT, split=DEFAULT_COCO_SPLIT)


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


demo_samples = find_multi_object_images(coco_index, N_DEMO_SAMPLES, seed=SEED + 1)   # same seed prefix as the 10-sample runs
print(f"Found {len(demo_samples)} multi-object images (target was {N_DEMO_SAMPLES}).")


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


results = {tag: [] for tag in head_sets}
for sample in tqdm(demo_samples, desc="Steering (3 head-sets/image)"):
    try:
        image = Image.open(sample["image_path"]).convert("RGB")
        baseline_inputs = prepare_inputs(processor, image, CAPTION_PROMPT, DEVICE)
        baseline_prompt_len = int(baseline_inputs["input_ids"].shape[1])
        baseline_sequences = run_generation(model=model, inputs=baseline_inputs, max_new_tokens=MAX_NEW_TOKENS)
        baseline_text = decode_generated_text(processor, baseline_sequences, baseline_prompt_len)

        x, y, w, h = sample["alternate"]["ann"]["bbox"]
        alt_bbox_xyxy = (x, y, x + w, y + h)
        alt_cat = sample["alternate"]["category"]
        ref_caption = f"a photo of a {alt_cat}."
        baseline_sim = semantic_similarity(baseline_text, ref_caption, device="cpu")

        for tag, heads_by_layer in head_sets.items():
            steered_text = caption_with_steering(image, CAPTION_PROMPT, alt_bbox_xyxy, heads_by_layer, n_heads)
            steered_sim = semantic_similarity(steered_text, ref_caption, device="cpu")
            matches = mentions_category(steered_text, alt_cat) or steered_sim >= SIM_THRESHOLD
            results[tag].append({
                "image_id": sample["image_id"], "alternate": alt_cat,
                "match": matches, "steered_sim": steered_sim, "baseline_sim": baseline_sim,
            })
    except Exception as exc:
        print(f"Skipping image_id={sample.get('image_id')}: {exc}")

print(f"\n{'head-set':>12s}  {'n':>4s}  {'match rate':>11s}  {'95% CI':>16s}  {'mean sim(steered,alt)':>22s}  {'mean sim(base,alt)':>19s}")
for tag, res in results.items():
    ci = bootstrap_ci([r["match"] for r in res])
    print(f"{tag:>12s}  {ci['n']:4d}  {ci['accuracy']:11.3f}  [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]  "
          f"{np.mean([r['steered_sim'] for r in res]):22.3f}  {np.mean([r['baseline_sim'] for r in res]):19.3f}")

print("\n=== Pairwise paired Wilcoxon on continuous steered_sim ===")
tags = list(results.keys())
for i, a in enumerate(tags):
    for b in tags[i + 1:]:
        n = min(len(results[a]), len(results[b]))
        sim_a = [r["steered_sim"] for r in results[a][:n]]
        sim_b = [r["steered_sim"] for r in results[b][:n]]
        w = stats.wilcoxon(sim_a, sim_b)
        higher = a if np.mean(sim_a) > np.mean(sim_b) else b
        print(f"  {a:>12s} vs {b:<12s}: statistic={w.statistic:.1f}  p={w.pvalue:.3e}  -> higher mean sim: {higher}")

dump_json(REPO_ROOT / "logs" / "steer_demo_500" / "summary.json", {
    tag: {"n": len(res), "match_rate": float(np.mean([r["match"] for r in res])),
          "mean_steered_sim": float(np.mean([r["steered_sim"] for r in res]))}
    for tag, res in results.items()
})
print("\nDONE")
