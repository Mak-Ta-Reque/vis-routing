"""ImageNet-grid vir-head discovery using the comics-mirrored ordinal prompt
(no object naming, same verbose framing, same reference type as comics'
panel_query_prompt) at 2x2 and 4x4 grid sizes, then the same 500-image COCO
steering evaluation used for the comics/coco_lp/imagenet_lp comparison, so
the reference-type confound (ordinal vs. named-object) is removed this time.
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

from vis_head.common import DEFAULT_SEED, dump_json, make_output_paths
from vis_head.coco import DEFAULT_COCO_ROOT, DEFAULT_COCO_SPLIT, load_coco_index
from vis_head.vir import aggregate_region_attention, collect_last_query_attentions, rank_heads_by_score
from vis_head.imagenet_grid import DEFAULT_IMAGENET_ROOT, comics_style_ordinal_prompt, list_val_class_dirs, load_class_names, sample_grid
from vis_head.judge import bootstrap_ci, semantic_similarity
from vis_head.modeling import decode_generated_text, find_image_token_range, load_model_and_processor, model_dims, prepare_inputs, run_generation
from vis_head.regions import assign_grid_cells_to_tokens, bbox_to_token_positions, get_merged_grid_shape
from vis_head.steering import group_heads_by_layer, intervention_positions, make_static_attention_mask_hook, register_mask_hooks, remove_handles

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
DEVICE = "cuda:0"
SEED = DEFAULT_SEED
N_DISCOVERY_SAMPLES = 400
TOP_K_HEADS = 30
N_DEMO_SAMPLES = 500
CAPTION_PROMPT = "Describe what is in this image. Answer in one short sentence."
MAX_NEW_TOKENS = 30
SIM_THRESHOLD = 0.5
CELL_SIZE = 256
GRID_SIZES = {"ordinal_2x2": (2, 2), "ordinal_4x4": (4, 4)}

model, processor = load_model_and_processor(model_id=MODEL_ID, device=DEVICE)
n_layers, n_heads, spatial_merge = model_dims(model)
print(f"{n_layers} layers x {n_heads} heads")

imagenet_class_dirs = list_val_class_dirs(DEFAULT_IMAGENET_ROOT)
imagenet_class_names = load_class_names(DEFAULT_IMAGENET_ROOT)

head_sets = {}

# ---------------------------- comics (cached) ----------------------------
comics_ranking = json.loads((REPO_ROOT / "logs" / "vis_head_discovery_qwen2vl_instruct" / "vis_head_ranking.json").read_text())
head_sets["comics"] = group_heads_by_layer([(r["layer"], r["head"]) for r in comics_ranking[:TOP_K_HEADS]])
print(f"[comics] top-{TOP_K_HEADS} heads across {len(head_sets['comics'])} layers (cached).")

# ---------------------------- ordinal ImageNet-grid discovery, 2x2 and 4x4 ----------------------------
for tag, (rows, cols) in GRID_SIZES.items():
    n_cells = rows * cols
    print(f"\n=== ImageNet-grid discovery [{tag}] (comics-style ordinal prompt, N={N_DISCOVERY_SAMPLES}) ===")
    rng = np.random.RandomState(SEED)
    raw_sum = np.zeros((n_layers, n_heads), dtype=np.float64)
    valid = 0
    for _ in tqdm(range(N_DISCOVERY_SAMPLES), desc=f"Vir discovery [{tag}]"):
        grid = sample_grid(rows=rows, cols=cols, cell_size=CELL_SIZE, rng=rng,
                            class_dirs=imagenet_class_dirs, class_names=imagenet_class_names)
        target_cell = int(rng.randint(n_cells))
        prompt = comics_style_ordinal_prompt(target_cell + 1, n_cells)
        try:
            inputs = prepare_inputs(processor, grid.grid, prompt, DEVICE)
            region_ids, _ = assign_grid_cells_to_tokens(
                image_grid_thw=inputs["image_grid_thw"], rows=rows, cols=cols, spatial_merge=spatial_merge)
            attn_at_query = collect_last_query_attentions(model, inputs)
            region_attention = aggregate_region_attention(
                attn_at_query=attn_at_query, inputs=inputs, processor=processor, region_ids=region_ids, n_regions=n_cells)
            raw_sum += region_attention[:, :, target_cell]
            valid += 1
        except Exception as exc:
            print(f"Skipping grid: {exc}")

    vis_head_scores = (raw_sum / max(valid, 1)).astype(np.float32)
    vir_ranked = rank_heads_by_score(vis_head_scores)
    print(f"valid: {valid}/{N_DISCOVERY_SAMPLES}  mean raw score: {vis_head_scores.mean():.5f}")
    print(f"Top-10 heads: {[(r['layer'], r['head']) for r in vir_ranked[:10]]}")

    outputs = make_output_paths(f"vis_head_discovery_qwen2vl_instruct_imagenet_{tag}")
    np.save(outputs.logs_dir / "vis_head_scores.npy", vis_head_scores)
    dump_json(outputs.logs_dir / "vis_head_ranking.json", vir_ranked)
    head_sets[tag] = group_heads_by_layer([(r["layer"], r["head"]) for r in vir_ranked[:TOP_K_HEADS]])


# ---------------------------- 500-image COCO steering eval, same images as before ----------------------------
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
            "alternate": {"category": alternate_cat, "ann": categories[alternate_cat]},
        })
        if len(picks) >= n_needed:
            break
    return picks


demo_samples = find_multi_object_images(coco_index, N_DEMO_SAMPLES, seed=SEED + 1)   # same seed -> same 500 images
print(f"\nFound {len(demo_samples)} multi-object COCO images (same set as the previous 500-image run).")


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
        x, y, w, h = sample["alternate"]["ann"]["bbox"]
        alt_bbox_xyxy = (x, y, x + w, y + h)
        alt_cat = sample["alternate"]["category"]
        ref_caption = f"a photo of a {alt_cat}."

        for tag, heads_by_layer in head_sets.items():
            steered_text = caption_with_steering(image, CAPTION_PROMPT, alt_bbox_xyxy, heads_by_layer, n_heads)
            steered_sim = semantic_similarity(steered_text, ref_caption, device="cpu")
            matches = mentions_category(steered_text, alt_cat) or steered_sim >= SIM_THRESHOLD
            results[tag].append({"image_id": sample["image_id"], "alternate": alt_cat, "match": matches, "steered_sim": steered_sim})
    except Exception as exc:
        print(f"Skipping image_id={sample.get('image_id')}: {exc}")

print(f"\n{'head-set':>14s}  {'n':>4s}  {'match rate':>11s}  {'95% CI':>16s}  {'mean sim(steered,alt)':>22s}")
for tag, res in results.items():
    ci = bootstrap_ci([r["match"] for r in res])
    print(f"{tag:>14s}  {ci['n']:4d}  {ci['accuracy']:11.3f}  [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]  "
          f"{np.mean([r['steered_sim'] for r in res]):22.3f}")

print("\n=== Pairwise paired Wilcoxon on continuous steered_sim ===")
tags = list(results.keys())
for i, a in enumerate(tags):
    for b in tags[i + 1:]:
        n = min(len(results[a]), len(results[b]))
        sim_a = [r["steered_sim"] for r in results[a][:n]]
        sim_b = [r["steered_sim"] for r in results[b][:n]]
        w = stats.wilcoxon(sim_a, sim_b)
        higher = a if np.mean(sim_a) > np.mean(sim_b) else b
        print(f"  {a:>14s} vs {b:<14s}: statistic={w.statistic:.1f}  p={w.pvalue:.3e}  -> higher mean sim: {higher}")

dump_json(REPO_ROOT / "logs" / "steer_demo_ordinal_grid" / "summary.json", {
    tag: {"n": len(res), "match_rate": float(np.mean([r["match"] for r in res])),
          "mean_steered_sim": float(np.mean([r["steered_sim"] for r in res]))}
    for tag, res in results.items()
})
print("\nDONE")
