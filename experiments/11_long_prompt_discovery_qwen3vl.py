"""Vir-head discovery on COCO (400 samples) and ImageNet-grid (400 samples)
using a comics-style long prompt instead of the minimal "Find the {name}."
used everywhere else in this project:

  long prompt = "Look carefully at this picture. Find the {name}. Answer briefly."

(mirrors comics' own "Look carefully at this six-panel comic strip. What is
happening in the Nth panel from the left? Answer briefly.")

Same model (Qwen3-VL-8B-Instruct) as the cached comics/coco/imagenet 3-way
comparison, so results are directly comparable against those existing
short-prompt rankings (logs/vis_head_discovery_compare_datasets_{coco,imagenet}/).
"""
import sys
from pathlib import Path

REPO_ROOT = Path("/mnt/abka03/Projects/vis-head")
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image
from scipy import stats
from tqdm.auto import tqdm

from vis_head.common import DEFAULT_MODEL_ID, DEFAULT_SEED, dump_json, make_output_paths
from vis_head.coco import DEFAULT_COCO_ROOT, DEFAULT_COCO_SPLIT, SelectionConfig, vis_head_score_from_patch_mask, load_coco_index, mask_to_patch_occupancy, segmentation_to_mask, select_target_objects, target_weight_fraction
from vis_head.vir import aggregate_region_attention, collect_last_query_attentions, panel_token_fractions, rank_heads_by_score
from vis_head.imagenet_grid import DEFAULT_IMAGENET_ROOT, list_val_class_dirs, load_class_names, sample_grid
from vis_head.modeling import find_image_token_range, load_model_and_processor, model_dims, prepare_inputs
from vis_head.regions import assign_grid_cells_to_tokens, get_merged_grid_shape

MODEL_ID = DEFAULT_MODEL_ID
DEVICE = "cuda:0"
SEED = DEFAULT_SEED
N_SAMPLES = 400
EPS = 1e-8
ROWS, COLS = 2, 2
N_CELLS = ROWS * COLS
CELL_SIZE = 256


def long_prompt(name: str) -> str:
    return f"Look carefully at this picture. Find the {name}. Answer briefly."


model, processor = load_model_and_processor(model_id=MODEL_ID, device=DEVICE)
n_layers, n_heads, spatial_merge = model_dims(model)
print(f"{n_layers} layers x {n_heads} heads")


# ---------------------------- COCO, 400 samples, long prompt ----------------------------
print("\n=== COCO discovery (long prompt, N=400) ===")
coco_index = load_coco_index(coco_root=DEFAULT_COCO_ROOT, split=DEFAULT_COCO_SPLIT)
coco_config = SelectionConfig(unique_category_only=True, min_area=400.0, max_samples=N_SAMPLES, seed=SEED)
coco_targets = select_target_objects(coco_index, coco_config)
print(f"COCO samples selected: {len(coco_targets)}")

coco_raw_sum = np.zeros((n_layers, n_heads), dtype=np.float64)
coco_norm_sum = np.zeros((n_layers, n_heads), dtype=np.float64)
coco_valid = 0
for target in tqdm(coco_targets, desc="COCO [long prompt]"):
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
        raw = vis_head_score_from_patch_mask(attn_at_query, img_start, img_end, patch_mask_flat)
        fraction = target_weight_fraction(patch_mask_flat)
        coco_raw_sum += raw
        coco_norm_sum += raw / max(fraction, EPS)
        coco_valid += 1
    except Exception as exc:
        print(f"Skipping {target.image_path}: {exc}")

coco_scores_long = (coco_raw_sum / max(coco_valid, 1)).astype(np.float32)
coco_scores_norm_long = (coco_norm_sum / max(coco_valid, 1)).astype(np.float32)
print(f"valid: {coco_valid}/{len(coco_targets)}  mean raw={coco_scores_long.mean():.5f}  mean norm={coco_scores_norm_long.mean():.4f}")

outputs = make_output_paths("vis_head_discovery_compare_datasets_coco_longprompt")
np.save(outputs.logs_dir / "vis_head_scores.npy", coco_scores_long)
np.save(outputs.logs_dir / "vis_head_scores_normalized.npy", coco_scores_norm_long)
coco_ranked_long = rank_heads_by_score(coco_scores_long)
dump_json(outputs.logs_dir / "vis_head_ranking.json", coco_ranked_long)


# ---------------------------- ImageNet-grid, 400 samples, long prompt ----------------------------
print("\n=== ImageNet-grid discovery (long prompt, N=400) ===")
imagenet_class_dirs = list_val_class_dirs(DEFAULT_IMAGENET_ROOT)
imagenet_class_names = load_class_names(DEFAULT_IMAGENET_ROOT)
rng = np.random.RandomState(SEED)

imagenet_raw_sum = np.zeros((n_layers, n_heads), dtype=np.float64)
imagenet_norm_sum = np.zeros((n_layers, n_heads), dtype=np.float64)
imagenet_valid = 0
for _ in tqdm(range(N_SAMPLES), desc="ImageNet [long prompt]"):
    grid = sample_grid(rows=ROWS, cols=COLS, cell_size=CELL_SIZE, rng=rng,
                        class_dirs=imagenet_class_dirs, class_names=imagenet_class_names)
    target_cell = int(rng.randint(N_CELLS))
    prompt = long_prompt(grid.cell_names[target_cell])
    try:
        inputs = prepare_inputs(processor, grid.grid, prompt, DEVICE)
        region_ids, _ = assign_grid_cells_to_tokens(
            image_grid_thw=inputs["image_grid_thw"], rows=ROWS, cols=COLS, spatial_merge=spatial_merge)
        attn_at_query = collect_last_query_attentions(model, inputs)
        region_attention = aggregate_region_attention(
            attn_at_query=attn_at_query, inputs=inputs, processor=processor, region_ids=region_ids, n_regions=N_CELLS)
        raw = region_attention[:, :, target_cell]
        fraction = panel_token_fractions(region_ids, N_CELLS)[target_cell]
        imagenet_raw_sum += raw
        imagenet_norm_sum += raw / max(fraction, EPS)
        imagenet_valid += 1
    except Exception as exc:
        print(f"Skipping grid: {exc}")

imagenet_scores_long = (imagenet_raw_sum / max(imagenet_valid, 1)).astype(np.float32)
imagenet_scores_norm_long = (imagenet_norm_sum / max(imagenet_valid, 1)).astype(np.float32)
print(f"valid: {imagenet_valid}/{N_SAMPLES}  mean raw={imagenet_scores_long.mean():.5f}  mean norm={imagenet_scores_norm_long.mean():.4f}")

outputs2 = make_output_paths("vis_head_discovery_compare_datasets_imagenet_longprompt")
np.save(outputs2.logs_dir / "vis_head_scores.npy", imagenet_scores_long)
np.save(outputs2.logs_dir / "vis_head_scores_normalized.npy", imagenet_scores_norm_long)
imagenet_ranked_long = rank_heads_by_score(imagenet_scores_long)
dump_json(outputs2.logs_dir / "vis_head_ranking.json", imagenet_ranked_long)


# ---------------------------- compare against the existing short-prompt rankings ----------------------------
print("\n=== Comparison: long prompt (N=400) vs. short prompt (cached, N~150-200) ===")


def compare(tag, scores_long, norm_long):
    cached_raw = np.load(REPO_ROOT / "logs" / f"vis_head_discovery_compare_datasets_{tag}" / "vis_head_scores.npy")
    cached_norm = np.load(REPO_ROOT / "logs" / f"vis_head_discovery_compare_datasets_{tag}" / "vis_head_scores_normalized.npy")
    ranked_long = rank_heads_by_score(scores_long)
    ranked_short = rank_heads_by_score(cached_raw)
    for K in (10, 50, 100):
        top_long = set((r["layer"], r["head"]) for r in ranked_long[:K])
        top_short = set((r["layer"], r["head"]) for r in ranked_short[:K])
        overlap = len(top_long & top_short) / K
        print(f"  [{tag}] top-{K:<4d} overlap (long vs short prompt): {overlap:.3f}")
    rho = stats.spearmanr(scores_long.reshape(-1), cached_raw.reshape(-1))
    rho_norm = stats.spearmanr(norm_long.reshape(-1), cached_norm.reshape(-1))
    print(f"  [{tag}] Spearman rho (raw):        {rho.correlation:.3f}")
    print(f"  [{tag}] Spearman rho (normalized):  {rho_norm.correlation:.3f}")
    print(f"  [{tag}] mean raw score   -- short: {cached_raw.mean():.5f}   long: {scores_long.mean():.5f}   ratio: {scores_long.mean()/max(cached_raw.mean(),EPS):.2f}x")
    print(f"  [{tag}] mean norm score  -- short: {cached_norm.mean():.4f}   long: {norm_long.mean():.4f}   ratio: {norm_long.mean()/max(cached_norm.mean(),EPS):.2f}x")


compare("coco", coco_scores_long, coco_scores_norm_long)
compare("imagenet", imagenet_scores_long, imagenet_scores_norm_long)

# also: does long-prompt COCO/ImageNet close the gap with comics' raw score magnitude?
comics_raw = np.load(REPO_ROOT / "logs" / "vis_head_discovery_compare_datasets_comics" / "vis_head_scores.npy")
print(f"\nComics mean raw score (reference, short prompt): {comics_raw.mean():.5f}")
print(f"COCO      mean raw score -- short: {np.load(REPO_ROOT/'logs'/'vis_head_discovery_compare_datasets_coco'/'vis_head_scores.npy').mean():.5f}   long: {coco_scores_long.mean():.5f}")
print(f"ImageNet  mean raw score -- short: {np.load(REPO_ROOT/'logs'/'vis_head_discovery_compare_datasets_imagenet'/'vis_head_scores.npy').mean():.5f}   long: {imagenet_scores_long.mean():.5f}")

print("\nDONE")
