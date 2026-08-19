"""Build a COCO-based Vis-Head dataset (see vis_head/coco.py).

For each selected COCO object annotation, writes:
  - metadata.jsonl   : {sample_id, image_id, annotation_id, category_id,
                        category_name, image_path, instruction}
  - ground_truth.jsonl : {sample_id, bbox, segmentation, patch_grid,
                          patch_mask, patch_mask_flat}

The model input (image + "Find the <category>.") never includes bbox,
segmentation, coordinates, annotation id, or an object crop — only
`metadata.jsonl` is meant to be shown to the model; `ground_truth.jsonl` is
held out for scoring vis heads.

Usage:
    python 06_build_coco_vis_head_dataset.py \
        --coco-root /mnt/abka03/raw_data_download/mscoco2024 \
        --split val2014 \
        --max-samples 500 \
        --model-id Qwen/Qwen3-VL-8B-Instruct
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tqdm.auto import tqdm

from vis_head.common import DEFAULT_MODEL_ID
from vis_head.coco import (
    DEFAULT_COCO_OUTPUT_DIR,
    DEFAULT_COCO_ROOT,
    DEFAULT_COCO_SPLIT,
    SelectionConfig,
    ground_truth_record,
    load_coco_index,
    mask_to_patch_occupancy,
    sample_metadata_record,
    segmentation_to_mask,
    select_target_objects,
    write_jsonl,
)
from vis_head.regions import get_merged_grid_shape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-root", type=Path, default=DEFAULT_COCO_ROOT)
    parser.add_argument("--split", type=str, default=DEFAULT_COCO_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_COCO_OUTPUT_DIR)
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID,
                         help="Only the image processor is loaded, to compute the real "
                              "visual patch grid for each image (no GPU needed).")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--max-per-category", type=int, default=None)
    parser.add_argument("--categories", type=str, default=None,
                         help="Comma-separated category names to restrict to (default: all).")
    parser.add_argument("--allow-ambiguous", action="store_true",
                         help="Allow target categories that occur more than once in the image.")
    parser.add_argument("--min-area", type=float, default=400.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from transformers import AutoImageProcessor
    image_processor = AutoImageProcessor.from_pretrained(args.model_id)

    print(f"Loading COCO {args.split} instances from {args.coco_root} ...")
    index = load_coco_index(coco_root=args.coco_root, split=args.split)

    config = SelectionConfig(
        unique_category_only=not args.allow_ambiguous,
        min_area=args.min_area,
        category_names=args.categories.split(",") if args.categories else None,
        max_per_category=args.max_per_category,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    targets = select_target_objects(index, config)
    print(f"Selected {len(targets)} target objects.")

    metadata_records = []
    ground_truth_records = []
    for target in tqdm(targets, desc="Building samples"):
        image_path = Path(target.image_path)
        if not image_path.exists():
            continue

        from PIL import Image
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        processed = image_processor(images=image, return_tensors="pt")
        grid_thw = processed["image_grid_thw"]
        spatial_merge = getattr(image_processor, "merge_size", 2)
        grid_shape = get_merged_grid_shape(grid_thw, spatial_merge)

        mask = segmentation_to_mask(target.segmentation, target.image_height, target.image_width)
        patch_mask = mask_to_patch_occupancy(mask, grid_shape)

        metadata_records.append(sample_metadata_record(target))
        ground_truth_records.append(ground_truth_record(target, grid_shape, patch_mask))

    write_jsonl(args.output_dir / "metadata.jsonl", metadata_records)
    write_jsonl(args.output_dir / "ground_truth.jsonl", ground_truth_records)
    print(f"Wrote {len(metadata_records)} samples to {args.output_dir}")


if __name__ == "__main__":
    main()
