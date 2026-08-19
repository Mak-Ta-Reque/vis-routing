"""COCO-based Vis-Head dataset: object selection, masks, and patch occupancy.

Builds one dataset sample per selected COCO object annotation. The model
input is always the *full, unmodified* image plus a short instruction
("Find the <category>."); the COCO bbox/segmentation are kept only as hidden
ground truth, mapped down to the VLM's visual-patch grid (see
`mask_to_patch_occupancy`) so it can later be compared against attention
maps to identify vis heads.

Only `iscrowd == 0` annotations are used: those carry polygon segmentations
(one or more closed polygons) rather than RLE-encoded crowd regions, which
keeps mask rasterization dependency-free (no pycocotools/RLE decoding
needed) and matches the "single, precise object" framing of the task.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw

from vis_head.common import REPO_ROOT, ensure_dir, json_default

DEFAULT_COCO_ROOT = Path(
    os.environ.get("GAZE_COCO_ROOT", "/mnt/abka03/raw_data_download/mscoco2024")
)
DEFAULT_COCO_SPLIT = "val2014"
DEFAULT_COCO_OUTPUT_DIR = REPO_ROOT / "data" / "coco_vis_head"


def instruction_for_category(category_name: str) -> str:
    return f"Find the {category_name.strip()}."


# ---------------------------------------------------------------------------
# 1. Load COCO
# ---------------------------------------------------------------------------


@dataclass
class CocoIndex:
    images_root: Path
    images_by_id: dict[int, dict]
    categories_by_id: dict[int, str]
    annotations_by_image: dict[int, list[dict]]


def coco_ann_file(coco_root: Path, split: str) -> Path:
    return Path(coco_root) / "annotations" / f"instances_{split}.json"


def coco_images_root(coco_root: Path, split: str) -> Path:
    return Path(coco_root) / split


def load_coco_index(
    coco_root: Path = DEFAULT_COCO_ROOT,
    split: str = DEFAULT_COCO_SPLIT,
) -> CocoIndex:
    """Parse COCO instance annotations into id-indexed lookup tables."""
    ann_path = coco_ann_file(coco_root, split)
    data = json.loads(ann_path.read_text())

    images_by_id = {int(img["id"]): img for img in data["images"]}
    categories_by_id = {int(cat["id"]): cat["name"] for cat in data["categories"]}

    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in data["annotations"]:
        annotations_by_image[int(ann["image_id"])].append(ann)

    return CocoIndex(
        images_root=coco_images_root(coco_root, split),
        images_by_id=images_by_id,
        categories_by_id=categories_by_id,
        annotations_by_image=dict(annotations_by_image),
    )


# ---------------------------------------------------------------------------
# 2. Select target objects
# ---------------------------------------------------------------------------


@dataclass
class TargetObject:
    sample_id: int
    image_id: int
    annotation_id: int
    category_id: int
    category_name: str
    image_path: str
    image_width: int
    image_height: int
    instruction: str
    bbox: list[float]
    segmentation: Any


@dataclass
class SelectionConfig:
    """Configurable filtering for which COCO objects become dataset samples."""

    unique_category_only: bool = True   # target category must appear exactly once in the image
    require_single_polygon: bool = False  # reject multi-part (occluded) segmentations
    exclude_iscrowd: bool = True
    min_area: float = 400.0             # px^2; drop tiny/near-invisible objects
    min_bbox_side: float = 8.0          # px; drop degenerate boxes
    category_names: Optional[Sequence[str]] = None  # None = all categories
    max_per_category: Optional[int] = None
    max_samples: Optional[int] = None
    seed: int = 42


def select_target_objects(index: CocoIndex, config: SelectionConfig = SelectionConfig()) -> list[TargetObject]:
    """Pick one object per sample, preferring unambiguous single-instance
    references ("Find the dog." with exactly one dog in the image)."""
    allowed_categories = set(config.category_names) if config.category_names else None
    rng = np.random.RandomState(config.seed)

    candidates: list[dict] = []
    for image_id, anns in index.annotations_by_image.items():
        category_counts: dict[int, int] = defaultdict(int)
        for ann in anns:
            if config.exclude_iscrowd and ann.get("iscrowd", 0):
                continue
            category_counts[int(ann["category_id"])] += 1

        for ann in anns:
            if config.exclude_iscrowd and ann.get("iscrowd", 0):
                continue
            category_id = int(ann["category_id"])
            category_name = index.categories_by_id.get(category_id)
            if category_name is None:
                continue
            if allowed_categories is not None and category_name not in allowed_categories:
                continue
            if config.unique_category_only and category_counts[category_id] != 1:
                continue
            if float(ann.get("area", 0.0)) < config.min_area:
                continue
            bbox = ann.get("bbox")
            if not bbox or bbox[2] < config.min_bbox_side or bbox[3] < config.min_bbox_side:
                continue
            segmentation = ann.get("segmentation")
            if not isinstance(segmentation, list) or len(segmentation) == 0:
                continue  # RLE / crowd segmentation, already excluded above but be defensive
            if config.require_single_polygon and len(segmentation) != 1:
                continue
            candidates.append({"image_id": image_id, "ann": ann, "category_name": category_name})

    # Shuffle deterministically so max_per_category / max_samples don't just
    # take the first COCO image ids.
    order = rng.permutation(len(candidates))
    candidates = [candidates[i] for i in order]

    per_category_count: dict[str, int] = defaultdict(int)
    targets: list[TargetObject] = []
    for item in candidates:
        category_name = item["category_name"]
        if config.max_per_category is not None and per_category_count[category_name] >= config.max_per_category:
            continue
        ann = item["ann"]
        image_id = item["image_id"]
        image_info = index.images_by_id[image_id]

        targets.append(
            TargetObject(
                sample_id=len(targets),
                image_id=image_id,
                annotation_id=int(ann["id"]),
                category_id=int(ann["category_id"]),
                category_name=category_name,
                image_path=str(index.images_root / image_info["file_name"]),
                image_width=int(image_info["width"]),
                image_height=int(image_info["height"]),
                instruction=instruction_for_category(category_name),
                bbox=[float(v) for v in ann["bbox"]],
                segmentation=ann["segmentation"],
            )
        )
        per_category_count[category_name] += 1
        if config.max_samples is not None and len(targets) >= config.max_samples:
            break

    return targets


# ---------------------------------------------------------------------------
# 4. Ground-truth object mask
# ---------------------------------------------------------------------------


def segmentation_to_mask(segmentation: list[list[float]], height: int, width: int) -> np.ndarray:
    """Rasterize COCO polygon segmentation(s) into a binary (H, W) mask."""
    mask_image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_image)
    for polygon in segmentation:
        points = list(zip(polygon[0::2], polygon[1::2]))
        if len(points) >= 3:
            draw.polygon(points, outline=1, fill=1)
    return (np.array(mask_image, dtype=np.uint8) > 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# 5/6. Map mask to the VLM's visual-patch grid
# ---------------------------------------------------------------------------


def mask_to_patch_occupancy(mask: np.ndarray, grid_shape: tuple[int, int, int]) -> np.ndarray:
    """Downsample a pixel-space binary mask to per-patch occupancy fractions.

    `grid_shape` is (t, merged_h, merged_w) as returned by
    `vis_head.regions.get_merged_grid_shape` for the *same* image after VLM
    preprocessing. Box-filter resizing area-averages the mask into each grid
    cell, which is exactly `object_pixels_in_patch / total_pixels_in_patch`
    because the VLM's resize preserves a uniform scale factor across the
    image (no cropping/padding is applied by the Qwen-VL processors this
    repo supports), so grid cells map onto proportional regions of the
    original image.
    """
    t, merged_h, merged_w = grid_shape
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))
    patch = np.asarray(
        mask_image.resize((merged_w, merged_h), Image.BOX), dtype=np.float64
    ) / 255.0
    patch = np.clip(patch, 0.0, 1.0)
    return np.tile(patch, (t, 1, 1))  # (t, merged_h, merged_w)


def flatten_patch_mask(patch_mask: np.ndarray) -> list[float]:
    """Row-major flatten matching the LM's image-token order (see regions.py)."""
    return patch_mask.reshape(-1).tolist()


# ---------------------------------------------------------------------------
# 7. Dataset output
# ---------------------------------------------------------------------------


def sample_metadata_record(target: TargetObject) -> dict:
    return {
        "sample_id": target.sample_id,
        "image_id": target.image_id,
        "annotation_id": target.annotation_id,
        "category_id": target.category_id,
        "category_name": target.category_name,
        "image_path": target.image_path,
        "instruction": target.instruction,
    }


def ground_truth_record(
    target: TargetObject,
    grid_shape: tuple[int, int, int],
    patch_mask: np.ndarray,
) -> dict:
    return {
        "sample_id": target.sample_id,
        "bbox": target.bbox,
        "segmentation": target.segmentation,
        "patch_grid": [int(grid_shape[1]), int(grid_shape[2])],
        "patch_mask": patch_mask[0].tolist(),
        "patch_mask_flat": flatten_patch_mask(patch_mask),
    }


def write_jsonl(path: Path, records: Sequence[dict]) -> Path:
    out = Path(path)
    ensure_dir(out.parent)
    with out.open("w") as f:
        for record in records:
            f.write(json.dumps(record, default=json_default) + "\n")
    return out


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


@dataclass
class CocoVisHeadDataset:
    """A generated COCO-gaze dataset loaded back from disk."""

    metadata: list[dict] = field(default_factory=list)
    ground_truth_by_id: dict[int, dict] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tuple[dict, dict]:
        meta = self.metadata[idx]
        return meta, self.ground_truth_by_id[meta["sample_id"]]


def target_weight_fraction(patch_mask_flat: Sequence[float]) -> float:
    """Mean per-token occupancy weight — the COCO analogue of
    `vis_head.gaze.panel_token_fractions`: how much of the image the target
    object effectively covers, used to area-normalize gaze scores so a small
    COCO object isn't mechanically penalized relative to a large comic panel.
    """
    weights = np.asarray(patch_mask_flat, dtype=np.float64)
    return float(weights.mean()) if weights.size else 0.0


def vis_head_score_from_patch_mask(
    attn_at_query: np.ndarray,
    img_start: int,
    img_end: int,
    patch_mask_flat: Sequence[float],
) -> np.ndarray:
    """Fraction of each head's image-token attention mass that lands on the
    target object, weighted by per-patch occupancy (`patch_mask_flat`, values
    in [0, 1]). Returns an (n_layers, n_heads) score matrix.

    Mirrors `vis_head.gaze.aggregate_region_attention`, but uses continuous
    patch-occupancy weights from a COCO segmentation instead of a one-hot
    panel assignment.
    """
    image_attention = attn_at_query[:, :, img_start:img_end].astype(np.float64)
    weights = np.asarray(patch_mask_flat, dtype=np.float64)
    usable = min(image_attention.shape[-1], weights.shape[0])
    return np.einsum("lht,t->lh", image_attention[:, :, :usable], weights[:usable])


def load_coco_vis_head_dataset(output_dir: Path = DEFAULT_COCO_OUTPUT_DIR) -> CocoVisHeadDataset:
    output_dir = Path(output_dir)
    metadata = read_jsonl(output_dir / "metadata.jsonl")
    ground_truth = read_jsonl(output_dir / "ground_truth.jsonl")
    ground_truth_by_id = {row["sample_id"]: row for row in ground_truth}
    return CocoVisHeadDataset(metadata=metadata, ground_truth_by_id=ground_truth_by_id)
