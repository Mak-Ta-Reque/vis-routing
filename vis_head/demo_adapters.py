"""Model-family adapters for the interactive steering demo (demo/backend).

Two families are supported (see the raw_vs_perlayer / multistage notebooks
for the architecture investigation behind each):

- "qwen": Qwen2-VL / Qwen2.5-VL / Qwen3-VL -- native dynamic-resolution
  vision tokens, patch grid geometry given directly by `image_grid_thw` +
  `spatial_merge_size`.
- "gemma4": Gemma-4 -- aspect-ratio-adaptive vision pooling. The processor
  does not return the post-pool 2D grid shape directly, only the pre-pool
  `image_position_ids` and the final `num_soft_tokens_per_image`; the
  post-pool grid is re-derived here from the same kernel-index math used by
  `Gemma4VisionPooler._avg_pool_by_positions` in `transformers`.

Both adapters expose the same contract used by the demo backend:

    load_model(model_id) -> (model, processor)
    prepare_inputs(family, processor, image, prompt, device) -> inputs
    image_token_range(family, inputs) -> (img_start, img_end)
    bbox_to_token_positions(family, inputs, img_start, img_end, bbox_frac)
        -> (target_positions, other_image_positions)

where `bbox_frac = (x0, y0, x1, y1)` is a bounding box in NORMALIZED
[0, 1] image coordinates (as drawn by the frontend on the original,
un-resized image) and the returned position lists are LM sequence indices
(absolute, i.e. already offset by img_start) ready for
`vis_head.steering.intervention_positions` / `make_static_attention_mask_hook`.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

FAMILY_IMAGE_TOKEN_ID = {
    "qwen": None,  # resolved dynamically via vis_head.modeling.find_image_token_range
    "gemma4": 258880,
    "gemma3n": 262145,
}
GEMMA3N_GRID_SIDE = 16   # 256 soft image tokens = 16x16, fixed regardless of input image size


def gemma_family(model_id: str) -> str:
    ml = model_id.lower()
    if "gemma-4" in ml or "gemma4" in ml:
        return "gemma4"
    if "gemma-3n" in ml or "gemma3n" in ml:
        return "gemma3n"
    if "qwen" in ml:
        return "qwen"
    raise ValueError(f"Unsupported model_id for demo: {model_id!r}")


def load_model(model_id: str, device: str = "cuda:0"):
    family = gemma_family(model_id)
    if family == "qwen":
        from vis_head.modeling import load_model_and_processor
        return load_model_and_processor(model_id=model_id, device=device)

    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device)
    model.eval()
    return model, processor


def model_dims(family: str, model) -> tuple[int, int]:
    if family == "qwen":
        from vis_head.modeling import model_dims as _qwen_dims
        n_layers, n_heads, _ = _qwen_dims(model)
        return n_layers, n_heads
    n_layers = len(model.model.language_model.layers)
    n_heads = model.config.text_config.num_attention_heads
    return n_layers, n_heads


def prepare_inputs(family: str, processor, image, prompt: str, device: str):
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    if family == "qwen":
        from vis_head.modeling import prepare_inputs as _qwen_prepare
        return _qwen_prepare(processor, image, prompt, device)
    return processor.apply_chat_template(
        [messages], tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(device)


def image_token_range(family: str, inputs, processor=None) -> tuple[int, int]:
    if family == "qwen":
        from vis_head.modeling import find_image_token_range
        if processor is None:
            raise ValueError("Qwen family requires a processor to resolve its image_token_id.")
        return find_image_token_range(inputs, processor)
    ids = inputs["input_ids"][0].tolist()
    token_id = FAMILY_IMAGE_TOKEN_ID[family]
    positions = [i for i, t in enumerate(ids) if t == token_id]
    if not positions:
        raise ValueError(f"No image tokens found for family={family!r}.")
    return positions[0], positions[-1] + 1


# --------------------------------------------------------------------------
# Qwen: patch grid geometry direct from image_grid_thw
# --------------------------------------------------------------------------

def qwen_bbox_to_token_positions(
    image_grid_thw, spatial_merge: int, img_start: int, img_end: int, bbox_frac: tuple[float, float, float, float]
) -> tuple[list[int], list[int]]:
    from vis_head.regions import get_merged_grid_shape
    t, merged_h, merged_w = get_merged_grid_shape(image_grid_thw, spatial_merge)
    x0, y0, x1, y1 = bbox_frac
    row0, row1 = int(np.floor(y0 * merged_h)), int(np.ceil(y1 * merged_h))
    col0, col1 = int(np.floor(x0 * merged_w)), int(np.ceil(x1 * merged_w))
    row0, row1 = max(0, row0), min(merged_h, max(row1, row0 + 1))
    col0, col1 = max(0, col0), min(merged_w, max(col1, col0 + 1))

    rows_idx, cols_idx = np.meshgrid(np.arange(merged_h), np.arange(merged_w), indexing="ij")
    in_box = (rows_idx >= row0) & (rows_idx < row1) & (cols_idx >= col0) & (cols_idx < col1)
    in_box_flat = np.tile(in_box.reshape(-1), t)

    n_tokens = min(in_box_flat.shape[0], img_end - img_start)
    target = [img_start + i for i in range(n_tokens) if in_box_flat[i]]
    other = [img_start + i for i in range(n_tokens) if not in_box_flat[i]]
    return target, other


# --------------------------------------------------------------------------
# Gemma-4: re-derive the post-pool 2D grid from pixel_position_ids
# --------------------------------------------------------------------------

def gemma4_output_grid(pixel_position_ids: torch.Tensor, output_length: int) -> tuple[int, int]:
    """Reproduce Gemma4VisionPooler._avg_pool_by_positions's kernel-index math
    to recover the post-pool (height, width) grid shape, which the processor
    does not expose directly. `pixel_position_ids`: (1, n_patches, 2) pre-pool
    (x, y) patch coordinates, as returned in the processor's `image_position_ids`
    output. `output_length`: the actual number of soft image tokens for this
    image (== number of image_token_id occurrences in input_ids)."""
    positions = pixel_position_ids[0]  # (n_patches, 2)
    input_seq_len = positions.shape[0]
    k = int(round((input_seq_len / output_length) ** 0.5))
    if k < 1:
        k = 1
    clamped = positions.clamp(min=0)
    max_x = int(clamped[:, 0].max().item()) + 1
    output_width = max(1, max_x // k)
    output_height = max(1, output_length // output_width)
    return output_height, output_width


def gemma4_bbox_to_token_positions(
    pixel_position_ids: torch.Tensor,
    output_length: int,
    img_start: int,
    img_end: int,
    bbox_frac: tuple[float, float, float, float],
) -> tuple[list[int], list[int]]:
    out_h, out_w = gemma4_output_grid(pixel_position_ids, output_length)
    x0, y0, x1, y1 = bbox_frac
    row0, row1 = int(np.floor(y0 * out_h)), int(np.ceil(y1 * out_h))
    col0, col1 = int(np.floor(x0 * out_w)), int(np.ceil(x1 * out_w))
    row0, row1 = max(0, row0), min(out_h, max(row1, row0 + 1))
    col0, col1 = max(0, col0), min(out_w, max(col1, col0 + 1))

    rows_idx, cols_idx = np.meshgrid(np.arange(out_h), np.arange(out_w), indexing="ij")
    in_box = (rows_idx >= row0) & (rows_idx < row1) & (cols_idx >= col0) & (cols_idx < col1)
    in_box_flat = in_box.reshape(-1)

    n_tokens = min(in_box_flat.shape[0], img_end - img_start, output_length)
    target = [img_start + i for i in range(n_tokens) if in_box_flat[i]]
    other = [img_start + i for i in range(n_tokens) if not in_box_flat[i]]
    return target, other


# --------------------------------------------------------------------------
# Gemma-3n: fixed 16x16 grid (256 soft tokens) regardless of input image size
# --------------------------------------------------------------------------

def gemma3n_bbox_to_token_positions(
    img_start: int, img_end: int, bbox_frac: tuple[float, float, float, float]
) -> tuple[list[int], list[int]]:
    side = GEMMA3N_GRID_SIDE
    x0, y0, x1, y1 = bbox_frac
    row0, row1 = int(np.floor(y0 * side)), int(np.ceil(y1 * side))
    col0, col1 = int(np.floor(x0 * side)), int(np.ceil(x1 * side))
    row0, row1 = max(0, row0), min(side, max(row1, row0 + 1))
    col0, col1 = max(0, col0), min(side, max(col1, col0 + 1))

    rows_idx, cols_idx = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    in_box = (rows_idx >= row0) & (rows_idx < row1) & (cols_idx >= col0) & (cols_idx < col1)
    in_box_flat = in_box.reshape(-1)

    n_tokens = min(in_box_flat.shape[0], img_end - img_start)
    target = [img_start + i for i in range(n_tokens) if in_box_flat[i]]
    other = [img_start + i for i in range(n_tokens) if not in_box_flat[i]]
    return target, other


def bbox_to_token_positions(
    family: str, inputs, model, img_start: int, img_end: int, bbox_frac: tuple[float, float, float, float]
) -> tuple[list[int], list[int]]:
    if family == "qwen":
        from vis_head.modeling import model_dims as _qwen_dims
        _, _, spatial_merge = _qwen_dims(model)
        return qwen_bbox_to_token_positions(inputs["image_grid_thw"], spatial_merge, img_start, img_end, bbox_frac)
    if family == "gemma4":
        output_length = img_end - img_start
        return gemma4_bbox_to_token_positions(
            inputs["image_position_ids"], output_length, img_start, img_end, bbox_frac
        )
    if family == "gemma3n":
        return gemma3n_bbox_to_token_positions(img_start, img_end, bbox_frac)
    raise ValueError(f"Unsupported family: {family!r}")
