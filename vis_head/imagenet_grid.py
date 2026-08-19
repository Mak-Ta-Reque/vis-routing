"""ImageNet-grid gaze dataset: a clean, controlled alternative to comic strips
and COCO for finding and stress-testing vis heads.

Each sample tiles `rows x cols` single-object ImageNet validation images into
one grid image (a MxN quilt of unrelated objects, one distinct class per
cell — no occlusion, no clutter, no ambiguity about where an object "is"
within its cell, unlike a COCO photo). The grid plays the same role comic
panels played in `vis_head/data.py`: a set of known, non-overlapping image
regions with ground-truth content, but built from real photographs instead of
drawn panels, and with an *independent* choice of which object sits in which
cell on every sample (unlike comics, whose panel content is fixed per strip).

This independence is what makes the grid useful for generalization testing:
holding the prompt template fixed and varying which object lands in the
target cell tests whether vis heads are *object-general*; holding the
object fixed and varying the instruction's wording (verb) tests whether gaze
heads are *prompt/action-general* rather than tied to a specific phrasing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw

from vis_head.common import REPO_ROOT

DEFAULT_IMAGENET_ROOT = Path(
    os.environ.get("GAZE_IMAGENET_ROOT", "/mnt/abka03/raw_data_download/imagenet")
)
DEFAULT_IMAGENET_SPLIT = "val"
DEFAULT_ROWS = 2
DEFAULT_COLS = 2
DEFAULT_CELL_SIZE = 256
DEFAULT_GAP = 6

# Verb/phrasing templates for the prompt-generalization test (Part 5): same
# object reference, different instruction wording. "object_prompt" (the
# default, `"Find the {name}."`) matches the phrasing used for the COCO gaze
# dataset (`vis_head/coco.py`) for cross-dataset comparability.
PROMPT_TEMPLATES: dict[str, str] = {
    "find": "Find the {name}.",
    "locate": "Locate the {name}.",
    "where_is": "Where is the {name}?",
    "point_to": "Point to the {name}.",
    "identify": "Identify the {name}.",
    # No instruction/verb at all — just the referring expression itself, to test
    # whether vis heads need a task framing ("find"/"locate"/...) or fire on the
    # bare reference alone.
    "bare_name": "{name}",
}


def ordinal_prompt(cell_index_1based: int, n_cells: int) -> str:
    """Position-only phrasing (no object name) — the ImageNet-grid analogue
    of comics' `panel_query_prompt`, for testing position- vs. object-driven
    reference."""
    return f"What is in cell {cell_index_1based} of {n_cells}, counting left to right, top to bottom?"


_NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
                 9: "nine", 10: "ten", 12: "twelve", 16: "sixteen"}


def comics_style_ordinal_prompt(cell_index_1based: int, n_cells: int) -> str:
    """Word-for-word mirror of `vis_head.gaze.panel_query_prompt`'s wording
    and structure ("Look carefully at this {count}-panel comic strip. What is
    happening in the {ordinal} panel from the left? Answer briefly."), for a
    controlled comparison: same verbose framing, same "Answer briefly."
    close, same *reference type* (purely ordinal, no object named) — the only
    thing that differs from the comics prompt is the domain noun
    ("panel"/"comic strip" -> "cell"/"picture grid"). Used to test whether
    comics' steering advantage survives once the reference-type confound
    (ordinal vs. named-object) is removed.
    """
    from vis_head.gaze import ordinal as _ordinal

    count_word = _NUMBER_WORDS.get(n_cells, str(n_cells))
    return (
        f"Look carefully at this {count_word}-cell picture grid. "
        f"What is happening in the {_ordinal(cell_index_1based)} cell, "
        "reading left to right, top to bottom? Answer briefly."
    )


def bare_row_col_prompt(cell_index_1based: int, cols: int) -> str:
    """Bare grid-coordinate reference, no instruction/verb, no object name —
    e.g. "(1, 1)" for the top-left cell — the minimal possible location prompt."""
    idx0 = cell_index_1based - 1
    row, col = divmod(idx0, cols)
    return f"({row + 1}, {col + 1})"


def bare_cell_prompt(cell_index_1based: int) -> str:
    """Bare cell-number reference, no instruction/verb, no object name —
    e.g. "4" for the 4th cell (row-major)."""
    return str(cell_index_1based)


_ROW_WORDS_BY_COUNT = {1: [""], 2: ["top", "bottom"], 3: ["top", "middle", "bottom"]}
_COL_WORDS_BY_COUNT = {1: [""], 2: ["left", "right"], 3: ["left", "center", "right"]}


def mcq_prompt(cell_index_1based: int, rows: int, cols: int, options: Sequence[str],
                option_letters: Sequence[str] = ("A", "B", "C", "D")) -> str:
    """Canonical MCQ causal-intervention prompt: names the target cell's
    location in words (e.g. "top-left") before listing the options.

    Chosen as the project default after `mcq_prompt_format_search_v2_located.ipynb`
    found this ("verbose_word_position") to be the best-performing MCQ format across
    LLaVA-1.5, LLaVA-1.6, and Gemma-3-4B-it (mean accuracy 0.531 vs. 0.24-0.29 for
    row/col-coordinate or bounding-box location phrasing, and vs. near-chance for the
    earlier, location-free "Which of the following is shown in this image?" prompt,
    which was ambiguous whenever the image contained more than one object). That
    search only tested 2x2 grids; word-position phrasing ("top-left" etc.) is only
    defined up to 3x3, so grids larger than 3x3 fall back to a row/col coordinate
    reference (validated as the second-best location phrasing in the same search).
    """
    option_lines = "  ".join(f"{l}) {n}" for l, n in zip(option_letters, options))
    if rows <= 3 and cols <= 3:
        loc = position_words_prompt(cell_index_1based, rows, cols)
        return (
            f"Which of the following is shown in the {loc} part of this image? "
            f"{option_lines}. Answer with only the letter."
        )
    loc = bare_row_col_prompt(cell_index_1based, cols)
    return (
        f"Which of the following is shown at grid position {loc} (row, column) "
        f"in this image? {option_lines}. Answer with only the letter."
    )


def position_words_prompt(cell_index_1based: int, rows: int, cols: int) -> str:
    """Word-based spatial reference with NO digit tokens at all — e.g.
    "top-left" for a 2x2 grid's cell 1 — a control for whether the
    positional/named-object head split is actually about digit tokens rather
    than about spatial-vs-semantic reference type. Only defined for
    rows, cols <= 3 (uses top/middle/bottom x left/center/right)."""
    if rows > 3 or cols > 3:
        raise ValueError("position_words_prompt only supports grids up to 3x3.")
    idx0 = cell_index_1based - 1
    row, col = divmod(idx0, cols)
    row_word = _ROW_WORDS_BY_COUNT[rows][row]
    col_word = _COL_WORDS_BY_COUNT[cols][col]
    return "-".join(w for w in (row_word, col_word) if w)


def _clean_name(names: tuple[str, ...]) -> str:
    """First WordNet lemma, lowercased, underscores->spaces."""
    return names[0].replace("_", " ").strip().lower()


def load_class_names(imagenet_root: Path = DEFAULT_IMAGENET_ROOT) -> dict[str, str]:
    """wnid -> human-readable class name, from the standard torchvision
    ImageNet `meta.bin` (a pickled (wnid_to_classes, val_wnids) tuple)."""
    import torch

    wnid_to_classes, _ = torch.load(Path(imagenet_root) / "meta.bin", weights_only=False)
    return {wnid: _clean_name(names) for wnid, names in wnid_to_classes.items()}


def list_val_class_dirs(imagenet_root: Path = DEFAULT_IMAGENET_ROOT, split: str = DEFAULT_IMAGENET_SPLIT) -> list[Path]:
    split_root = Path(imagenet_root) / split
    if not split_root.exists():
        return []
    return sorted(p for p in split_root.iterdir() if p.is_dir())


@dataclass
class ImageNetGrid:
    name: str
    grid: Image.Image
    cell_images: list[Image.Image]
    cell_wnids: list[str]
    cell_names: list[str]
    rows: int
    cols: int
    cell_size: int
    gap: int
    cell_bboxes: list[tuple[int, int, int, int]]   # (x0, y0, x1, y1), row-major


def _open_rgb_square(path: Path, size: int) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
        # center-crop to square, then resize — avoids distorting object shape.
        w, h = image.size
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
        image = image.crop((left, top, left + side, top + side))
        return image.resize((size, size), Image.LANCZOS)


def _cell_bboxes(rows: int, cols: int, cell_size: int, gap: int) -> list[tuple[int, int, int, int]]:
    boxes = []
    for r in range(rows):
        for c in range(cols):
            x0 = c * (cell_size + gap)
            y0 = r * (cell_size + gap)
            boxes.append((x0, y0, x0 + cell_size, y0 + cell_size))
    return boxes


def _assemble_grid(cell_images: Sequence[Image.Image], rows: int, cols: int, cell_size: int, gap: int) -> Image.Image:
    width = cols * cell_size + (cols - 1) * gap
    height = rows * cell_size + (rows - 1) * gap
    grid = Image.new("RGB", (width, height), (255, 255, 255))
    for idx, image in enumerate(cell_images):
        r, c = divmod(idx, cols)
        x0 = c * (cell_size + gap)
        y0 = r * (cell_size + gap)
        grid.paste(image, (x0, y0))
    return grid


def sample_grid(
    imagenet_root: Path = DEFAULT_IMAGENET_ROOT,
    rows: int = DEFAULT_ROWS,
    cols: int = DEFAULT_COLS,
    cell_size: int = DEFAULT_CELL_SIZE,
    gap: int = DEFAULT_GAP,
    rng: Optional[np.random.RandomState] = None,
    class_dirs: Optional[Sequence[Path]] = None,
    class_names: Optional[dict[str, str]] = None,
) -> ImageNetGrid:
    """Sample `rows*cols` distinct ImageNet classes (one image each) and tile
    them into a grid. Distinct classes per cell keep `"Find the {name}."`
    unambiguous, mirroring COCO's unique-category selection constraint."""
    rng = rng or np.random.RandomState()
    class_dirs = list(class_dirs) if class_dirs is not None else list_val_class_dirs(imagenet_root)
    class_names = class_names if class_names is not None else load_class_names(imagenet_root)
    n_cells = rows * cols
    if len(class_dirs) < n_cells:
        raise ValueError(f"Need >= {n_cells} classes, found {len(class_dirs)} under {imagenet_root}.")

    chosen = [class_dirs[i] for i in rng.choice(len(class_dirs), size=n_cells, replace=False)]
    cell_images, cell_wnids, cell_names = [], [], []
    for class_dir in chosen:
        images = sorted(p for p in class_dir.iterdir() if p.suffix.upper() in (".JPEG", ".JPG", ".PNG"))
        image_path = images[int(rng.randint(len(images)))]
        cell_images.append(_open_rgb_square(image_path, cell_size))
        cell_wnids.append(class_dir.name)
        cell_names.append(class_names.get(class_dir.name, class_dir.name))

    grid_image = _assemble_grid(cell_images, rows, cols, cell_size, gap)
    return ImageNetGrid(
        name="_".join(chosen[i].name for i in range(min(3, n_cells))) + (f"_+{n_cells - 3}" if n_cells > 3 else ""),
        grid=grid_image,
        cell_images=cell_images,
        cell_wnids=cell_wnids,
        cell_names=cell_names,
        rows=rows,
        cols=cols,
        cell_size=cell_size,
        gap=gap,
        cell_bboxes=_cell_bboxes(rows, cols, cell_size, gap),
    )


def draw_cell_labels(grid: ImageNetGrid) -> Image.Image:
    """Grid image annotated with cell indices — for visual sanity-checking,
    never shown to the model."""
    annotated = grid.grid.copy()
    draw = ImageDraw.Draw(annotated)
    for idx, (x0, y0, x1, y1) in enumerate(grid.cell_bboxes):
        draw.rectangle([x0, y0, x0 + 60, y0 + 18], fill="black")
        draw.text((x0 + 4, y0 + 2), f"#{idx + 1}", fill="white")
    return annotated
