"""Direct test: if the target object identity changes (always "golden retriever"
vs. always "persian cat"), does the same set of vis heads still fire?

Each sample tiles the fixed target class into one cell (a different photo of
that class each time) plus 3 random, distinct distractor classes in the other
cells — everything about the setup is identical between the "dog" and "cat"
conditions except which specific class occupies the target cell.
"""
import sys
from pathlib import Path

REPO_ROOT = Path("/mnt/abka03/Projects/vis-head")
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image
from scipy import stats
from tqdm.auto import tqdm

from vis_head.common import DEFAULT_MODEL_ID, DEFAULT_SEED, dump_json
from vis_head.vir import aggregate_region_attention, collect_last_query_attentions, rank_heads_by_score
from vis_head.imagenet_grid import DEFAULT_IMAGENET_ROOT, PROMPT_TEMPLATES, list_val_class_dirs, load_class_names
from vis_head.modeling import load_model_and_processor, model_dims, prepare_inputs
from vis_head.regions import assign_grid_cells_to_tokens

MODEL_ID = DEFAULT_MODEL_ID
DEVICE = "cuda:0"
SEED = DEFAULT_SEED
ROWS, COLS = 2, 2
N_CELLS = ROWS * COLS
CELL_SIZE = 256
N_PER_CLASS = 40

TARGET_CLASSES = {"dog (golden retriever)": "n02099601", "cat (persian cat)": "n02123394"}

model, processor = load_model_and_processor(model_id=MODEL_ID, device=DEVICE)
n_layers, n_heads, spatial_merge = model_dims(model)
print(f"{n_layers} layers x {n_heads} heads")

imagenet_class_dirs = {d.name: d for d in list_val_class_dirs(DEFAULT_IMAGENET_ROOT)}
imagenet_class_names = load_class_names(DEFAULT_IMAGENET_ROOT)
all_wnids = list(imagenet_class_dirs.keys())


def _open_rgb_square(path: Path, size: int) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
        w, h = image.size
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
        image = image.crop((left, top, left + side, top + side))
        return image.resize((size, size), Image.LANCZOS)


def build_fixed_target_grid(target_wnid: str, rng: np.random.RandomState):
    """Target class always occupies a randomly chosen cell; the other 3 cells
    get distinct random distractor classes (never the target class itself)."""
    target_cell = int(rng.randint(N_CELLS))
    distractor_wnids = []
    while len(distractor_wnids) < N_CELLS - 1:
        candidate = all_wnids[int(rng.randint(len(all_wnids)))]
        if candidate != target_wnid and candidate not in distractor_wnids:
            distractor_wnids.append(candidate)

    cell_wnids = []
    di = 0
    for i in range(N_CELLS):
        if i == target_cell:
            cell_wnids.append(target_wnid)
        else:
            cell_wnids.append(distractor_wnids[di])
            di += 1

    cell_images = []
    for wnid in cell_wnids:
        class_dir = imagenet_class_dirs[wnid]
        images = sorted(p for p in class_dir.iterdir() if p.suffix.upper() in (".JPEG", ".JPG", ".PNG"))
        image_path = images[int(rng.randint(len(images)))]
        cell_images.append(_open_rgb_square(image_path, CELL_SIZE))

    grid_image = Image.new("RGB", (COLS * CELL_SIZE, ROWS * CELL_SIZE), (255, 255, 255))
    for idx, img in enumerate(cell_images):
        r, c = divmod(idx, COLS)
        grid_image.paste(img, (c * CELL_SIZE, r * CELL_SIZE))

    target_name = imagenet_class_names[target_wnid]
    return grid_image, target_cell, target_name


def discover_for_class(target_wnid: str, label: str, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    raw_sum = np.zeros((n_layers, n_heads), dtype=np.float64)
    valid = 0
    for _ in tqdm(range(N_PER_CLASS), desc=f"[{label}]", leave=False):
        grid_image, target_cell, target_name = build_fixed_target_grid(target_wnid, rng)
        prompt = PROMPT_TEMPLATES["find"].format(name=target_name)
        try:
            inputs = prepare_inputs(processor, grid_image, prompt, DEVICE)
            region_ids, _ = assign_grid_cells_to_tokens(
                image_grid_thw=inputs["image_grid_thw"], rows=ROWS, cols=COLS, spatial_merge=spatial_merge)
            attn_at_query = collect_last_query_attentions(model, inputs)
            region_attention = aggregate_region_attention(
                attn_at_query=attn_at_query, inputs=inputs, processor=processor, region_ids=region_ids, n_regions=N_CELLS)
            raw_sum += region_attention[:, :, target_cell]
            valid += 1
        except Exception as exc:
            print(f"Skipping: {exc}")
    print(f"  [{label:>24s}] valid={valid}/{N_PER_CLASS}  mean score={raw_sum.mean() / max(valid, 1):.5f}")
    return (raw_sum / max(valid, 1)).astype(np.float32)


scores = {}
for i, (label, wnid) in enumerate(TARGET_CLASSES.items()):
    scores[label] = discover_for_class(wnid, label, seed=SEED + 500 + i)

names = list(scores.keys())
a, b = names
ranked_a, ranked_b = rank_heads_by_score(scores[a]), rank_heads_by_score(scores[b])

print(f"\n=== {a}  vs  {b} ===")
for K in (10, 50, 100):
    top_a = set((r["layer"], r["head"]) for r in ranked_a[:K])
    top_b = set((r["layer"], r["head"]) for r in ranked_b[:K])
    overlap = len(top_a & top_b) / K
    print(f"  top-{K:<4d} overlap: {overlap:.3f}  ({len(top_a & top_b)}/{K})")

rho = stats.spearmanr(scores[a].reshape(-1), scores[b].reshape(-1))
print(f"  Spearman rho: {rho.correlation:.3f}  (p={rho.pvalue:.3e})")
print(f"  Mean score: {a}={scores[a].mean():.5f}  {b}={scores[b].mean():.5f}")

dump_json(REPO_ROOT / "logs" / "dog_vs_cat_check" / "summary.json", {
    "classes": TARGET_CLASSES,
    "mean_scores": {k: float(v.mean()) for k, v in scores.items()},
    "spearman_rho": float(rho.correlation),
})
print("\nDONE")
