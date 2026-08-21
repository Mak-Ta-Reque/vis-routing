"""Vir-score computation and head ranking.

The vir score of head (l, h) is the mean raw post-softmax attention mass the
final prompt token places on panel k's image tokens when the prompt asks
about panel k, averaged over panels and strips (the diagonal of the 6x6
queried-panel x attended-panel matrix).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from vis_head.common import dump_json, ordinal
from vis_head.modeling import extract_prefill_attentions, find_image_token_range


_NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}


def panel_query_prompt(panel_index: int, n_panels: int = 6) -> str:
    count = _NUMBER_WORDS.get(n_panels, str(n_panels))
    return (
        f"Look carefully at this {count}-panel comic strip. "
        f"What is happening in the {ordinal(panel_index)} panel from the left? "
        "Answer briefly."
    )


def collect_last_query_attentions(model: Any, inputs: Any) -> np.ndarray:
    """One prefill pass; return attention from the last prompt token to every
    key position, stacked as (n_layers, n_heads, seq_len)."""
    output = extract_prefill_attentions(model, inputs)
    layers = []
    for attention in output.attentions:
        layers.append(attention[0, :, -1, :].detach().float().cpu().numpy())
    return np.stack(layers, axis=0)


def aggregate_region_attention(
    attn_at_query: np.ndarray,
    inputs: Any,
    processor: Any,
    region_ids: np.ndarray,
    n_regions: int,
) -> np.ndarray:
    """Sum raw attention over each panel's image tokens (no normalization).

    Keeping the attention raw — rather than re-normalizing across panels —
    means a head only scores high if it (a) actually puts attention mass on
    image tokens and (b) concentrates that mass on the queried panel. A head
    that ignores images entirely could still look perfectly diagonal after
    normalization; raw scoring sends it to zero. Rows do NOT sum to 1.
    """
    img_start, img_end = find_image_token_range(inputs, processor)
    image_attention = attn_at_query[:, :, img_start:img_end]
    usable = min(image_attention.shape[-1], region_ids.shape[0])
    image_attention = image_attention[:, :, :usable].astype(np.float64)
    region_ids_usable = region_ids[:usable]

    n_regions = int(n_regions)
    region_onehot = np.zeros((usable, n_regions), dtype=np.float64)
    region_onehot[np.arange(usable), region_ids_usable] = 1.0
    return np.einsum("lht,tr->lhr", image_attention, region_onehot)


def per_sample_head_scores(region_attn: np.ndarray, target_region: int) -> dict[str, np.ndarray]:
    """From one sample's (L, H, n_regions) region-attention matrix (the direct
    output of `aggregate_region_attention`), compute the five per-head
    quantities used for VIR head scoring comparison:

    1. raw target mass       -- S(R*): plain target-region attention (the
       project's original score). Over-selects heads that just attend
       broadly to the whole image, since they accumulate target mass without
       discriminating anything.
    2. target share           -- S(R*) / sum_R S(R): the naive normalized
       fix. Unstable for heads with near-zero total visual attention (small
       denominator inflates the ratio).
    3. total visual attention -- sum_R S(R), needed to diagnose (1) vs (2)
       and to pool (2) correctly across samples (see `pooled_target_share`).
    4. excess mass            -- S(R*) - (1/G) * sum_R S(R): zero for a
       uniformly-attending head regardless of total visual attention, zero
       for a head with no visual attention regardless of apparent
       selectivity in (2), and large only when a head both attends to the
       image AND concentrates on the target. No division, so no instability.

    Returns per-head (L, H) arrays: {"raw_target_mass", "target_share",
    "total_visual_attention", "excess_mass"}. Accumulate "raw_target_mass"
    for score (1); average "target_share" across samples for score (2, mean
    of ratios); use `pooled_target_share` on accumulated sums for score (3,
    pooled ratio); accumulate "total_visual_attention" for score (4);
    accumulate "excess_mass" for score (5).
    """
    n_regions = region_attn.shape[-1]
    raw_target_mass = region_attn[:, :, target_region]
    total_visual_attention = region_attn.sum(axis=-1)
    safe_total = np.where(total_visual_attention > 0, total_visual_attention, 1.0)
    target_share = raw_target_mass / safe_total
    excess_mass = raw_target_mass - total_visual_attention / n_regions
    return {
        "raw_target_mass": raw_target_mass,
        "target_share": target_share,
        "total_visual_attention": total_visual_attention,
        "excess_mass": excess_mass,
    }


def pooled_target_share(sum_raw_target_mass: np.ndarray, sum_total_visual_attention: np.ndarray) -> np.ndarray:
    """Score (3): sum_n S(R*) / sum_n sum_R S(R), computed from accumulated
    sums across samples (not a mean of per-sample ratios like score (2)) --
    avoids letting a single low-attention sample's unstable ratio dominate
    the average the way score (2) can."""
    safe = np.where(sum_total_visual_attention > 0, sum_total_visual_attention, 1.0)
    return sum_raw_target_mass / safe


def collect_cot_trace_region_attention(
    model: Any,
    inputs: Any,
    target_positions: Sequence[int],
    img_start: int,
    img_end: int,
    n_layers: int,
    n_heads: int,
    max_new_tokens: int,
) -> np.ndarray:
    """CoT-aware discovery signal: area-normalized target-region attention
    mass, averaged over every decode step of a full generation — not just
    the single final-prompt-token query used by `collect_last_query_attentions`
    / `aggregate_region_attention`.

    For reasoning/CoT-tuned models (e.g. Qwen3-VL-*-Thinking), the standard
    final-query method measures attention *before* any of the actual
    reasoning happens, and was empirically found to select heads with no
    significant causal steering effect. This function instead records every
    head's attention back to the target region across the whole generated
    reasoning trace (via `vis_head.steering.register_attention_trackers`) and
    averages it, which found heads with a far stronger causal effect for a
    CoT model in practice (steered accuracy 0.240 -> 0.713 on the same
    held-out eval set, zero head overlap with the final-query top-15 --
    see `cot_vis_head_discovery_qwen3_thinking.ipynb`).

    Returns a (n_layers, n_heads) array, area-normalized so that 1.0 means
    "attends to the target region exactly proportional to its share of all
    image tokens" (chance level) -- controls for region-size differences the
    way raw un-normalized attention mass would not.
    """
    from vis_head.modeling import run_generation
    from vis_head.steering import register_attention_trackers, remove_handles

    target_idx = np.asarray(list(target_positions), dtype=np.int64)
    n_image_tokens = max(img_end - img_start, 1)
    expected_share = len(target_idx) / n_image_tokens

    records: list[tuple[int, np.ndarray]] = []
    handles = register_attention_trackers(model, list(range(n_layers)), records)
    try:
        run_generation(model=model, inputs=inputs, max_new_tokens=max_new_tokens)
    finally:
        remove_handles(handles)

    scores = np.zeros((n_layers, n_heads), dtype=np.float64)
    if not records or len(target_idx) == 0:
        return scores.astype(np.float32)

    per_layer_steps: dict[int, list[np.ndarray]] = {}
    for layer_idx, attn in records:
        # attn: (n_heads, kv_len); target_idx positions are always < img_end,
        # so always valid columns regardless of how far decoding has progressed.
        target_mass = attn[:, target_idx].sum(axis=1)
        per_layer_steps.setdefault(layer_idx, []).append(target_mass)

    for layer_idx, steps in per_layer_steps.items():
        mean_mass = np.mean(steps, axis=0)
        scores[layer_idx] = mean_mass / max(expected_share, 1e-8)
    return scores.astype(np.float32)


def collect_last_query_attentions_value_weighted(
    model: Any,
    inputs: Any,
    processor: Any,
) -> np.ndarray:
    """Like `collect_last_query_attentions`, but weights each attention weight
    by the L2 norm of the corresponding image token's value vector
    (Kobayashi et al. 2020): `attn[l, h, t] * ||v[l, h, t]||`.

    Raw attention weight measures how much a head *looks* at a token; the
    value norm measures how much signal it actually *writes* when it does.
    A head can attend heavily while emitting a near-zero-norm value, which
    raw scoring counts as strong vir and value-weighting correctly discounts.

    Only the image-token columns are weighted (that's the range the value-norm
    tracker captures, and the only range the visual-routing score aggregates over);
    positions outside [img_start, img_end) are returned unweighted and are
    ignored by `aggregate_region_attention` anyway. Returns
    (n_layers, n_heads, seq_len), same shape/contract as the raw version.
    """
    from vis_head.modeling import find_image_token_range
    from vis_head.steering import register_prefill_value_norm_trackers, remove_handles

    img_start, img_end = find_image_token_range(inputs, processor)

    n_layers = len(_lm_layers_for_scoring(model))
    value_norms: dict[int, np.ndarray] = {}
    handles = register_prefill_value_norm_trackers(
        model=model,
        layers=list(range(n_layers)),
        img_start=img_start,
        img_end=img_end,
        storage=value_norms,
    )
    try:
        attn_at_query = collect_last_query_attentions(model, inputs)
    finally:
        remove_handles(handles)

    weighted = attn_at_query.astype(np.float64).copy()
    for layer_idx in range(weighted.shape[0]):
        norms = value_norms.get(layer_idx)
        if norms is None:
            continue
        usable = min(norms.shape[-1], img_end - img_start)
        weighted[layer_idx, :, img_start:img_start + usable] *= norms[:, :usable]
    return weighted


def _lm_layers_for_scoring(model: Any):
    from vis_head.modeling import language_model_layers
    return language_model_layers(model)


def collect_attribution_scores(model: Any, inputs: Any) -> np.ndarray:
    """Gradient-based attribution alternative to raw attention weight
    (attribution patching, Nanda et al. 2023): for each head and each
    sequence position, `(head_output_at_that_position) . d(next-token
    log-prob)/d(head_output_at_that_position)` — a first-order estimate of
    how much that head's contribution *at that position* causally moves the
    model's own greedy next-token prediction, via a single backward pass.

    Unlike raw attention (one layer's immediate attention pattern) or
    value-norm weighting (write magnitude, blind to downstream use), this
    accounts for whether the head's contribution actually survives through
    later layers to influence the output — attribution patching is used in
    place of exhaustive per-head ablation specifically because it's been
    shown to approximate real causal-patching effects at a fraction of the
    cost. Returns (n_layers, n_heads, seq_len), signed (can be negative:
    pushes probability away from the greedy token), same shape/contract as
    `collect_last_query_attentions` so it drops into
    `aggregate_region_attention` unchanged.
    """
    from vis_head.modeling import language_model_layers

    layers = language_model_layers(model)
    n_layers = len(layers)

    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module, args, kwargs):
            x = args[0] if args else kwargs.get("input")
            x.retain_grad()
            captured[layer_idx] = x
        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(make_hook(i), with_kwargs=True))

    try:
        model.zero_grad(set_to_none=True)
        out = model(**inputs, use_cache=False)
        logits = out.logits[0, -1].float()
        next_token = int(torch.argmax(logits).item())
        loss = -torch.log_softmax(logits, dim=-1)[next_token]
        loss.backward()
    finally:
        for h in handles:
            h.remove()

    seq_len = int(inputs["input_ids"].shape[1])
    n_heads = int(getattr(getattr(model.config, "text_config", None) or model.config, "num_attention_heads"))
    head_dim = int(layers[0].self_attn.o_proj.in_features // n_heads)

    attribution = np.zeros((n_layers, n_heads, seq_len), dtype=np.float64)
    for i in range(n_layers):
        x = captured.get(i)
        if x is None or x.grad is None:
            continue
        val = x[0].detach().view(seq_len, n_heads, head_dim)
        grad = x.grad[0].detach().view(seq_len, n_heads, head_dim)
        contrib = (val * grad).sum(dim=-1)   # (seq_len, n_heads)
        attribution[i] = contrib.float().cpu().numpy().T   # (n_heads, seq_len)
    model.zero_grad(set_to_none=True)
    return attribution


def panel_token_fractions(region_ids: np.ndarray, n_regions: int) -> np.ndarray:
    """Fraction of image tokens belonging to each region (panel).

    Used to area-normalize vir scores: a head that merely allocates
    attention proportional to region size looks "vir-y" under raw scoring
    even with no real targeting behavior, and larger regions (e.g. a comic
    panel, ~1/N of the image) mechanically out-score smaller ones (e.g. a
    small COCO object). Dividing the raw score by this fraction turns it into
    an enrichment ratio — attention density relative to chance — that is
    comparable across regions/datasets of very different size.
    """
    counts = np.bincount(region_ids, minlength=n_regions)[:n_regions].astype(np.float64)
    total = max(1, int(region_ids.shape[0]))
    return counts / total


def rank_heads_by_score(scores: np.ndarray) -> list[dict[str, float]]:
    ranked = []
    n_layers, n_heads = scores.shape
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            ranked.append(
                {
                    "layer": int(layer_idx),
                    "head": int(head_idx),
                    "score": float(scores[layer_idx, head_idx]),
                }
            )
    ranked.sort(key=lambda row: row["score"], reverse=True)
    return ranked


def normalize_scores_global_max(scores: np.ndarray) -> np.ndarray:
    """Divide every score by the single global maximum. A strictly monotonic
    rescaling (one shared positive constant) -- provably cannot change which
    heads rank highest, so `rank_heads_by_score` picks identical heads whether
    given raw or global-max-normalized scores. Included for magnitude reporting
    (comparing selectivity across checkpoints on a common 0-1 scale), not
    because it changes head *selection*."""
    m = scores.max()
    return scores / m if m > 0 else scores.copy()


def normalize_scores_per_layer_max(scores: np.ndarray) -> np.ndarray:
    """Divide each layer's scores by that layer's own maximum. Unlike global-max
    normalization, this is NOT a single shared constant across all heads, so it
    CAN reorder the top-K ranking: a head that stands out within a naturally
    low-attention layer can outrank a head with higher absolute magnitude that
    sits in a naturally high-attention layer. Empirically found (see
    `raw_vs_perlayer_head_selection.ipynb`) to select heads with a much weaker
    causal steering effect than raw selection on Qwen3-VL-8B-Instruct."""
    per_layer_max = scores.max(axis=1, keepdims=True)
    per_layer_max = np.where(per_layer_max > 0, per_layer_max, 1.0)
    return scores / per_layer_max


def normalize_scores_per_layer_zscore(scores: np.ndarray) -> np.ndarray:
    """Standardize each layer's scores to zero mean, unit std (within that
    layer only). Like `normalize_scores_per_layer_max`, this can reorder the
    top-K ranking relative to raw scoring, but uses a layer's full score
    distribution (mean/std) rather than just its max as the reference point --
    a head is ranked by how many standard deviations it sits above its own
    layer's typical selectivity, not by its ratio to the single best head in
    that layer."""
    mean = scores.mean(axis=1, keepdims=True)
    std = scores.std(axis=1, keepdims=True)
    std = np.where(std > 0, std, 1.0)
    return (scores - mean) / std


def score_variance_stats(per_sample_scores: np.ndarray) -> dict[str, np.ndarray]:
    """Given per-head scores from many discovery samples (grids), shape
    (n_samples, n_layers, n_heads), summarize each head's mean, std, and
    coefficient of variation (std/mean) ACROSS grid inputs. All of the
    project's existing discovery loops only ever accumulate a running sum
    (i.e. keep the mean) and throw the per-sample values away, so this needs
    per-sample scores stacked into an array first -- see `snr_score` below for
    the selection-ready summary metric built from these stats.

    A head with a high mean but also high variance across grids is a weak
    candidate for a *general-purpose* vir/routing head: its apparent
    selectivity may be driven by a handful of easy images rather than a
    consistent attention-routing behavior that holds across arbitrary inputs.
    """
    mean = per_sample_scores.mean(axis=0)
    std = per_sample_scores.std(axis=0)
    cv = np.divide(std, mean, out=np.full_like(std, np.inf), where=mean > 0)
    return {"mean": mean, "std": std, "cv": cv}


def snr_score(per_sample_scores: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Signal-to-noise-style selection score: mean score across discovery
    grids divided by (std + eps) across those same grids. Unlike plain mean
    scoring (what `rank_heads_by_score` normally ranks), this rewards heads
    that are both strong AND *consistent* across different grid inputs, and
    penalizes heads whose high average is really just high variance --
    occasional spikes on particular images rather than reliable routing.
    Feed the result into `rank_heads_by_score` exactly like a raw score
    matrix to get an SNR-based top-K head selection."""
    stats = score_variance_stats(per_sample_scores)
    return stats["mean"] / (stats["std"] + eps)


def compute_visual_head_scores(
    region_attn_all: np.ndarray,
    target_cells: np.ndarray,
    prompt_ids: np.ndarray | None = None,
    eps: float = 1e-8,
) -> dict[str, np.ndarray]:
    """Composite Visual Head Score: for each (layer, head), combine four
    complementary measures of whether that head's attention to visual tokens
    is genuinely doing position-conditioned object routing, rather than some
    simpler confound (a head that always attends heavily to *any* cell, or
    always the same absolute location regardless of content).

    Inputs
    ------
    region_attn_all : (n_samples, n_layers, n_heads, n_cells) per-sample
        region-aggregated attention (e.g. many calls to `aggregate_region_attention`
        stacked along axis 0 -- one row per (image, prompt) sample).
    target_cells : (n_samples,) int array, the ground-truth cell index holding
        the referenced object in each sample.
    prompt_ids : optional (n_samples,) int/str array grouping samples by which
        discovery-prompt phrasing was used (for instruction selectivity below).

    Per-sample components
    ----------------------
    1. object attention        -- attention mass on the target cell.
    2. control-corrected attn  -- object attention minus the mean attention on
                                   all OTHER cells in the same sample (background
                                   subtraction: a head that just attends heavily
                                   everywhere gets no credit for "finding" anything).
    3. spatial variance         -- variance of control-corrected attention across
                                   samples (e.g. across the imagenet_circular_grid
                                   position series for a fixed object) -- how STABLE
                                   the signal is when only position changes.
    4. object localization      -- object attention as a fraction of TOTAL attention
                                   summed over all cells (concentration ratio).
    5. spatial correspondence   -- whether the cell with maximum attention actually
                                   IS the target cell (0/1 match rate).

    Aggregate components (returned, all shape (n_layers, n_heads))
    -----------------------------------------------------------------
    - spatial_sensitivity: mean(control-corrected attn) / sqrt(spatial variance + eps)
      -- an SNR-style combination of components 2+3: strong AND stable signal.
    - object_localization: mean of component 4 across samples.
    - spatial_correspondence: mean of component 5 across samples (a 0-1 match rate).
    - instruction_selectivity: (only if `prompt_ids` given) std, across prompt
      groups, of each group's mean control-corrected attention -- rewards heads
      whose attention to the SAME object shifts with how the request is phrased,
      i.e. task-framing-sensitive rather than purely content-driven.
    - visual_head_score: sum of the four components above, each z-scored across
      all (layer, head) pairs first so no single component dominates by scale.
    """
    n_samples, n_layers, n_heads, n_cells = region_attn_all.shape
    target_cells = np.asarray(target_cells)

    object_attn = region_attn_all[np.arange(n_samples), :, :, target_cells]   # (n_samples, n_layers, n_heads)
    total_attn = region_attn_all.sum(axis=3)
    other_sum = total_attn - object_attn
    other_mean = other_sum / max(n_cells - 1, 1)
    control_corrected = object_attn - other_mean

    localization = object_attn / (total_attn + eps)
    argmax_cell = region_attn_all.argmax(axis=3)                              # (n_samples, n_layers, n_heads)
    correspondence = (argmax_cell == target_cells[:, None, None]).astype(np.float64)

    control_corrected_mean = control_corrected.mean(axis=0)
    spatial_variance = control_corrected.var(axis=0)
    spatial_sensitivity = control_corrected_mean / np.sqrt(spatial_variance + eps)
    object_localization = localization.mean(axis=0)
    spatial_correspondence = correspondence.mean(axis=0)

    if prompt_ids is not None:
        prompt_ids = np.asarray(prompt_ids)
        group_means = [control_corrected[prompt_ids == g].mean(axis=0) for g in np.unique(prompt_ids)]
        instruction_selectivity = np.stack(group_means, axis=0).std(axis=0)
    else:
        instruction_selectivity = np.zeros((n_layers, n_heads))

    def _zscore(mat: np.ndarray) -> np.ndarray:
        mu, sigma = mat.mean(), mat.std()
        return (mat - mu) / sigma if sigma > 0 else np.zeros_like(mat)

    visual_head_score = (
        _zscore(spatial_sensitivity)
        + _zscore(object_localization)
        + _zscore(spatial_correspondence)
        + _zscore(instruction_selectivity)
    )

    return {
        "spatial_sensitivity": spatial_sensitivity,
        "object_localization": object_localization,
        "spatial_correspondence": spatial_correspondence,
        "instruction_selectivity": instruction_selectivity,
        "control_corrected_mean": control_corrected_mean,
        "spatial_variance": spatial_variance,
        "visual_head_score": visual_head_score,
    }


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg false-discovery-rate correction over a flat array of
    p-values (one per hypothesis, e.g. one per (layer, head) pair). Returns
    (reject_mask, qvalues), both same shape as `pvalues`."""
    pvalues = np.asarray(pvalues, dtype=np.float64)
    flat = pvalues.ravel()
    n = flat.size
    order = np.argsort(flat)
    ranked = flat[order]
    ranks = np.arange(1, n + 1)
    raw_q = ranked * n / ranks
    q_sorted = np.minimum.accumulate(raw_q[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    qvalues_flat = np.empty(n)
    qvalues_flat[order] = q_sorted
    qvalues = qvalues_flat.reshape(pvalues.shape)
    reject = qvalues <= alpha
    return reject, qvalues


def permutation_test_visual_head_scores(
    region_attn_all: np.ndarray,
    target_cells: np.ndarray,
    prompt_ids: np.ndarray | None = None,
    n_permutations: int = 200,
    rng: np.random.RandomState | None = None,
    alpha: float = 0.05,
) -> dict[str, np.ndarray]:
    """Runs `compute_visual_head_scores` on the true labels, then rebuilds the
    null distribution of `visual_head_score` per (layer, head) by repeatedly
    shuffling which cell is treated as the "target" (independently per sample,
    drawn uniformly over the other cells) and recomputing the spatial
    components under that scrambled labeling -- entirely on already-collected
    attention tensors, no extra model forward passes needed. A head whose
    observed score rarely exceeds its own null distribution is not reliably
    doing position-conditioned routing; it's easy to fool with a random label.
    `instruction_selectivity` does not depend on target-cell identity, so it is
    computed once (unpermuted) and added back into every null draw.

    Returns the observed component dict (from `compute_visual_head_scores`)
    plus `pvalues`, `qvalues` (Benjamini-Hochberg corrected), and
    `significant_mask` (qvalues <= alpha)."""
    rng = rng or np.random.RandomState()
    n_samples, n_layers, n_heads, n_cells = region_attn_all.shape
    observed = compute_visual_head_scores(region_attn_all, target_cells, prompt_ids)

    null_ge_count = np.zeros((n_layers, n_heads), dtype=np.int64)
    for _ in range(n_permutations):
        shuffled = np.array([
            rng.choice([c for c in range(n_cells) if c != t]) for t in target_cells
        ])
        null_scores = compute_visual_head_scores(region_attn_all, shuffled, prompt_ids)
        null_ge_count += (null_scores["visual_head_score"] >= observed["visual_head_score"])

    pvalues = (null_ge_count + 1) / (n_permutations + 1)
    _, qvalues = benjamini_hochberg(pvalues, alpha=alpha)
    significant_mask = qvalues <= alpha

    observed["pvalues"] = pvalues
    observed["qvalues"] = qvalues
    observed["significant_mask"] = significant_mask
    return observed


def save_head_ranking(path: Path, ranking: Sequence[dict[str, float]]) -> Path:
    return dump_json(path, list(ranking))


def load_head_ranking(path: Path, top_k: int) -> list[tuple[int, int]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing vir-head ranking at {path}. Run 01_discover_vis_heads.py first."
        )
    ranking = json.loads(path.read_text())
    out = []
    for row in ranking[: max(1, int(top_k))]:
        out.append((int(row["layer"]), int(row["head"])))
    if not out:
        raise RuntimeError(f"No heads found in {path}")
    return out


def sample_non_vis_heads(
    n_layers: int,
    n_heads: int,
    exclude: set[tuple[int, int]],
    n_select: int,
    seed: int,
    scores: np.ndarray | None = None,
    max_score: float | None = None,
) -> list[tuple[int, int]]:
    """Seeded random control heads, optionally restricted to heads whose vir
    score is below `max_score` (e.g. the bottom-5% percentile cutoff)."""
    candidates = []
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            if (layer_idx, head_idx) in exclude:
                continue
            if scores is not None and max_score is not None:
                if float(scores[layer_idx, head_idx]) > max_score:
                    continue
            candidates.append((layer_idx, head_idx))
    if not candidates:
        return []

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(candidates), generator=generator).tolist()
    return [candidates[idx] for idx in perm[: min(n_select, len(candidates))]]
