"""Gaze-score computation and head ranking.

The gaze score of head (l, h) is the mean raw post-softmax attention mass the
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
    raw scoring counts as strong gaze and value-weighting correctly discounts.

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

    Used to area-normalize gaze scores: a head that merely allocates
    attention proportional to region size looks "gaze-y" under raw scoring
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


def save_head_ranking(path: Path, ranking: Sequence[dict[str, float]]) -> Path:
    return dump_json(path, list(ranking))


def load_head_ranking(path: Path, top_k: int) -> list[tuple[int, int]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing gaze-head ranking at {path}. Run 01_discover_vis_heads.py first."
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
    """Seeded random control heads, optionally restricted to heads whose gaze
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
