"""
Length-tier thresholds, composite scoring, and filtering helpers for phrase detection.

Lower ``composite_score`` is better (distance + coverage gap + short-phrase prior).
"""

from __future__ import annotations

from typing import Protocol


class _SpanLike(Protocol):
    phrase: str
    start_s: float
    end_s: float
    composite_score: float | None
    normalized_distance: float


# Length-tier max normalized edit distance (relative to reference phoneme count).
_SHORT_MAX_NORM = 0.25  # ≤5 phonemes
_MEDIUM_MAX_NORM = 0.45  # 6–12
_LONG_MAX_NORM = 0.65  # >12
_SHORT_LEN_CUTOFF = 5
_MEDIUM_LEN_CUTOFF = 12

# Composite: score = alpha * norm + beta * (1 - coverage) + short_phrase_penalty
DEFAULT_SCORE_ALPHA = 0.6
DEFAULT_SCORE_BETA = 0.4
SHORT_PHONEME_PENALTY = 0.2  # target_len < 4
SHORT_LEN_FOR_PENALTY = 4

# Minimum audio duration (seconds) by source / length
MIN_DURATION_CALLSIGN_S = 0.38
MIN_DURATION_SHORT_FACILITY_S = 0.28  # ≤5 ref phonemes
MIN_DURATION_DEFAULT_FACILITY_S = 0.18
MIN_DURATION_SINGLE_WORD_COMMON_S = 0.22  # one-word common hits


def length_tier_max_normalized_distance(target_phoneme_len: int) -> float:
    """Stricter caps for short references (e.g. fix names like BAYST)."""
    n = max(1, target_phoneme_len)
    if n <= _SHORT_LEN_CUTOFF:
        return _SHORT_MAX_NORM
    if n <= _MEDIUM_LEN_CUTOFF:
        return _MEDIUM_MAX_NORM
    return _LONG_MAX_NORM


def effective_max_norm_for_sliding(
    target_phoneme_len: int,
    base_max: float,
    *,
    apply_length_tier: bool = True,
) -> float:
    """Combine CLI/base cap with length tier (callsign often skips tier; uses ``base_max`` only)."""
    if not apply_length_tier:
        return base_max
    return min(base_max, length_tier_max_normalized_distance(target_phoneme_len))


def coverage_ratio(matched_phoneme_count: int, target_phoneme_len: int) -> float:
    if target_phoneme_len <= 0:
        return 1.0
    return min(1.0, matched_phoneme_count / target_phoneme_len)


def composite_score(
    normalized_distance: float,
    coverage: float,
    target_phoneme_len: int,
    *,
    alpha: float = DEFAULT_SCORE_ALPHA,
    beta: float = DEFAULT_SCORE_BETA,
) -> float:
    pen = SHORT_PHONEME_PENALTY if target_phoneme_len < SHORT_LEN_FOR_PENALTY else 0.0
    return alpha * normalized_distance + beta * (1.0 - coverage) + pen


def min_duration_seconds(phraseology_source: str, target_phoneme_len: int) -> float:
    """Minimum span length to accept a detection (reduces tiny-window false positives)."""
    if phraseology_source == "callsign":
        return MIN_DURATION_CALLSIGN_S
    if phraseology_source == "facility":
        if target_phoneme_len <= _SHORT_LEN_CUTOFF:
            return MIN_DURATION_SHORT_FACILITY_S
        return MIN_DURATION_DEFAULT_FACILITY_S
    if phraseology_source == "common" and target_phoneme_len <= 3:
        return MIN_DURATION_SINGLE_WORD_COMMON_S
    return 0.0


def passes_min_duration(start_s: float, end_s: float, phraseology_source: str, target_phoneme_len: int) -> bool:
    d = max(0.0, end_s - start_s)
    return d >= min_duration_seconds(phraseology_source, target_phoneme_len)


def nms_same_phrase_iou(
    group: list[_SpanLike],
    *,
    iou_threshold: float = 0.5,
) -> list[_SpanLike]:
    """Non-maximum suppression: keep best composite_score, drop IoU-overlapping duplicates."""
    if len(group) <= 1:
        return group
    ordered = sorted(
        group,
        key=lambda s: (
            s.composite_score if s.composite_score is not None else s.normalized_distance,
            s.normalized_distance,
            s.start_s,
        ),
    )
    kept: list[_SpanLike] = []
    for d in ordered:
        if any(_iou(d, k) >= iou_threshold for k in kept):
            continue
        kept.append(d)
    return kept


def cooldown_filter_same_phrase(
    group: list[_SpanLike],
    *,
    cooldown_s: float = 0.75,
) -> list[_SpanLike]:
    """
    After sorting by score (best first), drop any later detection whose start falls
    inside the cooldown window after an already-kept span's end (same phrase).
    """
    if len(group) <= 1:
        return group
    ordered = sorted(
        group,
        key=lambda s: (
            s.composite_score if s.composite_score is not None else s.normalized_distance,
            s.normalized_distance,
            s.start_s,
        ),
    )
    kept: list[_SpanLike] = []
    for d in ordered:
        if any(d.start_s < k.end_s + cooldown_s for k in kept):
            continue
        kept.append(d)
    return kept


def dedupe_same_phrase_nms_and_cooldown(
    detections: list[_SpanLike],
    *,
    iou_threshold: float = 0.5,
    cooldown_s: float = 0.75,
) -> list[_SpanLike]:
    """Per phrase text: cooldown (best score first) then IoU NMS."""
    from collections import defaultdict

    by_phrase: dict[str, list[_SpanLike]] = defaultdict(list)
    for d in detections:
        by_phrase[d.phrase].append(d)
    out: list[_SpanLike] = []
    for group in by_phrase.values():
        if len(group) == 1:
            out.append(group[0])
            continue
        g = cooldown_filter_same_phrase(group, cooldown_s=cooldown_s)
        g = nms_same_phrase_iou(g, iou_threshold=iou_threshold)
        out.extend(g)
    return out


def _iou(a: _SpanLike, b: _SpanLike) -> float:
    s = max(a.start_s, b.start_s)
    e = min(a.end_s, b.end_s)
    inter = max(0.0, e - s)
    if inter <= 0:
        return 0.0
    union = max(a.end_s, b.end_s) - min(a.start_s, b.start_s)
    if union <= 0:
        return 0.0
    return inter / union
