from __future__ import annotations

from dataclasses import dataclass

from app.audio.detection_quality import (
    composite_score as build_composite_score,
    coverage_ratio,
    effective_max_norm_for_sliding,
)


@dataclass
class KeyPhraseSegment:
    phrase: str
    start_s: float
    end_s: float
    distance: int
    normalized_distance: float
    matched_phonemes: list[str]
    #: ``common``, ``facility``, ``explicit``, or ``callsign`` (CLI ``--callsigns``).
    phraseology_source: str = "common"
    #: Length of the reference phoneme sequence for this phrase (for ranking / diagnostics).
    target_phoneme_len: int = 0
    #: ``len(matched_phonemes) / target_phoneme_len`` (capped at 1).
    coverage: float | None = None
    #: Lower is better: distance + coverage gap + optional short-phrase penalty.
    composite_score: float | None = None


def _edit_distance(a: list[str], b: list[str]) -> int:
    rows = len(a) + 1
    cols = len(b) + 1
    dp = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution/match
            )
    return dp[-1][-1]


def _effective_max_len_delta(target_len: int, base_delta: int) -> int:
    """
    Allow shorter windows in the transcript for longer expected phrases (fast / compressed speech).

    Keeps ``base_delta`` for short phrases to limit brute-force cost on huge facility vocabularies.
    """
    if target_len < 8:
        return base_delta
    return max(base_delta, min(12, target_len // 2))


def find_best_phrase_match(
    phrase: str,
    target_phonemes: list[str],
    segments: list[dict[str, float | str]],
    max_len_delta: int = 2,
    max_normalized_distance: float = 0.45,
    phraseology_source: str = "common",
    *,
    apply_length_tier: bool = True,
) -> KeyPhraseSegment | None:
    if not target_phonemes or not segments:
        return None

    phonemes = [str(segment["phoneme"]) for segment in segments]
    target_len = len(target_phonemes)
    effective_max = effective_max_norm_for_sliding(
        target_len, max_normalized_distance, apply_length_tier=apply_length_tier
    )
    span = _effective_max_len_delta(target_len, max_len_delta)
    min_window = max(1, target_len - span)
    max_window = target_len + span

    best: KeyPhraseSegment | None = None
    best_score = float("inf")
    for start_idx in range(len(phonemes)):
        for window_len in range(min_window, max_window + 1):
            end_idx = start_idx + window_len
            if end_idx > len(phonemes):
                continue

            candidate = phonemes[start_idx:end_idx]
            dist = _edit_distance(candidate, target_phonemes)
            norm_dist = dist / max(target_len, 1)
            if norm_dist > effective_max:
                continue

            cov = coverage_ratio(len(candidate), target_len)
            score = build_composite_score(norm_dist, cov, target_len)
            start_s = float(segments[start_idx]["start_s"])
            end_s = float(segments[end_idx - 1]["end_s"])
            match = KeyPhraseSegment(
                phrase=phrase,
                start_s=start_s,
                end_s=end_s,
                distance=dist,
                normalized_distance=norm_dist,
                matched_phonemes=candidate,
                phraseology_source=phraseology_source,
                target_phoneme_len=target_len,
                coverage=cov,
                composite_score=score,
            )

            if best is None or score < best_score or (
                abs(score - best_score) < 1e-9
                and (match.distance, match.start_s) < (best.distance, best.start_s)
            ):
                best_score = score
                best = match

    return best
