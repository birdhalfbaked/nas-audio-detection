from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KeyPhraseSegment:
    phrase: str
    start_s: float
    end_s: float
    distance: int
    normalized_distance: float
    matched_phonemes: list[str]


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


def find_best_phrase_match(
    phrase: str,
    target_phonemes: list[str],
    segments: list[dict[str, float | str]],
    max_len_delta: int = 2,
    max_normalized_distance: float = 0.45,
) -> KeyPhraseSegment | None:
    if not target_phonemes or not segments:
        return None

    phonemes = [str(segment["phoneme"]) for segment in segments]
    target_len = len(target_phonemes)
    min_window = max(1, target_len - max_len_delta)
    max_window = target_len + max_len_delta

    best: KeyPhraseSegment | None = None
    for start_idx in range(len(phonemes)):
        for window_len in range(min_window, max_window + 1):
            end_idx = start_idx + window_len
            if end_idx > len(phonemes):
                continue

            candidate = phonemes[start_idx:end_idx]
            dist = _edit_distance(candidate, target_phonemes)
            norm_dist = dist / max(target_len, 1)
            if norm_dist > max_normalized_distance:
                continue

            start_s = float(segments[start_idx]["start_s"])
            end_s = float(segments[end_idx - 1]["end_s"])
            match = KeyPhraseSegment(
                phrase=phrase,
                start_s=start_s,
                end_s=end_s,
                distance=dist,
                normalized_distance=norm_dist,
                matched_phonemes=candidate,
            )

            if best is None or (match.distance, match.start_s) < (best.distance, best.start_s):
                best = match

    return best
