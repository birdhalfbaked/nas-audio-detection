"""Helpers to convert ASR phoneme segments into a flat phoneme list."""

from __future__ import annotations


def phonemes_from_segments(segments: list[dict[str, float | str]]) -> list[str]:
    return [str(segment["phoneme"]) for segment in segments]
