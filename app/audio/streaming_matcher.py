"""
Progressive phrase detection: walk a merged phoneme trie as timestamped phonemes arrive,
then finalize candidates once paths end or the stream finishes.

Beam entries carry :class:`BeamHypothesis` state (cost, survival steps, rank/cost history)
so terminals are only accepted when stable over phoneme steps (see ``streaming_stability``).
"""

from __future__ import annotations

import math

from app.audio.detection_quality import (
    composite_score as build_composite_score,
    coverage_ratio,
    length_tier_max_normalized_distance,
)
from app.audio.phrase_tagger import KeyPhraseSegment
from app.audio.streaming_stability import (
    BeamHypothesis,
    apply_rank_and_cost_step,
    assign_normalized_ranks,
    extend_hypothesis,
    fresh_root_hypothesis,
    hypothesis_stable_for_terminal,
)

BeamKey = tuple[int, int, int]  # id(node), match_start_idx, next_stream_idx


def _facility_effective_max_norm(global_max: float, target_phoneme_len: int) -> float:
    return min(global_max, length_tier_max_normalized_distance(target_phoneme_len))


def _max_norm_for_phrase(
    phrase: str,
    source: str,
    global_max: float,
    sources: dict[str, str],
    target_phoneme_len: int,
) -> float:
    if sources.get(phrase, source) == "facility":
        return _facility_effective_max_norm(global_max, target_phoneme_len)
    return global_max


def _global_max_edits(
    phrases: list[str],
    phrase_sources: dict[str, str],
    global_max: float,
    phonemizer,
) -> int:
    caps: list[int] = []
    for phrase in phrases:
        n = max(1, len(phonemizer(phrase)))
        mn = _max_norm_for_phrase(
            phrase,
            phrase_sources.get(phrase, "common"),
            global_max,
            phrase_sources,
            n,
        )
        caps.append(int(math.ceil(mn * n)) + 2)
    return max(caps) if caps else 4


def _epsilon_close(
    beam: dict[BeamKey, BeamHypothesis],
    nodes: dict[int, object],
    max_cost: int,
) -> dict[BeamKey, BeamHypothesis]:
    """Reference-only extensions (audio is missing a trie phoneme)."""
    changed = True
    while changed:
        changed = False
        for (nid, start, ni), hyp in list(beam.items()):
            if hyp.cost >= max_cost:
                continue
            node = nodes.get(nid)
            if node is None or not getattr(node, "children", None):
                continue
            for _ph, child in node.children.items():
                nc = hyp.cost + 1
                if nc > max_cost:
                    continue
                cid = id(child)
                nodes[cid] = child
                key = (cid, start, ni)
                child_hyp = extend_hypothesis(hyp, nc)
                prev = beam.get(key)
                if prev is None or nc < prev.cost:
                    beam[key] = child_hyp
                    changed = True
    return beam


def _prune_beam(beam: dict[BeamKey, BeamHypothesis], width: int) -> dict[BeamKey, BeamHypothesis]:
    if len(beam) <= width:
        return beam
    ranked = sorted(beam.items(), key=lambda item: (item[1].cost, item[0][1], item[0][2]))
    return dict(ranked[:width])


def detect_phrases_streaming(
    trie: "PhonemeTrie",
    phrases: list[str],
    segments: list[dict[str, float | str]],
    phrase_sources: dict[str, str],
    phonemizer,
    *,
    max_normalized_distance: float = 0.45,
    beam_width: int = 64,
    stability_min_phoneme_steps: int = 4,
    stability_max_avg_normalized_rank: float = 0.38,
) -> list[KeyPhraseSegment]:
    """
    Streaming trie beam over ``segments``; emit best span per (phrase, start_index).

    Terminals are recorded only when :func:`hypothesis_stable_for_terminal` passes
    (enough phoneme steps, competitive rank history, no rapid cost blow-up).
    """
    root = trie._root
    nodes: dict[int, object] = {id(root): root}
    max_cost = _global_max_edits(phrases, phrase_sources, max_normalized_distance, phonemizer)
    target_len_by_phrase = {p: max(1, len(phonemizer(p))) for p in phrases}

    rid = id(root)
    beam: dict[BeamKey, BeamHypothesis] = {(rid, 0, 0): fresh_root_hypothesis()}
    best_hits: dict[tuple[str, int], KeyPhraseSegment] = {}

    def record_terminal(
        node: object,
        start_i: int,
        end_i: int,
        hyp: BeamHypothesis,
    ) -> None:
        terms = getattr(node, "terminal_words", None) or set()
        if not terms:
            return
        if not (
            hypothesis_stable_for_terminal(
                hyp,
                min_phoneme_steps=stability_min_phoneme_steps,
                max_avg_normalized_rank=stability_max_avg_normalized_rank,
                relaxed=False,
            )
            or hypothesis_stable_for_terminal(
                hyp,
                min_phoneme_steps=stability_min_phoneme_steps,
                max_avg_normalized_rank=stability_max_avg_normalized_rank,
                relaxed=True,
            )
        ):
            return
        cost = hyp.cost
        start_s = float(segments[start_i]["start_s"])
        end_s = float(segments[end_i]["end_s"])
        matched = [str(segments[k]["phoneme"]) for k in range(start_i, end_i + 1)]
        for phrase in terms:
            tl = target_len_by_phrase.get(phrase, 1)
            src = phrase_sources.get(phrase, "common")
            max_norm = _max_norm_for_phrase(
                phrase, src, max_normalized_distance, phrase_sources, tl
            )
            norm = cost / tl
            if norm > max_norm:
                continue
            cov = coverage_ratio(len(matched), tl)
            comp = build_composite_score(norm, cov, tl)
            seg = KeyPhraseSegment(
                phrase=phrase,
                start_s=start_s,
                end_s=end_s,
                distance=cost,
                normalized_distance=norm,
                matched_phonemes=matched,
                phraseology_source=src,
                target_phoneme_len=tl,
                coverage=cov,
                composite_score=comp,
            )
            key = (phrase, start_i)
            prev = best_hits.get(key)
            if prev is None or (
                seg.composite_score,
                seg.distance,
                seg.normalized_distance,
                -float(seg.end_s),
            ) < (
                prev.composite_score or prev.normalized_distance,
                prev.distance,
                prev.normalized_distance,
                -float(prev.end_s),
            ):
                best_hits[key] = seg

    def merge_next(
        next_beam: dict[BeamKey, BeamHypothesis],
        key: BeamKey,
        new_cost: int,
        parent: BeamHypothesis | None,
    ) -> None:
        if parent is None:
            h = fresh_root_hypothesis()
            h.cost = new_cost
            h.cost_history.clear()
            h.cost_history.append(new_cost)
        else:
            h = extend_hypothesis(parent, new_cost)
        prev = next_beam.get(key)
        if prev is None or new_cost < prev.cost:
            next_beam[key] = h

    for i, seg in enumerate(segments):
        p = str(seg["phoneme"])
        next_beam: dict[BeamKey, BeamHypothesis] = {}

        merge_next(next_beam, (rid, i, i), 0, None)

        for (nid, start, ni), hyp in beam.items():
            if ni != i:
                continue
            node = nodes.get(nid)
            if node is None:
                continue
            if hyp.cost + 1 <= max_cost:
                merge_next(next_beam, (nid, start, i + 1), hyp.cost + 1, hyp)
            for edge_ph, child in node.children.items():
                step = 0 if edge_ph == p else 1
                nc = hyp.cost + step
                if nc <= max_cost:
                    cid = id(child)
                    nodes[cid] = child
                    merge_next(next_beam, (cid, start, i + 1), nc, hyp)

        next_beam = _epsilon_close(next_beam, nodes, max_cost)
        next_beam = _prune_beam(next_beam, beam_width)

        ranks = assign_normalized_ranks(next_beam)
        for key, hyp in next_beam.items():
            r = ranks.get(key, 0.0)
            apply_rank_and_cost_step(hyp, normalized_rank=r)

        end_i = i
        for (nid, start, ni), hyp in next_beam.items():
            if ni != i + 1:
                continue
            node = nodes.get(nid)
            if node is not None:
                record_terminal(node, start, end_i, hyp)

        beam = next_beam

    final = _epsilon_close(dict(beam), nodes, max_cost)
    final = _prune_beam(final, beam_width)
    ranks = assign_normalized_ranks(final)
    for key, hyp in final.items():
        r = ranks.get(key, 0.0)
        apply_rank_and_cost_step(hyp, normalized_rank=r)

    last_i = len(segments) - 1
    if last_i >= 0:
        for (nid, start, ni), hyp in final.items():
            if ni == len(segments):
                node = nodes.get(nid)
                if node is not None:
                    record_terminal(node, start, last_i, hyp)

    return list(best_hits.values())
