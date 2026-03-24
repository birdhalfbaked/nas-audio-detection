"""
Phoneme-step stability for trie-beam hypotheses: survival length, rank consistency,
and non-worsening cost trend before committing a terminal.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class BeamHypothesis:
    """State for one active beam path (one trie node + span + stream index)."""

    cost: int
    #: Consecutive phoneme steps this hypothesis has been updated (after transitions).
    steps_alive: int
    #: Normalized ranks in [0, 1] after each prune (0 = best cost in beam).
    rank_history: deque[float] = field(default_factory=lambda: deque(maxlen=8))
    #: Recent edit costs (for monotonic worsening check).
    cost_history: deque[int] = field(default_factory=lambda: deque(maxlen=8))


def extend_hypothesis(parent: BeamHypothesis, new_cost: int) -> BeamHypothesis:
    """Child hypothesis after consuming one phoneme step."""
    return BeamHypothesis(
        cost=new_cost,
        steps_alive=parent.steps_alive + 1,
        rank_history=deque(parent.rank_history, maxlen=8),
        cost_history=deque(parent.cost_history, maxlen=8),
    )


def fresh_root_hypothesis() -> BeamHypothesis:
    return BeamHypothesis(cost=0, steps_alive=1, rank_history=deque(), cost_history=deque([0]))


def assign_normalized_ranks(beam: dict) -> dict:
    """
    Map each beam key to normalized rank in [0, 1], 0 = lowest cost (best).
    ``beam`` values must have a ``.cost`` attribute.
    """
    if not beam:
        return {}
    items = sorted(beam.items(), key=lambda kv: (kv[1].cost, kv[0]))
    n = len(items)
    if n == 1:
        return {items[0][0]: 0.0}
    out: dict = {}
    for rank_idx, (key, _) in enumerate(items):
        out[key] = rank_idx / (n - 1)
    return out


def apply_rank_and_cost_step(
    hyp: BeamHypothesis,
    *,
    normalized_rank: float,
) -> None:
    """Record post-prune normalized rank (cost history is updated on transitions only)."""
    hyp.rank_history.append(normalized_rank)


def hypothesis_stable_for_terminal(
    hyp: BeamHypothesis,
    *,
    min_phoneme_steps: int = 4,
    max_avg_normalized_rank: float = 0.38,
    worsening_run: int = 2,
    worsening_delta: int = 2,
    relaxed: bool = False,
) -> bool:
    """
    Require enough tracked steps, competitive average rank, and no rapid cost blow-up.

    ``relaxed`` (short streams / epsilon tail): require only ``steps_alive >= 2`` and
    no monotonic rapid cost increase; skip average-rank gate.
    """
    ch = list(hyp.cost_history)
    if len(ch) >= worsening_run + 1:
        tail = ch[-(worsening_run + 1) :]
        increases = 0
        for a, b in zip(tail, tail[1:]):
            if b >= a + worsening_delta:
                increases += 1
        if increases >= worsening_run:
            return False

    if relaxed:
        return hyp.steps_alive >= 2 and len(hyp.rank_history) >= 1

    min_steps = min_phoneme_steps
    if hyp.steps_alive < min_steps:
        return False
    rh = list(hyp.rank_history)
    if len(rh) < min_steps:
        return False
    window = rh[-min_phoneme_steps:] if len(rh) >= min_phoneme_steps else rh
    if window and sum(window) / len(window) > max_avg_normalized_rank:
        return False
    return True
