from collections import deque

from app.audio.streaming_stability import (
    BeamHypothesis,
    assign_normalized_ranks,
    hypothesis_stable_for_terminal,
)


def test_assign_normalized_ranks_extremes() -> None:
    a = BeamHypothesis(1, 1)
    b = BeamHypothesis(5, 1)
    c = BeamHypothesis(9, 1)
    beam = {(0, 0, 0): a, (1, 0, 0): b, (2, 0, 0): c}
    r = assign_normalized_ranks(beam)
    assert r[(0, 0, 0)] == 0.0
    assert r[(2, 0, 0)] == 1.0


def test_hypothesis_stable_strict_vs_relaxed() -> None:
    h = BeamHypothesis(
        cost=2,
        steps_alive=5,
        rank_history=deque([0.1, 0.15, 0.2, 0.12], maxlen=8),
        cost_history=deque([0, 1, 2, 2, 2], maxlen=8),
    )
    assert hypothesis_stable_for_terminal(
        h, min_phoneme_steps=4, max_avg_normalized_rank=0.38, relaxed=False
    )

    short = BeamHypothesis(
        cost=1,
        steps_alive=2,
        rank_history=deque([0.2], maxlen=8),
        cost_history=deque([0, 1], maxlen=8),
    )
    assert not hypothesis_stable_for_terminal(
        short, min_phoneme_steps=4, max_avg_normalized_rank=0.38, relaxed=False
    )
    assert hypothesis_stable_for_terminal(
        short, min_phoneme_steps=4, max_avg_normalized_rank=0.38, relaxed=True
    )
