from app.audio.phrase_tagger import find_best_phrase_match


def test_find_best_phrase_match_returns_timestamp_span() -> None:
    phrase = "contact departure"
    target_phonemes = ["k", "ɑː", "n", "t", "æ", "k", "t", "d", "ɪ", "p", "ɑː", "tʃ", "ɚ"]
    segments = [
        {"phoneme": "n", "start_s": 0.00, "end_s": 0.10},
        {"phoneme": "k", "start_s": 0.10, "end_s": 0.20},
        {"phoneme": "ɑː", "start_s": 0.20, "end_s": 0.30},
        {"phoneme": "n", "start_s": 0.30, "end_s": 0.40},
        {"phoneme": "t", "start_s": 0.40, "end_s": 0.50},
        {"phoneme": "æ", "start_s": 0.50, "end_s": 0.60},
        {"phoneme": "k", "start_s": 0.60, "end_s": 0.70},
        {"phoneme": "t", "start_s": 0.70, "end_s": 0.80},
        {"phoneme": "d", "start_s": 0.80, "end_s": 0.90},
        {"phoneme": "ɪ", "start_s": 0.90, "end_s": 1.00},
        {"phoneme": "p", "start_s": 1.00, "end_s": 1.10},
        {"phoneme": "ɑː", "start_s": 1.10, "end_s": 1.20},
        {"phoneme": "tʃ", "start_s": 1.20, "end_s": 1.30},
        {"phoneme": "ɚ", "start_s": 1.30, "end_s": 1.40},
        {"phoneme": "z", "start_s": 1.40, "end_s": 1.50},
    ]

    match = find_best_phrase_match(phrase, target_phonemes, segments)

    assert match is not None
    assert match.phrase == phrase
    assert match.start_s == 0.10
    assert match.end_s == 1.40
