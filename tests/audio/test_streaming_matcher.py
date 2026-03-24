from app.audio.streaming_matcher import detect_phrases_streaming
from app.utils import PhonemeTrie


def _letter_phonemizer(word: str) -> list[str]:
    return list(word)


def _segments(tokens: str) -> list[dict]:
    out: list[dict] = []
    for i, tok in enumerate(tokens):
        out.append({"phoneme": tok, "start_s": float(i) * 0.1, "end_s": float(i + 1) * 0.1})
    return out


def test_streaming_detects_exact_match() -> None:
    trie = PhonemeTrie(phonemizer=_letter_phonemizer)
    trie.insert_phoneme_path(_letter_phonemizer("cat"), "cat")
    segs = _segments("cat")
    hits = detect_phrases_streaming(
        trie,
        ["cat"],
        segs,
        {"cat": "common"},
        _letter_phonemizer,
        max_normalized_distance=0.45,
        beam_width=32,
    )
    assert len(hits) == 1
    assert hits[0].phrase == "cat"
    assert hits[0].distance == 0


def test_streaming_one_substitution_still_matches() -> None:
    trie = PhonemeTrie(phonemizer=_letter_phonemizer)
    trie.insert_phoneme_path(_letter_phonemizer("cat"), "cat")
    segs = _segments("cxt")  # 'a' misheard as 'x'
    hits = detect_phrases_streaming(
        trie,
        ["cat"],
        segs,
        {"cat": "common"},
        _letter_phonemizer,
        max_normalized_distance=0.45,
        beam_width=32,
    )
    assert len(hits) >= 1
    best = min(hits, key=lambda h: (h.distance, h.normalized_distance))
    assert best.phrase == "cat"
    assert best.distance == 1


def test_facility_streaming_respects_length_tier_max_norm() -> None:
    """Short facility references use a strict length-tier cap even when global max is very loose."""
    trie = PhonemeTrie(phonemizer=_letter_phonemizer)
    trie.insert_phoneme_path(_letter_phonemizer("abcd"), "abcd")
    # All wrong phonemes → cost 4 / 4 = 1.0 normalized; must not match when cap is 0.5.
    segs = _segments("xxxx")
    hits = detect_phrases_streaming(
        trie,
        ["abcd"],
        segs,
        {"abcd": "facility"},
        _letter_phonemizer,
        max_normalized_distance=1.0,
        beam_width=32,
    )
    assert hits == []


def test_streaming_trailing_reference_phoneme_via_epsilon() -> None:
    trie = PhonemeTrie(phonemizer=_letter_phonemizer)
    trie.insert_phoneme_path(_letter_phonemizer("cat"), "cat")
    # Audio missing final "t" — epsilon insert completes the trie path.
    segs = _segments("ca")
    hits = detect_phrases_streaming(
        trie,
        ["cat"],
        segs,
        {"cat": "common"},
        _letter_phonemizer,
        max_normalized_distance=0.45,
        beam_width=32,
    )
    assert any(h.phrase == "cat" and h.distance == 1 for h in hits)
