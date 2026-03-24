from types import SimpleNamespace

import numpy as np

from app.audio.detection_quality import dedupe_same_phrase_nms_and_cooldown
from app.audio.processor import AudioTranscriber
from app.audio.phrase_tagger import KeyPhraseSegment


class _FakeTokenizer:
    pad_token_id = 0

    def __init__(self, id_to_token: dict[int, str]):
        self._id_to_token = id_to_token

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self._id_to_token.get(token_id, "")


class _FakeProcessor:
    def __init__(self, tokenizer: _FakeTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, chunk, sampling_rate: int, return_tensors: str):
        _ = sampling_rate, return_tensors
        return SimpleNamespace(input_values=np.array([chunk], dtype=np.float32))


class _FakeModel:
    def __init__(self, predicted_ids_per_call: list[list[int]], vocab_size: int = 8):
        self._predicted_ids_per_call = predicted_ids_per_call
        self._call_index = 0
        self._vocab_size = vocab_size

    def __call__(self, input_values):
        _ = input_values
        predicted_ids = self._predicted_ids_per_call[self._call_index]
        self._call_index += 1

        logits = np.full((1, len(predicted_ids), self._vocab_size), fill_value=-1000.0)
        for frame_idx, token_id in enumerate(predicted_ids):
            logits[0, frame_idx, token_id] = 1000.0

        return SimpleNamespace(logits=logits)


def test_audio_transcriber_returns_empty_for_empty_audio() -> None:
    transcriber = AudioTranscriber(
        processor=_FakeProcessor(_FakeTokenizer({1: "s"})),
        model=_FakeModel([[0]]),
    )

    assert transcriber.transcribe_phonemes(np.array([], dtype=np.float32), sample_rate=16_000) == []


def test_audio_transcriber_chunked_timestamps_with_fakes() -> None:
    tokenizer = _FakeTokenizer({1: "s", 2: "t"})
    processor = _FakeProcessor(tokenizer)
    model = _FakeModel(
        predicted_ids_per_call=[
            [0, 1, 1, 0],  # First chunk emits "s" from 0.25s to 0.75s
            [0, 2, 2, 0],  # Second chunk emits "t" from 1.25s to 1.75s
        ]
    )
    transcriber = AudioTranscriber(processor=processor, model=model)

    audio_data = np.array([0.0] * 8, dtype=np.float32)
    segments = transcriber.transcribe_phonemes(
        audio_data,
        sample_rate=4,
        chunk_seconds=1.0,
        stride_seconds=0.0,
    )

    assert segments == [
        {"phoneme": "s", "start_s": 0.25, "end_s": 0.75},
        {"phoneme": "t", "start_s": 1.25, "end_s": 1.75},
    ]


def test_audio_transcriber_detects_key_phrases() -> None:
    tokenizer = _FakeTokenizer({1: "k", 2: "t"})
    processor = _FakeProcessor(tokenizer)
    model = _FakeModel(predicted_ids_per_call=[[0, 1, 1, 0, 2, 2, 0]])

    def fake_phrase_phonemizer(text: str) -> list[str]:
        return {"contact": ["k", "t"], "departure": ["d"]}[text]

    transcriber = AudioTranscriber(
        processor=processor,
        model=model,
        phrase_phonemizer=fake_phrase_phonemizer,
    )

    detections = transcriber(
        audio_data=np.array([0.0] * 4, dtype=np.float32),
        key_phrases=["contact", "departure"],
        sample_rate=4,
        chunk_seconds=1.0,
        stride_seconds=0.0,
    )

    assert len(detections) == 1
    assert detections[0].phrase == "contact"


def test_overlap_resolution_prefers_higher_confidence_detection() -> None:
    detections = [
        KeyPhraseSegment(
            phrase="contact departure",
            start_s=1.05,
            end_s=1.76,
            distance=4,
            normalized_distance=0.308,
            matched_phonemes=["k", "ɑː"],
        ),
        KeyPhraseSegment(
            phrase="contact approach",
            start_s=1.05,
            end_s=1.68,
            distance=5,
            normalized_distance=0.417,
            matched_phonemes=["k", "ɑː"],
        ),
    ]

    resolved = AudioTranscriber._resolve_overlapping_detections(
        detections=detections, overlap_threshold=0.5
    )

    assert len(resolved) == 1
    assert resolved[0].phrase == "contact departure"


def test_overlap_resolution_prefers_common_over_facility_when_distances_equal() -> None:
    detections = [
        KeyPhraseSegment(
            phrase="alpha",
            start_s=1.0,
            end_s=2.0,
            distance=2,
            normalized_distance=0.2,
            matched_phonemes=["a"],
            phraseology_source="common",
            target_phoneme_len=5,
        ),
        KeyPhraseSegment(
            phrase="bravo",
            start_s=1.0,
            end_s=2.0,
            distance=2,
            normalized_distance=0.2,
            matched_phonemes=["a"],
            phraseology_source="facility",
            target_phoneme_len=5,
        ),
    ]

    resolved = AudioTranscriber._resolve_overlapping_detections(
        detections=detections, overlap_threshold=0.5
    )

    assert len(resolved) == 1
    assert resolved[0].phrase == "alpha"
    assert resolved[0].phraseology_source == "common"


def test_overlap_resolution_prefers_facility_when_it_fits_better() -> None:
    detections = [
        KeyPhraseSegment(
            phrase="alpha",
            start_s=1.0,
            end_s=2.0,
            distance=3,
            normalized_distance=0.35,
            matched_phonemes=["a"],
            phraseology_source="common",
            target_phoneme_len=8,
        ),
        KeyPhraseSegment(
            phrase="bravo",
            start_s=1.0,
            end_s=2.0,
            distance=1,
            normalized_distance=0.15,
            matched_phonemes=["a"],
            phraseology_source="facility",
            target_phoneme_len=6,
        ),
    ]

    resolved = AudioTranscriber._resolve_overlapping_detections(
        detections=detections, overlap_threshold=0.5
    )

    assert len(resolved) == 1
    assert resolved[0].phrase == "bravo"


def test_overlap_resolution_allows_nested_hit_when_overlap_is_small_vs_longer_span() -> None:
    """Short detection inside a longer one: ratio uses max(duration), so both can survive."""
    detections = [
        KeyPhraseSegment(
            phrase="short",
            start_s=1.0,
            end_s=1.5,
            distance=1,
            normalized_distance=0.1,
            matched_phonemes=["x"],
            phraseology_source="common",
        ),
        KeyPhraseSegment(
            phrase="longer",
            start_s=1.0,
            end_s=2.5,
            distance=1,
            normalized_distance=0.1,
            matched_phonemes=["x"],
            phraseology_source="common",
        ),
    ]

    resolved = AudioTranscriber._resolve_overlapping_detections(
        detections=detections, overlap_threshold=0.5
    )

    assert len(resolved) == 2
    phrases = {r.phrase for r in resolved}
    assert phrases == {"short", "longer"}


def test_dedupe_same_phrase_cooldown_keeps_best_composite_first() -> None:
    a = KeyPhraseSegment(
        phrase="Delta 2323",
        start_s=0.1,
        end_s=0.5,
        distance=8,
        normalized_distance=0.73,
        matched_phonemes=[],
        phraseology_source="callsign",
        target_phoneme_len=11,
        composite_score=0.73,
    )
    b = KeyPhraseSegment(
        phrase="Delta 2323",
        start_s=0.4,
        end_s=1.2,
        distance=13,
        normalized_distance=0.68,
        matched_phonemes=[],
        phraseology_source="callsign",
        target_phoneme_len=11,
        composite_score=0.68,
    )
    out = dedupe_same_phrase_nms_and_cooldown([a, b], iou_threshold=0.5, cooldown_s=0.75)
    assert len(out) == 1
    assert out[0].phrase == "Delta 2323"
    assert out[0].composite_score == 0.68


def test_dedupe_contact_common_keeps_one_per_overlap_cluster() -> None:
    approach = KeyPhraseSegment(
        phrase="contact approach",
        start_s=1.0,
        end_s=1.7,
        distance=0,
        normalized_distance=0.3,
        matched_phonemes=[],
        phraseology_source="common",
        target_phoneme_len=10,
        composite_score=0.3,
    )
    departure = KeyPhraseSegment(
        phrase="contact departure",
        start_s=1.0,
        end_s=1.8,
        distance=0,
        normalized_distance=0.2,
        matched_phonemes=[],
        phraseology_source="common",
        target_phoneme_len=12,
        composite_score=0.2,
    )
    out = AudioTranscriber._dedupe_overlapping_contact_common([approach, departure])
    assert len(out) == 1
    assert out[0].phrase == "contact departure"


def test_hybrid_sliding_common_and_streaming_facility_large_vocab() -> None:
    """Sliding windows for common ATC; facility-only trie beam for large fix vocab."""
    cd = [
        "k",
        "ɑː",
        "n",
        "t",
        "æ",
        "k",
        "t",
        "d",
        "ɪ",
        "p",
        "ɑː",
        "tʃ",
        "ɚ",
    ]
    fac = [f"F{i}" for i in range(180)]

    def phon(w: str) -> list[str]:
        if w == "contact departure":
            return cd
        return [f"z{i}" for i, _ in enumerate(w)]  # unique multiphone to inflate trie width

    phrase_sources = dict.fromkeys(fac, "facility")
    phrase_sources["contact departure"] = "common"
    key_phrases = ["contact departure"] + fac

    segs = [
        {"phoneme": p, "start_s": round(i * 0.2, 3), "end_s": round((i + 1) * 0.2, 3)}
        for i, p in enumerate(cd)
    ]

    transcriber = AudioTranscriber(phrase_phonemizer=phon)
    transcriber.transcribe_phonemes = lambda **kwargs: segs  # type: ignore[method-assign]

    hits = transcriber(
        np.zeros(8, dtype=np.float32),
        key_phrases,
        phrase_sources=phrase_sources,
        streaming_beam_width=48,
    )

    assert any(h.phrase == "contact departure" and h.distance == 0 for h in hits)
