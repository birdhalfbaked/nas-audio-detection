from contextlib import nullcontext

import numpy as np

from app.audio.phonemes import Phonemizer
from app.audio.phrase_tagger import KeyPhraseSegment, find_best_phrase_match

try:
    import torch
except (
    ModuleNotFoundError
):  # pragma: no cover - exercised in environments without torch.
    torch = None
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# load model and processor


class AudioTranscriber:
    PROCESSOR = None
    MODEL = None

    def __init__(self, processor=None, model=None, phrase_phonemizer=None):
        self.PROCESSOR = processor
        self.MODEL = model
        self._phrase_phonemizer = phrase_phonemizer

    @classmethod
    def _load_defaults(cls) -> tuple[Wav2Vec2Processor, Wav2Vec2ForCTC]:
        if cls.PROCESSOR is None:
            cls.PROCESSOR = Wav2Vec2Processor.from_pretrained(
                "facebook/wav2vec2-lv-60-espeak-cv-ft"
            )
        if cls.MODEL is None:
            cls.MODEL = Wav2Vec2ForCTC.from_pretrained(
                "facebook/wav2vec2-lv-60-espeak-cv-ft"
            )
        return cls.PROCESSOR, cls.MODEL

    def __call__(
        self,
        audio_data: np.ndarray,
        key_phrases: list[str],
        sample_rate: int = 16_000,
        chunk_seconds: float = 10.0,
        stride_seconds: float = 2.0,
        max_normalized_distance: float = 0.45,
        overlap_threshold: float = 0.5,
    ) -> list[KeyPhraseSegment]:
        """
        Return detected key-phrase segments with timestamps.
        """
        if not key_phrases:
            return []

        phoneme_segments = self.transcribe_phonemes(
            audio_data=audio_data,
            sample_rate=sample_rate,
            chunk_seconds=chunk_seconds,
            stride_seconds=stride_seconds,
        )
        if not phoneme_segments:
            return []

        phrase_phonemizer = self._phrase_phonemizer or Phonemizer(model_path="")
        detections: list[KeyPhraseSegment] = []
        for phrase in key_phrases:
            target_phonemes = phrase_phonemizer(phrase)
            match = find_best_phrase_match(
                phrase=phrase,
                target_phonemes=target_phonemes,
                segments=phoneme_segments,
                max_normalized_distance=max_normalized_distance,
            )
            if match is None:
                continue
            detections.append(match)

        filtered = self._resolve_overlapping_detections(
            detections=detections,
            overlap_threshold=overlap_threshold,
        )
        return sorted(filtered, key=lambda item: (item.start_s, item.distance))

    def transcribe_phonemes(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16_000,
        chunk_seconds: float = 10.0,
        stride_seconds: float = 2.0,
    ) -> list[dict[str, float | str]]:
        """
        Return phoneme-level timestamped transcription.

        Output format:
        [{"phoneme": "s", "start_s": 0.12, "end_s": 0.20}, ...]
        """
        if audio_data.size == 0:
            return []
        if self.PROCESSOR is None or self.MODEL is None:
            self.PROCESSOR, self.MODEL = self._load_defaults()

        chunk_size = max(1, int(chunk_seconds * sample_rate))
        stride_size = max(0, int(stride_seconds * sample_rate))

        segments: list[dict[str, float | str]] = []
        chunk_starts = list(range(0, len(audio_data), chunk_size))

        for chunk_index, chunk_start in enumerate(chunk_starts):
            chunk_end = min(chunk_start + chunk_size, len(audio_data))
            chunk = audio_data[chunk_start:chunk_end]
            if chunk.size == 0:
                continue

            inference_context = torch.no_grad() if torch is not None else nullcontext()
            with inference_context:
                input_values = self.PROCESSOR(
                    chunk, sampling_rate=sample_rate, return_tensors="pt"
                ).input_values
                logits = self.MODEL(input_values).logits

            predicted_ids = self._argmax_ids(logits)
            frame_count = int(logits.shape[1])
            seconds_per_frame = (chunk.size / sample_rate) / max(frame_count, 1)

            token_spans = self._ctc_token_spans(predicted_ids)
            chunk_segments = self._token_spans_to_segments(
                token_spans=token_spans,
                seconds_per_frame=seconds_per_frame,
                chunk_start_seconds=chunk_start / sample_rate,
            )

            # Use stride margins to reduce duplicate chunk overlap emissions.
            left_margin_s = 0.0 if chunk_index == 0 else (stride_size / sample_rate)
            right_margin_s = (
                float("inf")
                if chunk_index == len(chunk_starts) - 1
                else ((chunk.size - stride_size) / sample_rate)
            )

            for segment in chunk_segments:
                local_start_s = float(segment["start_s"]) - (chunk_start / sample_rate)
                if left_margin_s <= local_start_s <= right_margin_s:
                    segments.append(segment)

        return self._dedupe_adjacent_segments(segments)

    @staticmethod
    def _argmax_ids(logits) -> list[int]:
        if torch is not None and hasattr(logits, "detach"):
            return torch.argmax(logits, dim=-1)[0].detach().cpu().tolist()
        return np.argmax(logits, axis=-1)[0].tolist()

    def _ctc_token_spans(self, predicted_ids: list[int]) -> list[tuple[int, int, int]]:
        blank_id = self.PROCESSOR.tokenizer.pad_token_id
        spans: list[tuple[int, int, int]] = []

        current_id: int | None = None
        start_idx: int | None = None

        for frame_idx, token_id in enumerate(predicted_ids):
            if token_id == blank_id:
                if current_id is not None and start_idx is not None:
                    spans.append((current_id, start_idx, frame_idx - 1))
                    current_id = None
                    start_idx = None
                continue

            if current_id is None:
                current_id = token_id
                start_idx = frame_idx
                continue

            if token_id != current_id and start_idx is not None:
                spans.append((current_id, start_idx, frame_idx - 1))
                current_id = token_id
                start_idx = frame_idx

        if current_id is not None and start_idx is not None:
            spans.append((current_id, start_idx, len(predicted_ids) - 1))

        return spans

    def _token_spans_to_segments(
        self,
        token_spans: list[tuple[int, int, int]],
        seconds_per_frame: float,
        chunk_start_seconds: float,
    ) -> list[dict[str, float | str]]:
        segments: list[dict[str, float | str]] = []

        for token_id, frame_start, frame_end in token_spans:
            phoneme = self.PROCESSOR.tokenizer.convert_ids_to_tokens(token_id)
            if not phoneme or phoneme in {"<pad>", "<s>", "</s>", "|"}:
                continue

            start_s = chunk_start_seconds + (frame_start * seconds_per_frame)
            end_s = chunk_start_seconds + ((frame_end + 1) * seconds_per_frame)
            segments.append(
                {
                    "phoneme": phoneme,
                    "start_s": round(start_s, 4),
                    "end_s": round(end_s, 4),
                }
            )

        return segments

    @staticmethod
    def _dedupe_adjacent_segments(
        segments: list[dict[str, float | str]],
    ) -> list[dict[str, float | str]]:
        if not segments:
            return []

        deduped = [segments[0]]
        for segment in segments[1:]:
            prev = deduped[-1]
            same_phoneme = segment["phoneme"] == prev["phoneme"]
            contiguous = float(segment["start_s"]) <= float(prev["end_s"]) + 0.03

            if same_phoneme and contiguous:
                prev["end_s"] = max(float(prev["end_s"]), float(segment["end_s"]))
                continue

            deduped.append(segment)

        return deduped

    @staticmethod
    def _resolve_overlapping_detections(
        detections: list[KeyPhraseSegment], overlap_threshold: float
    ) -> list[KeyPhraseSegment]:
        if not detections:
            return []

        # Sort by confidence first: lower normalized distance and lower edit distance.
        ranked = sorted(
            detections,
            key=lambda item: (
                item.normalized_distance,
                item.distance,
                item.start_s,
                item.end_s,
            ),
        )

        selected: list[KeyPhraseSegment] = []
        for candidate in ranked:
            if any(
                AudioTranscriber._overlap_ratio(candidate, existing) >= overlap_threshold
                for existing in selected
            ):
                continue
            selected.append(candidate)

        return selected

    @staticmethod
    def _overlap_ratio(a: KeyPhraseSegment, b: KeyPhraseSegment) -> float:
        overlap_start = max(a.start_s, b.start_s)
        overlap_end = min(a.end_s, b.end_s)
        overlap = max(0.0, overlap_end - overlap_start)
        if overlap == 0.0:
            return 0.0

        shortest = min(a.end_s - a.start_s, b.end_s - b.start_s)
        if shortest <= 0:
            return 0.0
        return overlap / shortest
