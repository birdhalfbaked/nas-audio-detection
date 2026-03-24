from collections import defaultdict
from contextlib import nullcontext

import numpy as np

from app.audio.detection_quality import (
    dedupe_same_phrase_nms_and_cooldown,
    passes_min_duration,
)
from app.audio.phonemes import Phonemizer
from app.audio.phrase_tagger import KeyPhraseSegment, find_best_phrase_match
from app.audio.streaming_matcher import detect_phrases_streaming
from app.utils import PhonemeTrie


def _sliding_max_norm_for_source(source: str, global_max: float) -> float:
    """
    Floors tolerate wav2vec-vs-espeak mismatch for spoken callsigns and standard phraseology.

    Facility phrases use streaming (separate trie); this applies to sliding matches only.
    """
    if source == "callsign":
        return max(global_max, 0.75)
    if source in ("common", "explicit"):
        return max(global_max, 0.55)
    return global_max

# Lower is better when sorting overlap candidates (see ``_detection_rank_key``).
# On ties, prefer explicit instructions, then standard phraseology, then facility fixes
# (reduces spurious short fix hits vs ATC phrases like "contact departure").
_PHRASEOLOGY_SCOPE_RANK = {"explicit": 0, "callsign": 0, "common": 1, "facility": 2}

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
        *,
        phrase_sources: dict[str, str] | None = None,
        facility_streaming_trie: PhonemeTrie | None = None,
        sample_rate: int = 16_000,
        chunk_seconds: float = 10.0,
        stride_seconds: float = 2.0,
        max_normalized_distance: float = 0.45,
        max_phoneme_len_delta: int = 2,
        overlap_threshold: float = 0.5,
        streaming_beam_width: int = 128,
        dedupe_iou_threshold: float = 0.5,
        dedupe_cooldown_s: float = 0.75,
        streaming_stability_min_phoneme_steps: int = 4,
        streaming_stability_max_avg_rank: float = 0.38,
    ) -> list[KeyPhraseSegment]:
        """
        Return detected key-phrase segments with timestamps.

        **Common, explicit, and callsign** phrases use sliding-window edit distance (tolerant
        to ASR-vs-dictionary phoneme mismatch).         **Facility** phrases (fixes, etc.) use a
        progressive trie beam over ``facility_streaming_trie`` so large fix vocabularies stay
        tractable. If ``facility_streaming_trie`` is omitted but facility phrases are present,
        a trie is built from those phrases only.

        After matching, results are filtered for overlap (see ``overlap_threshold``), short
        nested fragments, then duplicate callsign lines (best fit per phrase), overlapping
        ``contact …`` common hits (best per time cluster), and duplicate facility/explicit
        spans for the same phrase.
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
        sources = phrase_sources or {}
        non_facility = [p for p in key_phrases if sources.get(p, "common") != "facility"]
        facility_only = [p for p in key_phrases if sources.get(p) == "facility"]

        detections: list[KeyPhraseSegment] = []
        for phrase in non_facility:
            src = sources.get(phrase, "common")
            max_norm = _sliding_max_norm_for_source(src, max_normalized_distance)
            match = find_best_phrase_match(
                phrase=phrase,
                target_phonemes=phrase_phonemizer(phrase),
                segments=phoneme_segments,
                max_len_delta=max_phoneme_len_delta,
                max_normalized_distance=max_norm,
                phraseology_source=src,
                apply_length_tier=(src != "callsign"),
            )
            if match is not None:
                detections.append(match)

        if facility_only:
            trie = facility_streaming_trie
            if trie is None:
                trie = PhonemeTrie(phonemizer=phrase_phonemizer)
                for phrase in facility_only:
                    trie.insert_phoneme_path(phrase_phonemizer(phrase), phrase)
            detections.extend(
                detect_phrases_streaming(
                    trie,
                    facility_only,
                    phoneme_segments,
                    sources,
                    phrase_phonemizer,
                    max_normalized_distance=max_normalized_distance,
                    beam_width=streaming_beam_width,
                    stability_min_phoneme_steps=streaming_stability_min_phoneme_steps,
                    stability_max_avg_normalized_rank=streaming_stability_max_avg_rank,
                )
            )

        filtered = [
            d
            for d in detections
            if passes_min_duration(
                d.start_s, d.end_s, d.phraseology_source, d.target_phoneme_len
            )
        ]
        filtered = self._resolve_overlapping_detections(
            detections=filtered,
            overlap_threshold=overlap_threshold,
        )
        filtered = AudioTranscriber._drop_short_nested_in_longer(filtered)
        filtered = dedupe_same_phrase_nms_and_cooldown(
            filtered,
            iou_threshold=dedupe_iou_threshold,
            cooldown_s=dedupe_cooldown_s,
        )
        filtered = AudioTranscriber._dedupe_overlapping_contact_common(filtered)
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

        # Prefer better phoneme fit, longer time coverage, then explicit/facility scope.
        ranked = sorted(
            detections,
            key=AudioTranscriber._detection_rank_key,
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
    def _time_span_subsumed(
        inner: KeyPhraseSegment, outer: KeyPhraseSegment, *, eps: float = 0.05
    ) -> bool:
        return outer.start_s - eps <= inner.start_s and inner.end_s <= outer.end_s + eps

    @staticmethod
    def _cluster_by_overlap_ratio(
        items: list[KeyPhraseSegment], overlap_threshold: float
    ) -> list[list[KeyPhraseSegment]]:
        """Group detections transitively when pairwise overlap / longest-span ratio is high."""
        n = len(items)
        if n <= 1:
            return [items] if items else []
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            pi, pj = find(i), find(j)
            if pi != pj:
                parent[pi] = pj

        for i in range(n):
            for j in range(i + 1, n):
                if AudioTranscriber._overlap_ratio(items[i], items[j]) >= overlap_threshold:
                    union(i, j)

        buckets: dict[int, list[KeyPhraseSegment]] = defaultdict(list)
        for i in range(n):
            buckets[find(i)].append(items[i])
        return list(buckets.values())

    @staticmethod
    def _best_in_cluster(cluster: list[KeyPhraseSegment]) -> KeyPhraseSegment:
        return min(cluster, key=AudioTranscriber._detection_rank_key)

    @staticmethod
    def _dedupe_overlapping_contact_common(
        detections: list[KeyPhraseSegment],
        *,
        overlap_threshold: float = 0.5,
    ) -> list[KeyPhraseSegment]:
        """
        Overlapping ``contact …`` common-phraseology hits usually share one acoustic event;
        keep the best-scoring phrase per overlap cluster.
        """
        contact = [
            d
            for d in detections
            if d.phraseology_source == "common" and d.phrase.startswith("contact ")
        ]
        rest = [
            d
            for d in detections
            if not (d.phraseology_source == "common" and d.phrase.startswith("contact "))
        ]
        if len(contact) <= 1:
            return detections
        clusters = AudioTranscriber._cluster_by_overlap_ratio(contact, overlap_threshold)
        picked = [AudioTranscriber._best_in_cluster(c) for c in clusters]
        return rest + picked

    @staticmethod
    def _drop_short_nested_in_longer(
        detections: list[KeyPhraseSegment],
    ) -> list[KeyPhraseSegment]:
        """
        Remove trivial common-word hits (e.g. digit names) that sit inside a longer match
        (typically callsign or multi-word ATC) on the same audio span.
        """
        if len(detections) < 2:
            return detections
        out: list[KeyPhraseSegment] = []
        for cand in detections:
            if cand.phraseology_source in ("facility", "callsign"):
                out.append(cand)
                continue
            if cand.target_phoneme_len > 4:
                out.append(cand)
                continue
            if any(
                longer is not cand
                and longer.target_phoneme_len >= 8
                and (
                    AudioTranscriber._overlap_ratio(cand, longer) >= 0.5
                    or AudioTranscriber._time_span_subsumed(cand, longer)
                )
                for longer in detections
            ):
                continue
            out.append(cand)
        return out

    @staticmethod
    def _rank_score(item: KeyPhraseSegment) -> float:
        if item.composite_score is not None:
            return item.composite_score
        return item.normalized_distance

    @staticmethod
    def _detection_rank_key(item: KeyPhraseSegment) -> tuple:
        duration = max(0.0, item.end_s - item.start_s)
        scope = _PHRASEOLOGY_SCOPE_RANK.get(
            item.phraseology_source,
            len(_PHRASEOLOGY_SCOPE_RANK) + 1,
        )
        return (
            AudioTranscriber._rank_score(item),
            item.normalized_distance,
            item.distance,
            -duration,
            -item.target_phoneme_len,
            scope,
            item.start_s,
            item.end_s,
            item.phrase,
        )

    @staticmethod
    def _overlap_ratio(a: KeyPhraseSegment, b: KeyPhraseSegment) -> float:
        overlap_start = max(a.start_s, b.start_s)
        overlap_end = min(a.end_s, b.end_s)
        overlap = max(0.0, overlap_end - overlap_start)
        if overlap == 0.0:
            return 0.0

        # Use the longer span so a short spurious hit nested inside a longer phrase
        # does not dominate the ratio (min() would often report near-total overlap).
        longest = max(a.end_s - a.start_s, b.end_s - b.start_s)
        if longest <= 0:
            return 0.0
        return overlap / longest
