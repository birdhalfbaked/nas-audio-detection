from __future__ import annotations

import wave
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.audio.phonemes import Phonemizer
from app.audio.phrase_tagger import find_best_phrase_match
from app.audio.processor import AudioTranscriber


def load_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        raw_frames = wav_file.readframes(frame_count)

    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    if sample_width not in dtype_map:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    audio = np.frombuffer(raw_frames, dtype=dtype_map[sample_width]).astype(np.float32)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    max_abs = float(np.max(np.abs(audio))) if audio.size > 0 else 1.0
    if max_abs > 0:
        audio = audio / max_abs

    return audio, sample_rate


def resample_linear(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    if from_sr == to_sr or audio.size == 0:
        return audio

    duration_s = len(audio) / from_sr
    out_len = max(1, int(round(duration_s * to_sr)))
    src_x = np.linspace(0.0, duration_s, num=len(audio), endpoint=False)
    dst_x = np.linspace(0.0, duration_s, num=out_len, endpoint=False)
    return np.interp(dst_x, src_x, audio).astype(np.float32)


def main() -> None:
    fixture = ROOT / "tests" / "fixtures" / "test_audio1.wav"
    target_phrase = "contact departure"

    audio, sample_rate = load_wav_mono(fixture)
    model_sample_rate = 16_000
    audio = resample_linear(audio, sample_rate, model_sample_rate)

    transcriber = AudioTranscriber()
    segments = transcriber.transcribe_phonemes(audio, sample_rate=model_sample_rate)

    phonemizer = Phonemizer(model_path="")
    target_phonemes = phonemizer(target_phrase)
    match = find_best_phrase_match(target_phrase, target_phonemes, segments)

    if match is None:
        print(f"No match found for phrase: {target_phrase}")
        return

    print(
        f"{match.phrase}: {match.start_s:.2f}s -> {match.end_s:.2f}s "
        f"(distance={match.distance}, normalized={match.normalized_distance:.3f})"
    )
    print("Matched phonemes:", " ".join(match.matched_phonemes))


if __name__ == "__main__":
    main()
