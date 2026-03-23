from __future__ import annotations

import argparse
import json
from pathlib import Path
import wave

import numpy as np

from app.audio.processor import AudioTranscriber
from app.facilities import Facility
from app.phraseology import get_common_phraseology_trie


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


def _resolve_phrases(phrases: list[str], phrases_file: Path | None) -> list[str]:
    resolved = [phrase.strip() for phrase in phrases if phrase.strip()]
    if phrases_file is not None:
        file_phrases = [
            line.strip()
            for line in phrases_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        resolved.extend(file_phrases)
    # Keep order, dedupe exact repeats.
    return list(dict.fromkeys(resolved))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flight-audio-detection",
        description="Detect key ATC phrases from an input WAV file.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect = subparsers.add_parser(
        "detect",
        help="Process a WAV file and detect key phrases.",
    )
    detect.add_argument(
        "--audio-file",
        type=Path,
        required=True,
        help="Path to input WAV file.",
    )
    detect.add_argument(
        "--facility",
        type=str,
        help="Optional facility ID; includes facility-specific phraseology.",
    )
    detect.add_argument(
        "--phrase",
        action="append",
        default=[],
        help="Key phrase to detect; repeatable.",
    )
    detect.add_argument(
        "--phrases-file",
        type=Path,
        help="Optional newline-delimited phrase file.",
    )
    detect.add_argument(
        "--target-sample-rate",
        type=int,
        default=16_000,
        help="Resample target rate for model input.",
    )
    detect.add_argument("--chunk-seconds", type=float, default=10.0)
    detect.add_argument("--stride-seconds", type=float, default=2.0)
    detect.add_argument("--max-normalized-distance", type=float, default=0.45)
    detect.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON output instead of table output.",
    )

    return parser


def _run_detect(args: argparse.Namespace) -> int:
    phrases = _resolve_phrases(args.phrase, args.phrases_file)
    if args.facility:
        phrases = list(dict.fromkeys(Facility(args.facility).phrases() + phrases))
    elif not phrases:
        phrases = get_common_phraseology_trie().words()
    if not phrases:
        raise ValueError("No phrases available. Provide phrases or generate a common trie first.")
    if not args.audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {args.audio_file}")

    audio, sample_rate = load_wav_mono(args.audio_file)
    audio = resample_linear(audio, sample_rate, args.target_sample_rate)

    transcriber = AudioTranscriber()
    detections = transcriber(
        audio_data=audio,
        key_phrases=phrases,
        sample_rate=args.target_sample_rate,
        chunk_seconds=args.chunk_seconds,
        stride_seconds=args.stride_seconds,
        max_normalized_distance=args.max_normalized_distance,
    )

    if args.json:
        print(json.dumps([segment.__dict__ for segment in detections], indent=2))
        return 0

    if not detections:
        print("No key phrases detected.")
        return 0

    print("Detected key phrases:")
    for segment in detections:
        print(
            f"- {segment.phrase}: {segment.start_s:.2f}s -> {segment.end_s:.2f}s "
            f"(distance={segment.distance}, normalized={segment.normalized_distance:.3f})"
        )
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "detect":
        return _run_detect(args)
    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
