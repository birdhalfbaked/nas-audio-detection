from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.audio.phonemes import Phonemizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phonemize-phrase",
        description="Convert input phrase text into phoneme tokens.",
    )
    parser.add_argument(
        "phrase",
        type=str,
        help="Phrase to phonemize (quote multi-word phrases).",
    )
    parser.add_argument(
        "--joined",
        action="store_true",
        help="Print phonemes as one space-joined line.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    phonemizer = Phonemizer(model_path="")
    phonemes = phonemizer(args.phrase)

    if args.joined:
        print(" ".join(phonemes))
    else:
        print(phonemes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
