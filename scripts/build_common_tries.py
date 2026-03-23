from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

common_phrases = [
    "say again",
    "disregard",
    "report",
    "turn left",
    "turn right",
    "cleared to land",
    "cleared for takeoff",
    "runway",
    "wind",
    "visibility",
    "information",
    "atis",
    "be advised",
    "declaring an emergency",
    "contact ground",
    "contact departure",
    "contact tower",
    "contact approach",
    "contact center",
    "monitor tower",
    "monitor ground",
    "taxi via",
]

alphabet = [
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliet",
    "kilo",
    "lima",
    "mike",
    "november",
    "oscar",
    "papa",
    "quebec",
    "romeo",
    "sierra",
    "tango",
    "uniform",
    "victor",
    "whiskey",
    "x-ray",
    "yankee",
    "zulu",
    "one",
    "two",
    "tree",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "niner",
    "zero",
]


def build_common_trie():
    from app.audio.phonemes import Phonemizer
    from app.utils import PhonemeTrie

    words = sorted(set(common_phrases + alphabet))
    phonemizer = Phonemizer(model_path="")
    # Probe early so build fails fast if espeak backend is unavailable.
    _ = phonemizer("students")
    trie = PhonemeTrie(phonemizer=phonemizer)
    trie.insert_many(words)
    return trie


def save_common_trie(trie, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = trie.to_dict()
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    trie = build_common_trie()
    output_path = ROOT / "data" / "common_trie.json"
    save_common_trie(trie, output_path)
    print(f"Saved common trie to {output_path}")


if __name__ == "__main__":
    main()
