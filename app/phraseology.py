from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from app.audio.phonemes import Phonemizer
from app.utils import PhonemeTrie


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
COMMON_TRIE_PATH = DATA_DIR / "common_trie.json"
FACILITY_TRIES_DIR = DATA_DIR / "facility_data"


@lru_cache(maxsize=1)
def get_common_phraseology_trie() -> PhonemeTrie:
    if not COMMON_TRIE_PATH.exists():
        raise FileNotFoundError(
            f"Common phraseology trie not found at {COMMON_TRIE_PATH}. "
            "Run scripts/build_common_tries.py first."
        )
    payload = json.loads(COMMON_TRIE_PATH.read_text(encoding="utf-8"))
    return PhonemeTrie.from_dict(phonemizer=Phonemizer(model_path=""), data=payload)


@lru_cache(maxsize=32)
def get_facility_phraseology_trie(facility_id: str) -> PhonemeTrie | None:
    trie_path = FACILITY_TRIES_DIR / f"{facility_id.lower()}_trie.json"
    if not trie_path.exists():
        return None
    payload = json.loads(trie_path.read_text(encoding="utf-8"))
    return PhonemeTrie.from_dict(phonemizer=Phonemizer(model_path=""), data=payload)


def build_facility_streaming_trie(
    phonemizer: Phonemizer,
    phrases: list[str],
    phrase_sources: dict[str, str],
    facility_id: str,
) -> PhonemeTrie:
    """
    Trie for facility-only progressive beam search (fixes, etc.).

    Merging common phraseology into this trie would make the beam compete with many
    ``contact …`` branches; non-facility phrases are matched separately with sliding windows.
    """
    ft = get_facility_phraseology_trie(facility_id)
    merged = PhonemeTrie(phonemizer=phonemizer) if ft is None else ft.copy()
    known: set[str] = set(merged.words())
    for phrase in phrases:
        if phrase_sources.get(phrase) != "facility":
            continue
        if phrase not in known:
            merged.insert_phoneme_path(phonemizer(phrase), phrase)
            known.add(phrase)
    return merged
