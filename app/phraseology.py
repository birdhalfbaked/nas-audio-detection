from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from app.audio.phonemes import Phonemizer
from app.utils import PhonemeTrie


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
COMMON_TRIE_PATH = DATA_DIR / "common_trie.json"


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
    trie_path = DATA_DIR / "facilities" / f"{facility_id.lower()}_trie.json"
    if not trie_path.exists():
        return None
    payload = json.loads(trie_path.read_text(encoding="utf-8"))
    return PhonemeTrie.from_dict(phonemizer=Phonemizer(model_path=""), data=payload)
