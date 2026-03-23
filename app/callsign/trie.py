"""Build a phoneme trie from callsigns / aircraft list."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.callsign.parse import spoken_phrase_variants_for_callsign
from app.utils import PhonemeTrie

if TYPE_CHECKING:
    from app.external_clients.flightradar import FlightRadarAircraft


def build_callsign_trie_for_aircraft(
    aircraft: list[FlightRadarAircraft],
    phonemizer,
) -> PhonemeTrie:
    """
    Insert each aircraft callsign using all spoken variants of the flight number.

    Terminal labels are canonical (e.g. ``Delta 2323``) while phoneme paths follow
    each spoken variant (digit-by-digit vs grouped pairs).
    """
    trie = PhonemeTrie(phonemizer=phonemizer)
    for ac in aircraft:
        variants = spoken_phrase_variants_for_callsign(ac.callsign)
        for canonical_label, spoken_phrase in variants:
            phonemes = phonemizer(spoken_phrase)
            trie.insert_phoneme_path(phonemes, canonical_label)
    return trie
