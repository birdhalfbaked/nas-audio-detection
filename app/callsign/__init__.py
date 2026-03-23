"""Callsign parsing, spoken variants, and phoneme trie helpers for ATC-style callsigns."""

from app.callsign.airlines import ICAO_AIRLINE_NAME
from app.callsign.parse import (
    spoken_phrase_variants_for_callsign,
    split_callsign_airline_and_digits,
)
from app.callsign.transcript import phonemes_from_segments
from app.callsign.trie import build_callsign_trie_for_aircraft

__all__ = [
    "ICAO_AIRLINE_NAME",
    "split_callsign_airline_and_digits",
    "spoken_phrase_variants_for_callsign",
    "build_callsign_trie_for_aircraft",
    "phonemes_from_segments",
]
