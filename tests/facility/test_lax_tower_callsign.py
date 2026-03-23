"""
Example tower facility: mocked flight-data client + callsign trie with pronunciation variants.

Uses a word-token stub phonemizer so tests do not require espeak.

Test coordinates / range are fixture data only (not shipped as app config).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import wave

import pytest

from app.callsign.parse import number_spoken_variants, spoken_phrase_variants_for_callsign
from app.callsign.transcript import phonemes_from_segments
from app.external_clients.flightradar import FlightRadarAircraft
from app.facilities import Facility, get_bounds_from_center
from app.utils import PhonemeTrie

# Example tower position + search radius (NM) for tests — not application defaults.
STUB_TOWER_LAT = 33.942496
STUB_TOWER_LON = -118.408049
STUB_AIRCRAFT_RANGE_NM = 15.0


def _stub_phonemizer(text: str) -> list[str]:
    """Tokenize on whitespace; enough structure for trie + beam search tests."""
    return text.lower().split()


def _empty_common_trie() -> PhonemeTrie:
    return PhonemeTrie(phonemizer=_stub_phonemizer)


@patch("app.facilities.get_common_phraseology_trie", _empty_common_trie)
def test_dal2323_maps_to_delta_and_two_number_variants() -> None:
    variants = spoken_phrase_variants_for_callsign("DAL2323")
    assert len(variants) == 2
    labels = {canonical for canonical, _ in variants}
    assert labels == {"Delta 2323"}
    spoken = {s for _, s in variants}
    assert "Delta two three two three" in spoken
    assert "Delta twenty three twenty three" in spoken


def test_number_variants_for_2323() -> None:
    v = number_spoken_variants("2323")
    assert "two three two three" in v
    assert "twenty three twenty three" in v


@patch("app.facilities.get_common_phraseology_trie", _empty_common_trie)
def test_facility_builds_callsign_trie_from_mocked_client_with_center_and_range() -> None:
    mock_fr = MagicMock()
    mock_fr.get_nearby_aircraft.return_value = [
        FlightRadarAircraft(
            callsign="DAL2323",
            latitude=33.9,
            longitude=-118.4,
            altitude=3000,
            ground_speed=200,
            squawk="1200",
        )
    ]

    facility = Facility(
        "example_tower",
        latitude=STUB_TOWER_LAT,
        longitude=STUB_TOWER_LON,
        aircraft_range_nm=STUB_AIRCRAFT_RANGE_NM,
        flight_client=mock_fr,
        phonemizer=_stub_phonemizer,
    )
    trie = facility.build_callsign_trie()
    assert trie is not None

    mock_fr.get_nearby_aircraft.assert_called_once()
    args, _ = mock_fr.get_nearby_aircraft.call_args
    expected_bounds = get_bounds_from_center(
        STUB_TOWER_LAT, STUB_TOWER_LON, STUB_AIRCRAFT_RANGE_NM
    )
    assert args[0].as_param() == expected_bounds.as_param()

    # Digit-by-digit transcript
    q1 = _stub_phonemizer("delta two three two three")
    assert trie.search_phonemes(q1, top_k=1) == ["Delta 2323"]

    # Grouped pairs transcript
    q2 = _stub_phonemizer("delta twenty three twenty three")
    assert trie.search_phonemes(q2, top_k=1) == ["Delta 2323"]


@patch("app.facilities.get_common_phraseology_trie", _empty_common_trie)
def test_test_audio_fixture_transcript_finds_delta_2323_with_stub_asr() -> None:
    """
    End-to-end shape: load WAV, use stub ASR phonemes matching one variant,
    trie built from mocked DAL2323 must recover canonical callsign.
    """
    mock_fr = MagicMock()
    mock_fr.get_nearby_aircraft.return_value = [
        FlightRadarAircraft(
            callsign="DAL2323",
            latitude=33.9,
            longitude=-118.4,
            altitude=3000,
            ground_speed=200,
            squawk="1200",
        )
    ]
    facility = Facility(
        "example_tower",
        latitude=STUB_TOWER_LAT,
        longitude=STUB_TOWER_LON,
        aircraft_range_nm=STUB_AIRCRAFT_RANGE_NM,
        flight_client=mock_fr,
        phonemizer=_stub_phonemizer,
    )
    trie = facility.build_callsign_trie()
    assert trie is not None

    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "test_audio1.wav"
    assert fixture.is_file()

    with wave.open(str(fixture), "rb") as wav:
        _ = wav.getnframes()  # ensure readable

    # Stub ASR: pretend the audio said "delta two three two three" (word tokens).
    fake_segments = [
        {"phoneme": p, "start_s": i * 0.1, "end_s": (i + 1) * 0.1}
        for i, p in enumerate(_stub_phonemizer("delta two three two three"))
    ]
    transcript = phonemes_from_segments(fake_segments)
    hits = trie.search_phonemes(transcript, top_k=3)
    assert "Delta 2323" in hits


def test_get_bounds_from_center_symmetric_around_point() -> None:
    bounds = get_bounds_from_center(33.942496, -118.408049, 10.0)
    assert bounds.lat_max - bounds.lat_min == pytest.approx(20.0 / 60.0)
    # Longitude span wider at this latitude (cos < 1)
    assert bounds.lon_max - bounds.lon_min > bounds.lat_max - bounds.lat_min
