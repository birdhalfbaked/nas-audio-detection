from __future__ import annotations

import math

from app.audio.phonemes import Phonemizer
from app.callsign.trie import build_callsign_trie_for_aircraft
from app.external_clients.flightradar import FlightRadarBounds, FlightRadarClient
from app.phraseology import get_common_phraseology_trie, get_facility_phraseology_trie
from app.utils import PhonemeTrie


def get_bounds_from_center(
    latitude: float,
    longitude: float,
    range_nm: float,
) -> FlightRadarBounds:
    """
    Axis-aligned bounding box that contains a circle of radius ``range_nm`` nautical
    miles around ``(latitude, longitude)``.

    Uses the standard approximation: 1 NM ≡ 1' of latitude (1/60°); longitude
    spacing scales by ``cos(latitude)``.
    """
    if range_nm <= 0:
        raise ValueError("range_nm must be positive")

    d_lat = range_nm / 60.0
    cos_lat = math.cos(math.radians(latitude))
    cos_lat = max(cos_lat, 1e-6)
    d_lon = range_nm / (60.0 * cos_lat)

    return FlightRadarBounds(
        lat_min=latitude - d_lat,
        lon_min=longitude - d_lon,
        lat_max=latitude + d_lat,
        lon_max=longitude + d_lon,
    )


class Facility:
    def __init__(
        self,
        identifier: str,
        *,
        latitude: float | None = None,
        longitude: float | None = None,
        aircraft_range_nm: float | None = None,
        flight_client: FlightRadarClient | None = None,
        phonemizer=None,
    ):
        self.identifier = identifier
        self.common_phraseology_trie = get_common_phraseology_trie()
        self.facility_trie = self._load_saved_trie()
        self.latitude = latitude
        self.longitude = longitude
        self.aircraft_range_nm = aircraft_range_nm
        self._flight_client = flight_client
        self._phonemizer = phonemizer

    def search_bounds(self) -> FlightRadarBounds | None:
        """Search bounds for the configured facility center and radius, if all are set."""
        if (
            self.latitude is None
            or self.longitude is None
            or self.aircraft_range_nm is None
        ):
            return None
        return get_bounds_from_center(
            self.latitude,
            self.longitude,
            self.aircraft_range_nm,
        )

    def phrases(self) -> list[str]:
        phrases, _ = self.phrases_with_sources()
        return phrases

    def phrases_with_sources(self) -> tuple[list[str], dict[str, str]]:
        """
        Ordered phrase list and a map ``phrase -> phraseology_source``.

        ``common`` entries come from the global trie; ``facility`` from the airport trie
        (fix names, etc.). Words present in both are tagged ``common``.
        """
        sources: dict[str, str] = {}
        ordered: list[str] = []
        for w in self.common_phraseology_trie.words():
            sources[w] = "common"
            ordered.append(w)
        if self.facility_trie is not None:
            for w in self.facility_trie.words():
                if w not in sources:
                    sources[w] = "facility"
                    ordered.append(w)
        return ordered, sources

    def build_callsign_trie(self) -> PhonemeTrie | None:
        """
        Fetch aircraft in the facility's search area and build a trie of canonical
        callsign labels with multiple phoneme paths per flight-number pronunciation variant.
        """
        bounds = self.search_bounds()
        if self._flight_client is None or bounds is None:
            return None
        phonemizer = self._phonemizer or Phonemizer(model_path="")
        aircraft = self._flight_client.get_nearby_aircraft(bounds)
        return build_callsign_trie_for_aircraft(aircraft, phonemizer)

    def _load_atc_standard_trie(self) -> PhonemeTrie:
        """
        Loads the ATC standard trie that represents common callsigns and phraseology across
        all facilities and controllers
        """
        return self.common_phraseology_trie

    def _load_saved_trie(self) -> PhonemeTrie | None:
        """
        Loads a saved trie that represents identifiers that are relevant to the facility
        """
        return get_facility_phraseology_trie(self.identifier)

    def _build_and_save_trie(self, identifiers: list[str]) -> PhonemeTrie:
        """
        Builds a trie that represents identifiers that are relevant to the facility
        and saves it to a file
        """
        raise NotImplementedError(
            "Facility trie building is not implemented yet. "
            "Use scripts/build_common_tries.py for common trie generation."
        )
