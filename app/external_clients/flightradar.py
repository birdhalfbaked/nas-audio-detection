"""
FlightRadar24 API client

custom since the sdk offered is not the most ergonomic to use atm.
Mostly just httpx client wrapper around the api.
"""

from dataclasses import dataclass

import httpx


class FlightRadarBounds:
    """
    FlightRadarBounds represents a bounding box in the format of [lat_min, lon_min, lat_max, lon_max]
    """

    def __init__(self, lat_min: float, lon_min: float, lat_max: float, lon_max: float):
        self.lat_min = lat_min
        self.lon_min = lon_min
        self.lat_max = lat_max
        self.lon_max = lon_max

    def as_param(self) -> str:
        return f"{self.lat_min:0.3f},{self.lon_min:0.3f},{self.lat_max:0.3f},{self.lon_max:0.3f}"


@dataclass
class FlightRadarAircraft:
    callsign: str
    latitude: float
    longitude: float
    altitude: int
    ground_speed: int
    squawk: str


class FlightRadarClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = httpx.Client(
            base_url="https://fr24api.flightradar24.com/api",
            headers={
                "Accept-Version": "v1",
                "Authorization": f"Bearer {self.api_key}",
            },
            timeout=20.0,
        )

    def get_nearby_aircraft(
        self, bounds: FlightRadarBounds
    ) -> list[FlightRadarAircraft]:
        response = self.client.get(
            "/live/flight-positions/light",
            params={"bounds": bounds.as_param()},
        )
        response.raise_for_status()

        payload = response.json()
        items = self._extract_items(payload)
        aircraft: list[FlightRadarAircraft] = []
        for item in items:
            callsign = str(item.get("callsign") or "")
            latitude = self._as_float(item, "lat", "latitude")
            longitude = self._as_float(item, "lon", "longitude")
            altitude = self._as_int(item, "alt", "altitude")
            ground_speed = self._as_int(item, "gs", "ground_speed", "groundSpeed")
            squawk = str(item.get("squawk") or "")

            if latitude is None or longitude is None:
                continue

            aircraft.append(
                FlightRadarAircraft(
                    callsign=callsign,
                    latitude=latitude,
                    longitude=longitude,
                    altitude=altitude or 0,
                    ground_speed=ground_speed or 0,
                    squawk=squawk,
                )
            )

        return aircraft

    @staticmethod
    def _extract_items(payload: object) -> list[dict]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            data = payload.get("data")
            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict)]
            if isinstance(data, dict):
                rows = data.get("items") or data.get("rows")
                if isinstance(rows, list):
                    return [item for item in rows if isinstance(item, dict)]
        return []

    @staticmethod
    def _as_float(item: dict, *keys: str) -> float | None:
        for key in keys:
            value = item.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _as_int(item: dict, *keys: str) -> int | None:
        for key in keys:
            value = item.get(key)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return None
