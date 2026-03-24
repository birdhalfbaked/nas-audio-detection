"""Geodesy helpers for FAA coordinate strings and great-circle distances."""

from __future__ import annotations

import math

# Mean Earth radius for approximate NM distance (navigation use).
_EARTH_RADIUS_NM = 3440.065


def faa_coordinate_to_decimal(value: str) -> float:
    """
    Convert FAA coordinate strings to signed decimal degrees.

    Supports:
    - DMS: ``62-40-59.0000N``, ``164-43-19.9000W``
    - Arc-seconds: ``186762.0200N`` (total seconds + hemisphere)
    """
    raw = value.strip()
    if not raw:
        raise ValueError("Empty coordinate value")

    hemi = raw[-1].upper()
    if hemi not in {"N", "S", "E", "W"}:
        raise ValueError(f"Invalid hemisphere in coordinate: {value!r}")

    body = raw[:-1]
    if "-" in body:
        parts = body.split("-")
        if len(parts) != 3:
            raise ValueError(f"Invalid DMS format: {value!r}")
        degrees = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
    else:
        total_seconds = float(body)
        decimal = total_seconds / 3600.0

    if hemi in {"S", "W"}:
        decimal *= -1.0
    return decimal


def distance_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in nautical miles (haversine)."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(
        dlambda / 2
    ) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return _EARTH_RADIUS_NM * c
