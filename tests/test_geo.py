"""Tests for ``app.geo`` distance and FAA coordinate parsing."""

import pytest

from app.geo import distance_nm, faa_coordinate_to_decimal


def test_faa_dms_round_trip_signs() -> None:
    assert faa_coordinate_to_decimal("31-52-48.010N") == pytest.approx(
        31 + 52 / 60 + 48.010 / 3600
    )
    assert faa_coordinate_to_decimal("110-31-02.110W") == pytest.approx(
        -(110 + 31 / 60 + 2.110 / 3600)
    )


def test_distance_nm_equator_one_degree() -> None:
    # ~1° latitude ≈ 60 NM
    d = distance_nm(0.0, 0.0, 1.0, 0.0)
    assert d == pytest.approx(60.0, rel=0.01)


def test_distance_nm_near_zero() -> None:
    assert distance_nm(40.0, -74.0, 40.0, -74.0) == pytest.approx(0.0, abs=1e-9)


def test_distance_nm_ten_nm_order_of_magnitude() -> None:
    # Move ~10 NM north from a mid-latitude point (rough check)
    dlat = 10.0 / 60.0
    d = distance_nm(39.0, -95.0, 39.0 + dlat, -95.0)
    assert d == pytest.approx(10.0, rel=0.02)
