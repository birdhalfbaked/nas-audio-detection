"""
Tests for FlightRadar24 client using mocked httpx.

Sample response shape follows the FR24 REST API pattern (versioned base URL, bearer auth,
`Accept-Version` header) described in the official overview:
https://fr24api.flightradar24.com/docs/endpoints/overview#usage

The live flight positions light payload is represented as a top-level `data` array of
position objects (callsign, lat/lon, alt, gs, squawk) — matching the documented
"Live flight positions light" product.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.external_clients.flightradar import (
    FlightRadarAircraft,
    FlightRadarBounds,
    FlightRadarClient,
)

# Representative JSON for GET /api/live/flight-positions/light (simplified from FR24 schema).
FR24_LIVE_POSITIONS_LIGHT_SAMPLE = {
    "data": [
        {
            "callsign": "UAL123",
            "lat": 47.45,
            "lon": -122.308,
            "alt": 3500,
            "gs": 180,
            "squawk": "1200",
        },
        {
            "callsign": "DAL456",
            "lat": 47.52,
            "lon": -122.25,
            "alt": 8200,
            "gs": 240,
            "squawk": "3456",
        },
    ]
}

# Alternate nesting some API responses use: data.items[]
FR24_NESTED_ITEMS_SAMPLE = {
    "data": {
        "items": [
            {
                "callsign": "SWA789",
                "latitude": 33.9425,
                "longitude": -118.4081,
                "altitude": 12000,
                "ground_speed": 410,
                "squawk": "7700",
            }
        ]
    }
}


@pytest.fixture
def mock_httpx_client() -> MagicMock:
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = FR24_LIVE_POSITIONS_LIGHT_SAMPLE

    mock_client = MagicMock()
    mock_client.get.return_value = mock_response

    with patch(
        "app.external_clients.flightradar.httpx.Client", return_value=mock_client
    ):
        yield mock_client


def test_get_nearby_aircraft_calls_correct_endpoint_and_params(
    mock_httpx_client: MagicMock,
) -> None:
    bounds = FlightRadarBounds(47.0, -123.0, 48.0, -121.0)
    client = FlightRadarClient(api_key="test-api-key")

    client.get_nearby_aircraft(bounds)

    mock_httpx_client.get.assert_called_once_with(
        "/live/flight-positions/light",
        params={"bounds": bounds.as_param()},
    )


def test_get_nearby_aircraft_parses_data_array_response(
    mock_httpx_client: MagicMock,
) -> None:
    bounds = FlightRadarBounds(47.0, -123.0, 48.0, -121.0)
    client = FlightRadarClient(api_key="test-api-key")

    aircraft = client.get_nearby_aircraft(bounds)

    assert len(aircraft) == 2
    assert aircraft[0] == FlightRadarAircraft(
        callsign="UAL123",
        latitude=47.45,
        longitude=-122.308,
        altitude=3500,
        ground_speed=180,
        squawk="1200",
    )
    assert aircraft[1] == FlightRadarAircraft(
        callsign="DAL456",
        latitude=47.52,
        longitude=-122.25,
        altitude=8200,
        ground_speed=240,
        squawk="3456",
    )


def test_get_nearby_aircraft_parses_nested_data_items() -> None:
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = FR24_NESTED_ITEMS_SAMPLE

    mock_client = MagicMock()
    mock_client.get.return_value = mock_response

    with patch(
        "app.external_clients.flightradar.httpx.Client", return_value=mock_client
    ):
        client = FlightRadarClient(api_key="key")
        bounds = FlightRadarBounds(33.0, -119.0, 34.0, -118.0)
        aircraft = client.get_nearby_aircraft(bounds)

    assert len(aircraft) == 1
    assert aircraft[0].callsign == "SWA789"
    assert aircraft[0].latitude == pytest.approx(33.9425)
    assert aircraft[0].longitude == pytest.approx(-118.4081)
    assert aircraft[0].altitude == 12000
    assert aircraft[0].ground_speed == 410
    assert aircraft[0].squawk == "7700"


def test_get_nearby_aircraft_skips_rows_without_coordinates() -> None:
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {"callsign": "BAD", "squawk": "0000"},  # no lat/lon
            {
                "callsign": "OK",
                "lat": 40.0,
                "lon": -74.0,
                "alt": 100,
                "gs": 50,
                "squawk": "1234",
            },
        ]
    }

    mock_client = MagicMock()
    mock_client.get.return_value = mock_response

    with patch(
        "app.external_clients.flightradar.httpx.Client", return_value=mock_client
    ):
        client = FlightRadarClient(api_key="key")
        aircraft = client.get_nearby_aircraft(
            FlightRadarBounds(39.0, -75.0, 41.0, -73.0)
        )

    assert len(aircraft) == 1
    assert aircraft[0].callsign == "OK"


def test_client_sets_bearer_and_accept_version_on_httpx_client() -> None:
    with patch("app.external_clients.flightradar.httpx.Client") as client_cls:
        FlightRadarClient(api_key="secret-token")

    client_cls.assert_called_once()
    _, kwargs = client_cls.call_args
    assert kwargs["base_url"] == "https://fr24api.flightradar24.com/api"
    assert kwargs["headers"]["Accept-Version"] == "v1"
    assert kwargs["headers"]["Authorization"] == "Bearer secret-token"
