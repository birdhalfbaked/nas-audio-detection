"""CLI-style callsign expansion for ``--callsigns``."""

from app.callsign.parse import callsign_spokens_for_detect


def test_callsign_spokens_expands_icao_and_maps_canonical() -> None:
    spoken, labels = callsign_spokens_for_detect(["DAL2323"])
    assert len(spoken) >= 1
    assert all(labels[s].startswith("Delta ") for s in spoken)
    assert labels[spoken[0]] == "Delta 2323"


def test_callsign_spokens_multiple_ids() -> None:
    spoken, labels = callsign_spokens_for_detect(["DAL2323", "UAL450"])
    canon = {labels[s] for s in spoken}
    assert "Delta 2323" in canon
    assert "United 450" in canon


def test_callsign_spokens_skips_invalid() -> None:
    spoken, labels = callsign_spokens_for_detect(["", "NOT_A_CALLSIGN", "DAL2323"])
    assert "Delta 2323" in set(labels.values())
