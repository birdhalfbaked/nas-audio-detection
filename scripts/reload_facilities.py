import json
from pathlib import Path


def dms_to_decimal(value: str) -> float:
    """
    Convert FAA coordinate formats into signed decimal degrees.

    Supports:
    - DMS: '62-40-59.0000N', '164-43-19.9000W'
    - Arc-seconds: '186762.0200N', '635966.9900W' (total seconds + hemisphere)
    """
    raw = value.strip()
    if not raw:
        raise ValueError("Empty DMS value")
    hemi = raw[-1].upper()
    if hemi not in {"N", "S", "E", "W"}:
        raise ValueError(f"Invalid hemisphere in DMS value: {value!r}")

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
        # Field is total arc-seconds with hemisphere suffix.
        total_seconds = float(body)
        decimal = total_seconds / 3600.0

    if hemi in {"S", "W"}:
        decimal *= -1.0
    return decimal


def parse_apt_file(path: str):
    rows = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            # keep only APT records
            if not line.startswith("APT"):
                continue

            identifier = line[27:31].strip()
            lat_raw = line[538:550].strip()  # e.g. lat seconds (186762.0200N)
            lon_raw = line[565:577].strip()  # e.g. long seconds (635966.9900W)
            lat = dms_to_decimal(lat_raw)
            lon = dms_to_decimal(lon_raw)

            rows.append(
                {
                    "identifier": identifier,
                    "lat": lat,
                    "lon": lon,
                }
            )
    return rows


def write_json(rows, out_path: str):
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    rows = parse_apt_file("data/raw/APT.txt")
    write_json(rows, "data/facilities.json")
    print(f"Wrote {len(rows)} APT rows to data/facilities.json")
