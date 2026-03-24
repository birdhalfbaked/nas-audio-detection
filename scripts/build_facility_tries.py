"""Build per-airport facility phoneme tries from nearby fixes.

Each facility (airport) in ``data/facilities.json`` gets a trie of NAV fixes whose
coordinates fall within a given radius (default 10 NM) of the airport. Pronunciation
paths come from ``data/fixes.json`` (precomputed in ``scripts/build_nav_data.py``).

Output: ``data/facility_data/<identifier.lower()>_trie.json`` — same layout expected by
``app.phraseology.get_facility_phraseology_trie``.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.geo import distance_nm, faa_coordinate_to_decimal  # noqa: E402
from app.utils import PhonemeTrie  # noqa: E402

FIXES_PATH = ROOT / "data" / "fixes.json"
FACILITIES_PATH = ROOT / "data" / "facilities.json"
OUTPUT_DIR = ROOT / "data" / "facility_data"

DEFAULT_RADIUS_NM = 25.0
PROGRESS_EVERY = 500


def load_unique_fixes(path: Path) -> list[dict]:
    by_artcc: dict[str, list[dict]] = json.loads(path.read_text(encoding="utf-8"))
    by_id: dict[str, dict] = {}
    for _artcc, rows in by_artcc.items():
        for row in rows:
            fid = str(row.get("id", "")).strip().upper()
            if not fid or fid in by_id:
                continue
            by_id[fid] = row
    return list(by_id.values())


def parse_fix_coords(row: dict) -> tuple[float, float] | None:
    try:
        lat = faa_coordinate_to_decimal(str(row.get("lat", "")))
        lon = faa_coordinate_to_decimal(str(row.get("long", "")))
    except ValueError, TypeError:
        return None
    return lat, lon


def prepare_sorted_fixes(fixes: list[dict]) -> tuple[list[dict], list[float]]:
    indexed: list[dict] = []
    for row in fixes:
        coords = parse_fix_coords(row)
        if coords is None:
            continue
        pr = row.get("pronunciation")
        if not isinstance(pr, list) or not pr:
            continue
        lat, lon = coords
        fid = str(row.get("id", "")).strip().upper()
        if not fid:
            continue
        indexed.append(
            {
                "id": fid,
                "lat": lat,
                "long": lon,
                "pronunciation": pr,
            }
        )
    indexed.sort(key=lambda r: r["lat"])
    lats = [r["lat"] for r in indexed]
    return indexed, lats


def fixes_within_radius(
    lat0: float,
    lon0: float,
    fix_rows: list[dict],
    sorted_lats: list[float],
    radius_nm: float,
) -> list[dict]:
    lat_delta = radius_nm / 60.0
    cos_lat = math.cos(math.radians(lat0))
    max_abs_dlon = radius_nm / (60.0 * max(cos_lat, 1e-6))

    lo = bisect.bisect_left(sorted_lats, lat0 - lat_delta)
    hi = bisect.bisect_right(sorted_lats, lat0 + lat_delta)
    out: list[dict] = []
    for row in fix_rows[lo:hi]:
        if abs(row["long"] - lon0) > max_abs_dlon:
            continue
        if distance_nm(lat0, lon0, row["lat"], row["long"]) <= radius_nm:
            out.append(row)
    return out


def _noop_phonemizer(_: str) -> list[str]:
    """Trie construction only uses ``insert_phoneme_path``; search uses a real phonemizer at load time."""
    return []


def build_airport_trie(nearby: list[dict]) -> PhonemeTrie:
    trie = PhonemeTrie(phonemizer=_noop_phonemizer)
    for row in nearby:
        trie.insert_phoneme_path(row["pronunciation"], terminal_label=row["id"])
    return trie


def write_trie(path: Path, trie: PhonemeTrie) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(trie.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-airport tries from fixes within a radius (NM) of each facility."
    )
    parser.add_argument(
        "--radius-nm",
        type=float,
        default=DEFAULT_RADIUS_NM,
        help=f"Great-circle radius in nautical miles (default: {DEFAULT_RADIUS_NM})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N facilities (for testing).",
    )
    args = parser.parse_args()
    radius_nm = float(args.radius_nm)

    t0 = time.perf_counter()
    print("Loading fixes …", flush=True)
    raw_fixes = load_unique_fixes(FIXES_PATH)
    fix_rows, sorted_lats = prepare_sorted_fixes(raw_fixes)
    print(
        f"  unique fix records: {len(raw_fixes)}; "
        f"with coordinates + pronunciation: {len(fix_rows)}",
        flush=True,
    )

    facilities = json.loads(FACILITIES_PATH.read_text(encoding="utf-8"))
    if not isinstance(facilities, list):
        raise TypeError(f"Expected list in {FACILITIES_PATH}")

    if args.limit is not None:
        facilities = facilities[: max(0, args.limit)]

    written = 0
    removed = 0
    skipped_missing_coords = 0

    for i, fac in enumerate(facilities):
        ident = str(fac.get("identifier", "")).strip()
        if not ident:
            continue
        try:
            lat0 = float(fac["lat"])
            lon0 = float(fac["lon"])
        except KeyError, TypeError, ValueError:
            skipped_missing_coords += 1
            continue

        nearby = fixes_within_radius(lat0, lon0, fix_rows, sorted_lats, radius_nm)
        out_path = OUTPUT_DIR / f"{ident.lower()}_trie.json"

        if not nearby:
            if out_path.exists():
                out_path.unlink()
                removed += 1
            continue

        trie = build_airport_trie(nearby)
        write_trie(out_path, trie)
        written += 1

        if (i + 1) % PROGRESS_EVERY == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"  … {i + 1}/{len(facilities)} facilities "
                f"({written} tries written, {elapsed:.1f}s)",
                flush=True,
            )

    elapsed = time.perf_counter() - t0
    print(
        f"Done: wrote {written} tries to {OUTPUT_DIR}, "
        f"removed {removed} stale empty files, "
        f"skipped {skipped_missing_coords} facilities without coords "
        f"({elapsed:.1f}s).",
        flush=True,
    )


if __name__ == "__main__":
    main()
