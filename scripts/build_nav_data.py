import csv
import json
import sys
import time
from collections import defaultdict
from itertools import islice
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# dictionary of ARTCC to FIX list
FIXES = defaultdict(list)


BLANK = "__blank"


FIELD_WIDTHS = [
    ("type", 4),
    ("fix_id", 30),
    ("fix_state_name", 30),
    ("region_code", 2),
    ("lat", 14),
    ("long", 14),
    ("category", 3),
    ("info_1", 22),
    ("info_2", 22),
    ("previous_name", 33),
    ("charts", 38),
    ("to_publish", 1),
    ("fix_use", 15),
    ("nas_id", 5),
    ("artcc_high", 4),
    ("artcc_low", 4),
    ("country", 30),
    ("pitch", 1),
    ("catch", 1),
    ("atcaa", 1),
    (BLANK, 192),
]

BATCH_SIZE = 32
PROGRESS_EVERY_BATCHES = 100


def load_pronunciation_overrides(path: Path) -> dict[str, str]:
    overrides: dict[str, str] = {}
    if not path.exists():
        return overrides

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            fix_id = row[0].strip().upper()
            spoken = row[1].strip()
            if fix_id and spoken:
                overrides[fix_id] = spoken
    return overrides


def parse_fixed_width_line(line: str) -> dict[str, str]:
    cursor = 0
    parsed: dict[str, str] = {}
    for field_name, width in FIELD_WIDTHS:
        raw_value = line[cursor : cursor + width]
        cursor += width
        value = raw_value.strip()
        if field_name == BLANK:
            continue
        parsed[field_name] = value
    return parsed


def build_compact_fix(
    raw_fix: dict[str, str],
    pronunciation: list[str],
) -> dict[str, str | list[str]]:
    fix_id = raw_fix.get("fix_id", "").upper()
    return {
        "id": fix_id,
        "lat": raw_fix.get("lat", ""),
        "long": raw_fix.get("long", ""),
        "pronunciation": pronunciation,
        "artcc_high": raw_fix.get("artcc_high", ""),
        "artcc_low": raw_fix.get("artcc_low", ""),
    }


def _iter_batches(iterable, batch_size: int):
    iterator = iter(iterable)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            return
        yield batch


def process_batch(
    lines: list[str],
) -> list[dict[str, str | list[str]]]:
    processed: list[dict[str, str | list[str]]] = []
    for line in lines:
        if not line.startswith("FIX1"):
            continue
        raw_fix = parse_fixed_width_line(line.rstrip("\n"))
        processed.append(raw_fix)
    return processed


def pronunciation_from_tokens(
    spoken: str,
    phonemizer,
) -> list[str]:
    tokens = [token.strip() for token in spoken.split(" ") if token.strip()]
    pronunciation: list[str] = []
    for token in tokens:
        pronunciation.extend(phonemizer(token))
    return pronunciation


def load_nav_fixes() -> dict[str, int]:
    from app.audio.phonemes import Phonemizer

    FIXES.clear()

    overrides = load_pronunciation_overrides(Path("data/pronunciations.csv"))
    phonemizer = Phonemizer(model_path="")
    _ = phonemizer("students")

    metrics = {
        "lines_seen": 0,
        "batches_processed": 0,
        "fix_records": 0,
        "facility_assignments": 0,
        "overrides_loaded": len(overrides),
    }

    with open("data/raw/FIX.txt", encoding="utf-8") as f:
        for completed_batches, batch in enumerate(
            _iter_batches(f, batch_size=BATCH_SIZE), start=1
        ):
            metrics["lines_seen"] += len(batch)
            raw_fixes = process_batch(batch)
            metrics["batches_processed"] += 1

            for raw_fix in raw_fixes:
                fix_id = str(raw_fix.get("fix_id", "")).upper()
                if not fix_id:
                    continue
                spoken = overrides.get(fix_id, fix_id)
                pronunciation = pronunciation_from_tokens(
                    spoken=spoken,
                    phonemizer=phonemizer,
                )
                fix = build_compact_fix(raw_fix, pronunciation)
                metrics["fix_records"] += 1
                artcc_keys = {
                    str(fix.get("artcc_high", "")),
                    str(fix.get("artcc_low", "")),
                }
                for artcc in sorted(key for key in artcc_keys if key):
                    FIXES[artcc].append(fix)
                    metrics["facility_assignments"] += 1

            if completed_batches % PROGRESS_EVERY_BATCHES == 0:
                print(
                    "Progress "
                    f"(completed_batches={completed_batches}, "
                    f"fix_records={metrics['fix_records']}, "
                    f"facility_assignments={metrics['facility_assignments']}, "
                    f"facilities_seen={len(FIXES)})"
                )

    return metrics


def write_fixes_json(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(dict(FIXES), f, indent=2, sort_keys=True, ensure_ascii=False)


def main() -> None:
    started_at = time.time()
    metrics = load_nav_fixes()
    write_fixes_json(Path("data/fixes.json"))
    elapsed_ms = int((time.time() - started_at) * 1000)
    print("Build complete: data/fixes.json")
    print(f"- lines_seen={metrics['lines_seen']}")
    print(
        f"- batches_processed={metrics['batches_processed']} (batch_size={BATCH_SIZE})"
    )
    print(f"- fix_records={metrics['fix_records']}")
    print(f"- facility_assignments={metrics['facility_assignments']}")
    print(f"- facilities_written={len(FIXES)}")
    print(f"- pronunciation_overrides_loaded={metrics['overrides_loaded']}")
    print(f"- elapsed_ms={elapsed_ms}")


if __name__ == "__main__":
    main()
