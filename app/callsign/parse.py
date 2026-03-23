"""
Parse callsigns and build spoken phrase variants for trie insertion.

We cannot know if flight numbers will be read digit-by-digit or as grouped pairs
(e.g. "2323" as "two three two three" vs "twenty-three twenty-three"), so we
generate both phrasings for the numeric part and combine with the airline name.
"""

from __future__ import annotations

import re

from app.callsign.airlines import ICAO_AIRLINE_NAME

_DIGIT_WORD = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}

_TEENS = [
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_TENS = [
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
]
_ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def _two_digit_to_words(two: str) -> str:
    if len(two) != 2 or not two.isdigit():
        raise ValueError(f"Expected two digits, got {two!r}")
    n = int(two)
    if n < 10:
        return _ONES[n]
    if n < 20:
        return _TEENS[n - 10]
    tens, ones = n // 10, n % 10
    if ones == 0:
        return _TENS[tens]
    return f"{_TENS[tens]} {_ONES[ones]}"


def _digits_by_digit_words(digits: str) -> str:
    return " ".join(_DIGIT_WORD[d] for d in digits if d in _DIGIT_WORD)


def _digits_by_pairs_words(digits: str) -> str | None:
    """If length is even, read as AB CD ... (e.g. 2323 -> twenty-three twenty-three)."""
    if len(digits) < 2 or len(digits) % 2 != 0:
        return None
    pairs = [digits[i : i + 2] for i in range(0, len(digits), 2)]
    return " ".join(_two_digit_to_words(p) for p in pairs)


def number_spoken_variants(digits: str) -> list[str]:
    """Return distinct phrasings for the numeric part only (no airline name)."""
    if not digits.isdigit():
        return []
    variants: list[str] = []
    by_digit = _digits_by_digit_words(digits)
    if by_digit:
        variants.append(by_digit)
    paired = _digits_by_pairs_words(digits)
    if paired and paired != by_digit:
        variants.append(paired)
    return list(dict.fromkeys(variants))


def split_callsign_airline_and_digits(callsign: str) -> tuple[str, str] | None:
    """
    Parse a callsign into (airline_name, digit_sequence).

    Accepts:
    - ICAO+digits: `DAL2323` -> (Delta, 2323) using ICAO_AIRLINE_NAME
    - Spoken-style: `Delta 2323` -> (Delta, 2323)
    """
    raw = callsign.strip()
    if not raw:
        return None

    # e.g. Delta 2323
    space_match = re.match(r"^([A-Za-z]+)\s+(\d+)$", raw)
    if space_match:
        airline_name, digits = space_match.group(1), space_match.group(2)
        return airline_name, digits

    compact = raw.replace(" ", "").upper()
    m = re.match(r"^([A-Z]{3})(\d+)$", compact)
    if m:
        icao, digits = m.group(1), m.group(2)
        airline = ICAO_AIRLINE_NAME.get(icao, icao.title())
        return airline, digits

    return None


def spoken_phrase_variants_for_callsign(callsign: str) -> list[tuple[str, str]]:
    """
    Return list of (canonical_label, spoken_phrase) for trie insertion.

    canonical_label is stable, e.g. "Delta 2323".
    spoken_phrase is what we pass to the phonemizer (e.g. "Delta two three two three").
    """
    parsed = split_callsign_airline_and_digits(callsign)
    if parsed is None:
        return []
    airline_name, digits = parsed
    canonical = f"{airline_name} {digits}"
    variants: list[tuple[str, str]] = []
    for num_spoken in number_spoken_variants(digits):
        phrase = f"{airline_name} {num_spoken}"
        variants.append((canonical, phrase))
    return list(dict.fromkeys(variants))