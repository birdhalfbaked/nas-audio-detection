import pytest
from phonemizer.backend.espeak.espeak import EspeakBackend

from app.audio.phonemes import Phonemizer

pytestmark = pytest.mark.skipif(
    not EspeakBackend.is_available(),
    reason="espeak-ng not installed; phonemizer tests require it",
)


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("students", ["s", "t", "uː", "d", "ə", "n", "t", "s"]),
    ],
)
def test_phonemizer_returns_phonemes_for_phrases(
    phrase: str, expected: list[str] | None
) -> None:
    phonemizer = Phonemizer(model_path="")
    phonemes = phonemizer(phrase)

    assert phonemes
    assert phonemes != phrase

    if expected is not None:
        assert phonemes == expected
