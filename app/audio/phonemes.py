"""
User phonemizer to take a static word and translate to phonemes
"""

from functools import partial

from phonemizer import phonemize, separator


class Phonemizer:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.phonemize = partial(
            phonemize,
            backend="espeak",
            language="en-us",
            separator=separator.Separator(phone=" ", word=None),
        )

    def __call__(self, word: str) -> str:
        return self.phonemize(word).strip().split(" ")
