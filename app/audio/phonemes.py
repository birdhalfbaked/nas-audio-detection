"""
User phonemizer to take a static word and translate to phonemes
"""

from phonemizer import separator
from phonemizer.backend.espeak.espeak import EspeakBackend


class Phonemizer:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._separator = separator.Separator(phone=" ", word=None)
        # Keep a single backend instance to avoid repeated backend setup/native
        # resource churn on every call.
        self._backend = EspeakBackend(
            language="en-us",
            preserve_punctuation=False,
        )

    def __call__(self, word: str) -> str:
        if not word:
            return []
        phonemized = self._backend.phonemize(
            [word],
            separator=self._separator,
            strip=True,
            njobs=1,
        )
        if not phonemized:
            return []
        return [token for token in phonemized[0].split(" ") if token]
