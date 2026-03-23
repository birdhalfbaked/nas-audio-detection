from app.phraseology import get_common_phraseology_trie, get_facility_phraseology_trie
from app.utils import PhonemeTrie


class Facility:
    def __init__(self, identifier: str):
        self.identifier = identifier
        self.common_phraseology_trie = get_common_phraseology_trie()
        self.facility_trie = self._load_saved_trie()

    def phrases(self) -> list[str]:
        phrases = self.common_phraseology_trie.words()
        if self.facility_trie is not None:
            phrases.extend(self.facility_trie.words())
        return list(dict.fromkeys(phrases))

    def _load_atc_standard_trie(self) -> PhonemeTrie:
        """
        Loads the ATC standard trie that represents common callsigns and phraseology across
        all facilities and controllers
        """
        return self.common_phraseology_trie

    def _load_saved_trie(self) -> PhonemeTrie | None:
        """
        Loads a saved trie that represents identifiers that are relevant to the facility
        """
        return get_facility_phraseology_trie(self.identifier)

    def _build_and_save_trie(self, identifiers: list[str]) -> PhonemeTrie:
        """
        Builds a trie that represents identifiers that are relevant to the facility
        and saves it to a file
        """
        raise NotImplementedError(
            "Facility trie building is not implemented yet. "
            "Use scripts/build_common_tries.py for common trie generation."
        )
