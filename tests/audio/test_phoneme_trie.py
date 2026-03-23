from app.utils import PhonemeTrie


def test_phoneme_trie_uses_phonemizer_for_insert_and_search() -> None:
    calls: list[str] = []
    table = {
        "students": ["s", "t", "uː", "d", "ə", "n", "t", "s"],
    }

    def stub_phonemizer(word: str) -> list[str]:
        calls.append(word)
        return table[word]

    trie = PhonemeTrie(phonemizer=stub_phonemizer)
    trie.insert("students")
    trie.search("students")

    assert calls == ["students", "students"]


def test_phoneme_trie_beam_search_returns_best_candidates() -> None:
    table = {
        "students": ["s", "t", "uː", "d", "ə", "n", "t", "s"],
        "student": ["s", "t", "uː", "d", "ə", "n", "t"],
        "steward": ["s", "t", "j", "uː", "ə", "d"],
        "pilot": ["p", "aɪ", "l", "ə", "t"],
        # Query has one substituted phoneme compared with "students" / "student".
        "studants": ["s", "t", "uː", "d", "æ", "n", "t", "s"],
    }

    def stub_phonemizer(word: str) -> list[str]:
        return table[word]

    trie = PhonemeTrie(phonemizer=stub_phonemizer)
    trie.insert_many(["students", "student", "steward", "pilot"])

    results = trie.search("studants", top_k=2, beam_width=16)

    assert results[0] == "students"
    assert "student" in results
