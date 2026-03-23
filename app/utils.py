from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


PhonemizerCallable = Callable[[str], list[str]]


@dataclass
class _TrieNode:
    children: dict[str, "_TrieNode"] = field(default_factory=dict)
    terminal_words: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class _BeamState:
    node: _TrieNode
    query_idx: int
    cost: int


class PhonemeTrie:
    def __init__(self, phonemizer: PhonemizerCallable):
        self._phonemizer = phonemizer
        self._root = _TrieNode()
        self._max_word_phoneme_len = 0

    def insert(self, word: str) -> None:
        phonemes = self._phonemizer(word)
        self.insert_phoneme_path(phonemes, word)

    def insert_phoneme_path(self, phonemes: list[str], terminal_label: str) -> None:
        """Walk trie using explicit phonemes; store ``terminal_label`` at the leaf."""
        node = self._root
        for phoneme in phonemes:
            node = node.children.setdefault(phoneme, _TrieNode())
        node.terminal_words.add(terminal_label)
        self._max_word_phoneme_len = max(self._max_word_phoneme_len, len(phonemes))

    def insert_many(self, words: list[str]) -> None:
        for word in words:
            self.insert(word)

    def words(self) -> list[str]:
        return sorted(self._collect_words(self._root))

    def to_dict(self) -> dict:
        return {
            "max_word_phoneme_len": self._max_word_phoneme_len,
            "root": self._node_to_dict(self._root),
        }

    @classmethod
    def from_dict(cls, phonemizer: PhonemizerCallable, data: dict) -> "PhonemeTrie":
        trie = cls(phonemizer=phonemizer)
        trie._max_word_phoneme_len = int(data.get("max_word_phoneme_len", 0))
        trie._root = cls._node_from_dict(data.get("root", {}))
        return trie

    def search(self, word: str, top_k: int = 3, beam_width: int | None = None) -> list[str]:
        query_phonemes = self._phonemizer(word)
        return self.search_phonemes(
            query_phonemes, top_k=top_k, beam_width=beam_width
        )

    def search_phonemes(
        self,
        query_phonemes: list[str],
        top_k: int = 3,
        beam_width: int | None = None,
    ) -> list[str]:
        if top_k <= 0:
            return []

        beam = [_BeamState(node=self._root, query_idx=0, cost=0)]
        candidates: dict[str, int] = {}
        width = beam_width if beam_width is not None else max(20, top_k * 5)

        max_steps = len(query_phonemes) + self._max_word_phoneme_len + 2
        for _ in range(max_steps):
            next_states: dict[tuple[int, int], _BeamState] = {}

            for state in beam:
                if state.query_idx == len(query_phonemes) and state.node.terminal_words:
                    for candidate in state.node.terminal_words:
                        prev = candidates.get(candidate)
                        if prev is None or state.cost < prev:
                            candidates[candidate] = state.cost

                # Delete a query phoneme (skip noisy/mismatched query token).
                if state.query_idx < len(query_phonemes):
                    self._add_best_state(
                        next_states,
                        _BeamState(
                            node=state.node,
                            query_idx=state.query_idx + 1,
                            cost=state.cost + 1,
                        ),
                    )

                for phoneme, child in state.node.children.items():
                    # Insert trie phoneme (candidate has extra token compared to query).
                    self._add_best_state(
                        next_states,
                        _BeamState(
                            node=child,
                            query_idx=state.query_idx,
                            cost=state.cost + 1,
                        ),
                    )

                    # Match/substitute when both query and trie consume a token.
                    if state.query_idx < len(query_phonemes):
                        mismatch_penalty = 0 if query_phonemes[state.query_idx] == phoneme else 1
                        self._add_best_state(
                            next_states,
                            _BeamState(
                                node=child,
                                query_idx=state.query_idx + 1,
                                cost=state.cost + mismatch_penalty,
                            ),
                        )

            if not next_states:
                break

            beam = sorted(next_states.values(), key=lambda s: s.cost)[:width]

        ranked_words = sorted(candidates.items(), key=lambda item: (item[1], item[0]))
        return [word for word, _ in ranked_words[:top_k]]

    @staticmethod
    def _add_best_state(
        state_map: dict[tuple[int, int], _BeamState], state: _BeamState
    ) -> None:
        key = (id(state.node), state.query_idx)
        existing = state_map.get(key)
        if existing is None or state.cost < existing.cost:
            state_map[key] = state

    @staticmethod
    def _node_to_dict(node: _TrieNode) -> dict:
        return {
            "terminal_words": sorted(node.terminal_words),
            "children": {
                phoneme: PhonemeTrie._node_to_dict(child)
                for phoneme, child in node.children.items()
            },
        }

    @staticmethod
    def _node_from_dict(data: dict) -> _TrieNode:
        node = _TrieNode()
        node.terminal_words = set(data.get("terminal_words", []))
        node.children = {
            phoneme: PhonemeTrie._node_from_dict(child_data)
            for phoneme, child_data in data.get("children", {}).items()
        }
        return node

    @staticmethod
    def _collect_words(node: _TrieNode) -> set[str]:
        words = set(node.terminal_words)
        for child in node.children.values():
            words.update(PhonemeTrie._collect_words(child))
        return words
