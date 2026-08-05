"""Unit tests for SemanticRetriever batching and embedding caching (T12)."""

from collections import OrderedDict
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from care_asr.contracts.retrieval_input import RetrievalCandidate
from src.retrieval.phonetic import PhoneticRetriever
from src.retrieval.semantic import SemanticRetriever

# Token texts whose ord-sums produce distinct FakeTokenizer codes:
# "d" -> 100 % 10 = 0, "e" -> 101 % 10 = 1, "f" -> 102 % 10 = 2
TOKEN_CODE_ZERO = "d"
TOKEN_CODE_ONE = "e"
TOKEN_CODE_TWO = "f"


class FakeTokenizer:
    """Minimal tokenizer that encodes each text with a deterministic code."""

    def __call__(self, texts: Any, **kwargs: Any) -> dict[str, Any]:
        batch = len(texts) if isinstance(texts, list) else 1
        seq_len = 8
        input_ids = torch.zeros((batch, seq_len), dtype=torch.long)
        for row in range(batch):
            text = texts[row] if isinstance(texts, list) else texts
            input_ids[row, 0] = sum(ord(char) for char in text) % 10
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones((batch, seq_len), dtype=torch.long),
        }


class FakeModel:
    """Minimal model that counts forward passes and pools token codes."""

    def __init__(self) -> None:
        self.forward_calls = 0

    def __call__(self, **inputs: Any) -> SimpleNamespace:
        self.forward_calls += 1
        input_ids = inputs["input_ids"]
        batch, seq_len, dim = input_ids.shape[0], input_ids.shape[1], 768
        hidden = torch.zeros((batch, seq_len, dim))
        hidden[:, :, 0] = input_ids[:, 0:1].float()
        hidden[:, :, 1] = torch.arange(seq_len).unsqueeze(0).float()
        return SimpleNamespace(last_hidden_state=hidden)


class FakeIndex:
    """Minimal FAISS index mapping embeddings to mapping keys 0/1."""

    def __init__(self) -> None:
        self.search_calls = 0
        self.last_top_k = 0

    def search(self, embeddings: Any, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        self.search_calls += 1
        self.last_top_k = top_k
        n = embeddings.shape[0]
        distances = np.zeros((n, top_k), dtype=np.float32)
        indices = np.zeros((n, top_k), dtype=np.int64)
        for row in range(n):
            # Pooled channel 0 holds the token code (L2-normalized); code 0 maps
            # to key 0, codes 1+ to key 1.
            position = 1 if float(embeddings[row, 0]) > 0.1 else 0
            indices[row] = (np.arange(top_k) + position) % 2
        return distances, indices


@pytest.fixture
def retriever() -> SemanticRetriever:
    """Builds an available SemanticRetriever around fakes, bypassing heavy __init__."""
    retriever = SemanticRetriever.__new__(SemanticRetriever)
    retriever.available = True
    retriever.tokenizer = FakeTokenizer()
    retriever.model = FakeModel()
    retriever.index = FakeIndex()
    retriever.mapping = {"0": "metformin", "1": "diabetes"}
    retriever.query_batch_size = 2
    retriever._embed_cache = OrderedDict()
    retriever._embed_cache_maxsize = 1000
    return retriever


def names(candidates: list[RetrievalCandidate]) -> list[str]:
    """Extracts candidate names for assertions."""
    return [candidate.candidate for candidate in candidates]


def test_retrieve_many_matches_sequential_retrieve(retriever: SemanticRetriever) -> None:
    """Batched retrieval returns the same candidates as per-token retrieve()."""
    tokens = [TOKEN_CODE_ZERO, TOKEN_CODE_ONE]
    batched = retriever.retrieve_many(tokens, top_k=3)
    sequential = [retriever.retrieve(token, top_k=3) for token in tokens]
    assert [names(results) for results in batched] == [names(results) for results in sequential]


def test_duplicate_tokens_embed_once(retriever: SemanticRetriever) -> None:
    """Identical tokens are deduplicated before embedding."""
    retriever.retrieve_many([TOKEN_CODE_ZERO, TOKEN_CODE_ZERO, TOKEN_CODE_ONE], top_k=3)
    assert retriever.model.forward_calls == 1


def test_cache_hit_avoids_recomputation(retriever: SemanticRetriever) -> None:
    """Repeated queries reuse cached embeddings instead of re-running the model."""
    retriever.retrieve_many([TOKEN_CODE_ZERO], top_k=3)
    retriever.retrieve_many([TOKEN_CODE_ZERO], top_k=3)
    assert retriever.model.forward_calls == 1

    retriever.retrieve_many([TOKEN_CODE_ZERO, TOKEN_CODE_ONE], top_k=3)
    assert retriever.model.forward_calls == 2


def test_empty_input_returns_empty_list(retriever: SemanticRetriever) -> None:
    """An empty token list yields no results and no model calls."""
    assert retriever.retrieve_many([]) == []
    assert retriever.model.forward_calls == 0


def test_ordering_preserved(retriever: SemanticRetriever) -> None:
    """Results are mapped back to the original input order."""
    results = retriever.retrieve_many([TOKEN_CODE_ONE, TOKEN_CODE_ZERO, TOKEN_CODE_ONE], top_k=2)
    assert names(results[0]) == names(results[2])
    assert names(results[0]) != names(results[1])
    assert names(results[1]) != []


def test_top_k_respected(retriever: SemanticRetriever) -> None:
    """The FAISS search receives top_k and returns at most top_k candidates."""
    results = retriever.retrieve_many([TOKEN_CODE_ZERO, TOKEN_CODE_ONE], top_k=4)
    assert retriever.index.last_top_k == 4
    assert all(len(candidates) <= 4 for candidates in results)
    assert len(results[0]) == 4


def test_retrieve_backward_compatible(retriever: SemanticRetriever) -> None:
    """retrieve() keeps its signature and delegates to retrieve_many."""
    single = retriever.retrieve(TOKEN_CODE_ZERO, top_k=3)
    assert isinstance(single, list)
    assert all(isinstance(candidate, RetrievalCandidate) for candidate in single)
    assert names(single) == names(retriever.retrieve_many([TOKEN_CODE_ZERO], top_k=3)[0])


def test_retrieve_unavailable_returns_empty() -> None:
    """Unavailable retrievers return empty results without embedding."""
    retriever = SemanticRetriever.__new__(SemanticRetriever)
    retriever.available = False
    assert retriever.retrieve(TOKEN_CODE_ZERO) == []
    assert retriever.retrieve_many([TOKEN_CODE_ZERO, TOKEN_CODE_ONE]) == [[], []]


def test_mapping_dict_values_resolve_concept_name(retriever: SemanticRetriever) -> None:
    """Mapping values written as {concept_id, concept_name} dicts resolve to names."""
    retriever.mapping = {
        "0": {"concept_id": "3", "concept_name": "metformin"},
        "1": "diabetes",
    }
    results = retriever.retrieve_many([TOKEN_CODE_ZERO], top_k=1)
    assert names(results[0]) == ["metformin"]


def test_cache_is_bounded(retriever: SemanticRetriever) -> None:
    """The cache evicts the oldest entry when it reaches its max size."""
    retriever._embed_cache_maxsize = 2
    retriever.query_batch_size = 10
    retriever.retrieve_many([TOKEN_CODE_ZERO, TOKEN_CODE_ONE, TOKEN_CODE_TWO], top_k=2)
    assert len(retriever._embed_cache) == 2
    assert TOKEN_CODE_ZERO not in retriever._embed_cache

    retriever.retrieve_many([TOKEN_CODE_ZERO], top_k=2)
    assert retriever.model.forward_calls == 2


class CountingDoubleMetaphone:
    """Fake Double Metaphone that records every encode() call."""

    def __init__(self, code_map: dict[str, tuple[str, ...]]) -> None:
        self.code_map = code_map
        self.encode_calls: list[str] = []

    def encode(self, token: str) -> tuple[str, ...]:
        self.encode_calls.append(token)
        return self.code_map.get(token, ("", ""))


@pytest.fixture
def phonetic_retriever() -> PhoneticRetriever:
    """Builds a PhoneticRetriever around a counting Double Metaphone and vocab."""
    retriever = PhoneticRetriever.__new__(PhoneticRetriever)
    retriever.max_distance = 2
    retriever.faiss_available = False
    retriever._encoding_cache = OrderedDict()
    retriever._encoding_cache_maxsize = 1000
    retriever._dm = CountingDoubleMetaphone(
        {
            "aspirin": ("ASPRN", ""),
            "ibuprofen": ("APRFN", "APRFR"),
            "metformin": ("MTFRM", "MTFRN"),
        }
    )
    retriever.metaphone_vocab = {
        "aspirin": ["ASPRN", ""],
        "asparin": ["ASPRN", ""],
        "asperin": ["ASPRN", ""],
        "ibuprofen": ["APRFN", "APRFR"],
        "metformin": ["MTFRM", "MTFRN"],
    }
    return retriever


def test_phonetic_retrieve_many_matches_sequential(phonetic_retriever: PhoneticRetriever) -> None:
    """Batched retrieval returns the same candidates as per-token retrieve()."""
    tokens = ["aspirin", "ibuprofen"]
    batched = phonetic_retriever.retrieve_many(tokens, top_k=3)
    sequential = [phonetic_retriever.retrieve(token, top_k=3) for token in tokens]
    assert [names(results) for results in batched] == [names(results) for results in sequential]
    assert all(candidate.source == "phonetic" for result in batched for candidate in result)


def test_phonetic_duplicate_tokens_encode_once(phonetic_retriever: PhoneticRetriever) -> None:
    """Identical tokens are encoded exactly once."""
    phonetic_retriever.retrieve_many(["aspirin", "ibuprofen", "aspirin", "aspirin"], top_k=3)
    assert phonetic_retriever._dm.encode_calls == ["aspirin", "ibuprofen"]


def test_phonetic_cache_hit_avoids_recomputation(phonetic_retriever: PhoneticRetriever) -> None:
    """Repeated queries reuse cached encodings instead of re-encoding."""
    phonetic_retriever.retrieve_many(["aspirin"], top_k=3)
    phonetic_retriever.retrieve_many(["aspirin"], top_k=3)
    phonetic_retriever.retrieve("aspirin", top_k=3)
    assert phonetic_retriever._dm.encode_calls == ["aspirin"]


def test_phonetic_empty_input_returns_empty_list(phonetic_retriever: PhoneticRetriever) -> None:
    """An empty token list yields no results and no encodings."""
    assert phonetic_retriever.retrieve_many([]) == []
    assert phonetic_retriever._dm.encode_calls == []


def test_phonetic_ordering_preserved(phonetic_retriever: PhoneticRetriever) -> None:
    """Results are mapped back to the original input order."""
    results = phonetic_retriever.retrieve_many(["ibuprofen", "aspirin", "ibuprofen"], top_k=3)
    assert names(results[0]) == ["ibuprofen"]
    assert names(results[1]) == ["aspirin", "asparin", "asperin"]
    assert names(results[2]) == names(results[0])


def test_phonetic_duplicate_outputs_preserved(phonetic_retriever: PhoneticRetriever) -> None:
    """Duplicate tokens yield duplicate, independent result lists."""
    results = phonetic_retriever.retrieve_many(["aspirin", "aspirin"], top_k=3)
    assert len(results) == 2
    assert names(results[0]) == names(results[1]) == ["aspirin", "asparin", "asperin"]
    assert results[0] is not results[1]


def test_phonetic_top_k_respected(phonetic_retriever: PhoneticRetriever) -> None:
    """Each token returns at most top_k candidates."""
    limited = phonetic_retriever.retrieve_many(["aspirin"], top_k=2)
    assert len(limited[0]) == 2
    full = phonetic_retriever.retrieve_many(["aspirin"], top_k=10)
    assert len(full[0]) == 3


def test_phonetic_retrieve_backward_compatible(phonetic_retriever: PhoneticRetriever) -> None:
    """retrieve() keeps its signature and delegates to retrieve_many."""
    single = phonetic_retriever.retrieve("aspirin", top_k=3)
    assert isinstance(single, list)
    assert all(isinstance(candidate, RetrievalCandidate) for candidate in single)
    assert names(single) == names(phonetic_retriever.retrieve_many(["aspirin"], top_k=3)[0])


def test_phonetic_cache_eviction(phonetic_retriever: PhoneticRetriever) -> None:
    """The oldest cache entry is evicted when the cache reaches its max size."""
    phonetic_retriever._encoding_cache_maxsize = 2
    phonetic_retriever.retrieve_many(["aspirin", "ibuprofen", "metformin"], top_k=3)
    assert len(phonetic_retriever._encoding_cache) == 2
    assert "aspirin" not in phonetic_retriever._encoding_cache
    assert "ibuprofen" in phonetic_retriever._encoding_cache
    assert "metformin" in phonetic_retriever._encoding_cache


def test_phonetic_cache_size_respected(phonetic_retriever: PhoneticRetriever) -> None:
    """The cache never exceeds its configured max size."""
    phonetic_retriever._encoding_cache_maxsize = 1
    phonetic_retriever.retrieve_many(["aspirin", "ibuprofen"], top_k=3)
    assert len(phonetic_retriever._encoding_cache) == 1
    assert "ibuprofen" in phonetic_retriever._encoding_cache


def test_phonetic_config_driven_cache_size(tmp_path) -> None:
    """The encoding cache size is read from the phonetic config block."""
    config = tmp_path / "retrieval.yaml"
    config.write_text("phonetic:\n  encoding_cache_maxsize: 3\n  max_phonetic_distance: 2\n")
    retriever = PhoneticRetriever(config_path=str(config))
    assert retriever._encoding_cache_maxsize == 3
    assert retriever.max_distance == 2


def test_phonetic_unavailable_returns_empty() -> None:
    """Retrievers without Double Metaphone return empty results without encoding."""
    retriever = PhoneticRetriever.__new__(PhoneticRetriever)
    retriever._dm = None
    retriever.metaphone_vocab = {"aspirin": ["ASPRN", ""]}
    assert retriever.retrieve("aspirin") == []
    assert retriever.retrieve_many(["aspirin", "metformin"]) == [[], []]
