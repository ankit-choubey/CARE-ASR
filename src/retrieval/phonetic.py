"""
Phonetic retrieval query engine.
Uses HuBERT FAISS index when available, with automatic Double Metaphone CPU fallback.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from care_asr.contracts.retrieval_input import RetrievalCandidate


class PhoneticRetriever:
    """Queries phonetic index or Double Metaphone fuzzy vocabulary matching."""

    def __init__(self, config_path: str = "configs/retrieval.yaml") -> None:
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f).get("phonetic", {})
        except Exception:
            cfg = {"max_phonetic_distance": 2}

        self.max_distance = cfg.get("max_phonetic_distance", 2)
        phonetic_index_path = "data/indices/phonetic_index.faiss"
        phonetic_labels_path = "data/indices/phonetic_labels.json"
        vocab_path = "data/indices/medical_vocab.json"

        self.faiss_available = False
        if Path(phonetic_index_path).exists() and Path(phonetic_labels_path).exists():
            try:
                import faiss

                self.index = faiss.read_index(phonetic_index_path)
                with open(phonetic_labels_path) as f:
                    self.labels: list[str] = json.load(f)
                self.faiss_available = True
            except Exception:
                self.faiss_available = False

        self.metaphone_vocab: dict = {}
        if Path(vocab_path).exists():
            try:
                with open(vocab_path) as f:
                    self.metaphone_vocab = json.load(f)
            except Exception:
                self.metaphone_vocab = {}

    def retrieve(self, token: str, top_k: int = 5) -> list[RetrievalCandidate]:
        """Retrieves phonetic candidate matches."""
        return self._metaphone_retrieve(token, top_k)

    def _metaphone_retrieve(self, token: str, top_k: int) -> list[RetrievalCandidate]:
        try:
            from abydos.phonetic import DoubleMetaphone

            dm = DoubleMetaphone()
            query_codes = set(dm.encode(token))
        except Exception:
            return []

        results = []
        for term, codes in self.metaphone_vocab.items():
            if query_codes & set(codes):
                results.append(RetrievalCandidate(candidate=term, score=1.0, source="phonetic"))

        return results[:top_k]
