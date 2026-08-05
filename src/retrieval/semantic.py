"""
Semantic retrieval query engine.
Searches ClinicalBERT + FAISS index for medical term candidates.

T12: Batched FAISS queries with a bounded per-instance embedding cache.
Query embeddings are produced with masked mean pooling to match the pooling
used during index construction (scripts/build_semantic_index.py).
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from numpy.typing import NDArray

from care_asr.contracts.retrieval_input import RetrievalCandidate


class SemanticRetriever:
    """Queries semantic FAISS index for clinical candidates.

    Attributes:
        available (bool): True when the FAISS index, mapping, and model loaded.
        index: Loaded FAISS index (unavailable when ``available`` is False).
        mapping (dict[str, Any]): Position-to-concept mapping; values may be
            plain name strings or ``{"concept_id", "concept_name"}`` dicts as
            written by ``scripts/build_semantic_index.py``.
        tokenizer: ClinicalBERT tokenizer.
        model: ClinicalBERT model in evaluation mode.
        query_batch_size (int): Maximum number of tokens embedded per forward pass.
        _embed_cache (OrderedDict[str, NDArray[np.float32]]): Bounded per-instance
            cache of normalized query-token embeddings.
        _embed_cache_maxsize (int): Maximum number of cached embeddings (0 disables).
    """

    available: bool
    index: Any
    mapping: dict[str, Any]
    tokenizer: Any
    model: Any
    query_batch_size: int
    _embed_cache: OrderedDict[str, NDArray[np.float32]]
    _embed_cache_maxsize: int

    def __init__(self, config_path: str = "configs/retrieval.yaml") -> None:
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f).get("faiss", {})
        except Exception:
            cfg = {
                "index_file": "data/indices/faiss_umls.index",
                "mapping_file": "data/indices/cui_mapping.json",
                "query_batch_size": 32,
                "embedding_cache_maxsize": 1000,
            }

        index_file = cfg.get("index_file", "data/indices/faiss_umls.index")
        mapping_file = cfg.get("mapping_file", "data/indices/cui_mapping.json")
        self.query_batch_size = max(1, int(cfg.get("query_batch_size", 32)))
        self._embed_cache_maxsize = max(0, int(cfg.get("embedding_cache_maxsize", 1000)))
        self._embed_cache: OrderedDict[str, NDArray[np.float32]] = OrderedDict()
        self.available = False

        if not Path(index_file).exists() or not Path(mapping_file).exists():
            return

        try:
            import faiss

            self.index = faiss.read_index(index_file)
            with open(mapping_file) as f:
                self.mapping = json.load(f)
        except Exception:
            self.available = False
            return

        try:
            # Lazy import: the module stays importable even when the transformers
            # install is broken; the model is only needed when an index is present.
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            self.model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            self.model.eval()
            self.available = True
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load ClinicalBERT model from checkpoint 'emilyalsentzer/Bio_ClinicalBERT': {exc}"
            ) from exc

    def _normalize_token(self, token: str) -> str:
        """Normalizes a query token for deterministic cache keying."""
        return " ".join(token.lower().strip().split())

    def _embed_batch(self, texts: list[str]) -> NDArray[np.float32]:
        """Embeds a batch of texts with one tokenizer call and one forward pass.

        Uses masked mean pooling followed by L2 normalization to match the
        pooling used when the FAISS index was built.

        Args:
            texts (list[str]): Texts to embed in a single forward pass.

        Returns:
            NDArray[np.float32]: Float32 embeddings of shape (len(texts), 768).
        """
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        last_hidden_state = outputs.last_hidden_state  # (N, seq_len, hidden)
        attention_mask = inputs["attention_mask"]  # (N, seq_len)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
            sum_embeddings = (last_hidden_state * mask).sum(dim=1)
            valid_counts = attention_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
            mean_pooled = sum_embeddings / valid_counts
        else:
            mean_pooled = last_hidden_state.mean(dim=1)

        normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
        return normalized.cpu().numpy().astype(np.float32)

    def _cache_embedding(self, token: str, embedding: NDArray[np.float32]) -> None:
        """Stores an embedding in the bounded cache, evicting the oldest when full."""
        if self._embed_cache_maxsize <= 0:
            return
        if token in self._embed_cache:
            self._embed_cache.move_to_end(token)
            return
        if len(self._embed_cache) >= self._embed_cache_maxsize:
            self._embed_cache.popitem(last=False)
        self._embed_cache[token] = embedding

    def _candidate_name(self, mapping_value: Any) -> str:
        """Extracts a candidate name from a mapping value.

        Mapping values written by ``scripts/build_semantic_index.py`` are
        ``{"concept_id", "concept_name"}`` dicts; plain name strings are also
        accepted for compatibility with older indexes.

        Args:
            mapping_value (Any): Raw value from the position-to-concept mapping.

        Returns:
            str: The concept name, or empty string when unavailable.
        """
        if isinstance(mapping_value, str):
            return mapping_value
        if isinstance(mapping_value, dict):
            return str(mapping_value.get("concept_name", ""))
        return str(mapping_value)

    def _to_candidates(
        self,
        distances: NDArray[np.float32],
        indices: NDArray[np.int64],
    ) -> list[RetrievalCandidate]:
        """Converts one FAISS search row into RetrievalCandidate objects."""
        candidates: list[RetrievalCandidate] = []
        for distance, index in zip(distances, indices, strict=False):
            if index != -1:
                cand_name = self._candidate_name(self.mapping.get(str(index)))
                if cand_name:
                    candidates.append(
                        RetrievalCandidate(
                            candidate=cand_name,
                            score=float(distance),
                            source="semantic",
                        )
                    )
        return candidates

    def retrieve_many(self, tokens: list[str], top_k: int = 5) -> list[list[RetrievalCandidate]]:
        """Retrieves candidates for many tokens with one batched FAISS search.

        Identical tokens are embedded once (deduplicated before embedding), cached
        embeddings are reused across calls, and results are mapped back to the
        original input order.

        Args:
            tokens (list[str]): Query tokens; duplicates are allowed.
            top_k (int): Maximum number of candidates per token.

        Returns:
            list[list[RetrievalCandidate]]: Candidates per input token, in input order.
        """
        if not tokens:
            return []
        if not self.available:
            return [[] for _ in tokens]
        try:
            normalized = [self._normalize_token(token) for token in tokens]
            unique_tokens = list(dict.fromkeys(normalized))

            missing = [token for token in unique_tokens if token not in self._embed_cache]
            for start in range(0, len(missing), self.query_batch_size):
                chunk = missing[start : start + self.query_batch_size]
                chunk_embeddings = self._embed_batch(chunk)
                for token, embedding in zip(chunk, chunk_embeddings, strict=True):
                    self._cache_embedding(token, embedding)

            matrix = np.vstack([self._embed_cache[token] for token in unique_tokens]).astype(np.float32)
            distances, indices = self.index.search(matrix, top_k)

            results_by_token = {
                token: self._to_candidates(distances[row], indices[row]) for row, token in enumerate(unique_tokens)
            }
            return [results_by_token[token] for token in normalized]
        except Exception:
            return [[] for _ in tokens]

    def retrieve(self, token: str, top_k: int = 5) -> list[RetrievalCandidate]:
        """Retrieves top_k semantic candidates for a single token.

        Delegates to ``retrieve_many``; the public signature and behavior are
        unchanged.

        Args:
            token (str): Query token.
            top_k (int): Maximum number of candidates.

        Returns:
            list[RetrievalCandidate]: Ranked semantic candidates.
        """
        results = self.retrieve_many([token], top_k)
        return results[0] if results else []
