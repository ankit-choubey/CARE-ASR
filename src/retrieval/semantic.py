"""
Semantic retrieval query engine.
Searches ClinicalBERT + FAISS index for medical term candidates.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModel, AutoTokenizer

from care_asr.contracts.retrieval_input import RetrievalCandidate


class SemanticRetriever:
    """Queries semantic FAISS index for clinical candidates."""

    def __init__(self, config_path: str = "configs/retrieval.yaml") -> None:
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f).get("faiss", {})
        except Exception:
            cfg = {
                "index_file": "data/indices/faiss_umls.index",
                "mapping_file": "data/indices/cui_mapping.json",
            }

        index_file = cfg.get("index_file", "data/indices/faiss_umls.index")
        mapping_file = cfg.get("mapping_file", "data/indices/cui_mapping.json")
        self.available = False

        if not Path(index_file).exists() or not Path(mapping_file).exists():
            return

        try:
            import faiss

            self.index = faiss.read_index(index_file)
            with open(mapping_file) as f:
                self.mapping: dict = json.load(f)
            self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            self.model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            self.model.eval()
            self.available = True
        except Exception:
            self.available = False

    def _embed(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            out = self.model(**inputs)
        emb = out.last_hidden_state[:, 0, :].numpy()
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        return emb.astype(np.float32)

    def retrieve(self, token: str, top_k: int = 5) -> list[RetrievalCandidate]:
        """Retrieves top_k semantic candidates from FAISS index."""
        if not self.available:
            return []
        try:
            distances, indices = self.index.search(self._embed(token), top_k)
            candidates = []
            for d, idx in zip(distances[0], indices[0], strict=False):
                if idx != -1:
                    cand_name = self.mapping.get(str(idx), "")
                    if cand_name:
                        candidates.append(
                            RetrievalCandidate(
                                candidate=cand_name,
                                score=float(d),
                                source="semantic",
                            )
                        )
            return candidates
        except Exception:
            return []
