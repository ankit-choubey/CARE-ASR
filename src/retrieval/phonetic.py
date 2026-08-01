"""

Phonetic retrieval engine for CARE-ASR (Task T6).

The phonetic retrieval engine recovers medical terms that match the
acoustic/phonetic sound of mistranscribed ASR spans, complementing the
semantic retrieval engine. HuBERT model loading, audio dataset loading,
phonetic embedding extraction, FAISS index construction, and index
persistence are implemented; retrieval remains reserved for a subsequent
T6 subtask.
=======
Phonetic retrieval query engine.
Uses HuBERT FAISS index when available, with automatic Double Metaphone CPU fallback.

from __future__ import annotations

import json
from pathlib import Path

from typing import Any

import faiss
import numpy as np
import torch
from datasets import Dataset, load_dataset
from numpy.typing import NDArray
from tqdm import tqdm
from transformers import AutoFeatureExtractor, FeatureExtractionMixin, HubertModel

HUBERT_CHECKPOINT = "facebook/hubert-base-ls960"
AFRISPEECH_DATASET = "intronhealth/afrispeech-200"


def load_hubert() -> tuple[FeatureExtractionMixin, HubertModel]:
    """Load the HuBERT model and feature extractor for phonetic embedding extraction.

    Downloads the facebook/hubert-base-ls960 checkpoint via the Hugging Face
    Transformers library, moves the model to the optimal available device
    (CUDA if available, otherwise CPU), and sets it to evaluation mode.

    Returns:
        Tuple of (feature_extractor, model) where:
            - feature_extractor is the audio feature extractor used to
              preprocess raw waveforms for HuBERT.
            - model is the HuBERT model in evaluation mode on the selected
              device.

    Raises:
        RuntimeError: If the checkpoint cannot be downloaded or loaded.
    """
    checkpoint = HUBERT_CHECKPOINT
    try:
        feature_extractor: FeatureExtractionMixin = AutoFeatureExtractor.from_pretrained(checkpoint)
        model: HubertModel = HubertModel.from_pretrained(checkpoint)
    except Exception as e:
        raise RuntimeError(f"Failed to load HuBERT model from checkpoint " f"'{checkpoint}': {e}") from e
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return feature_extractor, model


def load_audio_dataset() -> Dataset:
    """Load the AfriSpeech-200 audio corpus for phonetic embedding extraction.

    Downloads the intronhealth/afrispeech-200 dataset via the Hugging Face
    datasets library (all accents, test split) and validates that the loaded
    dataset is non-empty before returning it.

    Returns:
        Dataset: Audio dataset with utterance audio, sample rate, and
        transcript fields.

    Raises:
        RuntimeError: If the dataset cannot be downloaded or loaded, or if
            the loaded dataset contains no samples.
    """
    try:
        dataset = load_dataset(AFRISPEECH_DATASET, "all", split="test")
    except Exception as e:
        raise RuntimeError(f"Failed to load AfriSpeech-200 dataset from " f"'{AFRISPEECH_DATASET}': {e}") from e

    if dataset is None or len(dataset) == 0:
        raise RuntimeError(f"AfriSpeech-200 dataset '{AFRISPEECH_DATASET}' loaded with no samples.")

    return dataset


def extract_embeddings(
    feature_extractor: FeatureExtractionMixin,
    model: HubertModel,
    dataset: Dataset,
    batch_size: int = 32,
) -> NDArray[np.float32]:
    """Extract mean-pooled phonetic embeddings for every audio utterance.

    Iterates over the audio dataset in batches, preprocesses each waveform
    with the feature extractor, runs HuBERT inference under torch.no_grad(),
    and mean-pools the last hidden states across the time dimension to
    produce a single embedding per utterance.

    Args:
        feature_extractor: Audio feature extractor that preprocesses raw
            waveforms for HuBERT.
        model: HuBERT model in evaluation mode on the target device.
        dataset: Audio dataset with an "audio" column per sample containing
            the waveform array and sampling rate.
        batch_size: Number of utterances to process per inference batch.

    Returns:
        NDArray[np.float32]: Float32 array of shape (N, hidden_dimension)
        with one embedding per audio utterance.

    Raises:
        RuntimeError: If embedding extraction fails at any step, if the
            result is empty, or if the produced shape is inconsistent.
    """
    device = next(model.parameters()).device
    all_embeddings: list[NDArray[np.float32]] = []

    try:
        for i in tqdm(
            range(0, len(dataset), batch_size),
            desc="Extracting embeddings",
            unit="batch",
        ):
            batch = dataset[i : i + batch_size]
            waveforms = [sample["audio"]["array"] for sample in batch]
            sampling_rate = batch[0]["audio"]["sampling_rate"]

            inputs = feature_extractor(
                waveforms,
                sampling_rate=sampling_rate,
                padding=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            last_hidden_state = outputs.last_hidden_state  # (B, time, hidden)
            attention_mask = inputs.get("attention_mask")

            if attention_mask is not None:
                # Masked mean pooling so padding frames do not bias the embedding
                mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
                sum_embeddings = (last_hidden_state * mask).sum(dim=1)
                valid_counts = attention_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
                pooled = sum_embeddings / valid_counts
            else:
                pooled = last_hidden_state.mean(dim=1)

            # Move back to CPU before NumPy conversion
            all_embeddings.append(pooled.cpu().numpy().astype(np.float32))
    except Exception as exc:
        raise RuntimeError(f"Failed to extract phonetic embeddings with HuBERT: {exc}") from exc

    if not all_embeddings:
        raise RuntimeError("No phonetic embeddings were extracted from the audio dataset.")

    try:
        embeddings: NDArray[np.float32] = np.concatenate(all_embeddings, axis=0)
    except Exception as exc:
        raise RuntimeError(f"Failed to combine phonetic embeddings: {exc}") from exc

    if embeddings.dtype != np.float32:
        raise RuntimeError(f"Embedding dtype must be float32, got {embeddings.dtype}.")

    if embeddings.ndim != 2:
        raise RuntimeError(f"Embeddings must be 2D (N, D), got shape {embeddings.shape}.")

    if embeddings.shape[0] != len(dataset):
        raise RuntimeError(f"Embedding count mismatch: expected {len(dataset)} rows, " f"got {embeddings.shape[0]}.")

    return embeddings


def build_faiss_index(embeddings: NDArray[np.float32]) -> faiss.Index:
    """Build a FAISS inner-product index over the phonetic embeddings.

    Validates the embedding matrix and constructs an IndexFlatIP populated
    with every embedding vector, then verifies the index contains exactly
    the number of vectors provided.

    Args:
        embeddings: Float32 array of shape (N, embedding_dim) produced by
            extract_embeddings(), with one row per audio utterance.

    Returns:
        faiss.Index: IndexFlatIP FAISS index populated with the embeddings.

    Raises:
        RuntimeError: If the embeddings are invalid, the FAISS index cannot
            be created or populated, or the index size does not match the
            number of embeddings.
    """
    if embeddings.dtype != np.float32:
        raise RuntimeError(f"Embeddings dtype must be float32, got {embeddings.dtype}.")

    if embeddings.ndim != 2:
        raise RuntimeError(f"Embeddings must be a 2D array, got {embeddings.ndim}D.")

    if embeddings.shape[0] == 0:
        raise RuntimeError("Embeddings must contain at least one sample.")

    if embeddings.shape[1] == 0:
        raise RuntimeError("Embeddings must have a non-zero feature dimension.")

    try:
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
    except Exception as exc:
        raise RuntimeError(f"Failed to build FAISS index over {embeddings.shape[0]} " f"embeddings: {exc}") from exc

    if index.ntotal != embeddings.shape[0]:
        raise RuntimeError(
            f"FAISS index population mismatch: index contains "
            f"{index.ntotal} vectors, but {embeddings.shape[0]} embeddings "
            f"were provided."
        )

    return index


def _build_utterance_metadata(dataset: Dataset) -> dict[str, dict[str, Any]]:
    """Build a position-to-utterance metadata map for the FAISS index.

    Maps each sequential FAISS row index back to its source utterance in
    the audio dataset using only the fields available on each sample:
    audio_id, transcript, domain, accent, gender, age_group, country,
    speaker_id, duration, and the audio sampling rate.

    Args:
        dataset: Audio dataset in the same order as the embeddings added
            to the FAISS index.

    Returns:
        dict[str, dict[str, Any]]: Mapping from stringified row positions
        (matching FAISS index order) to utterance metadata records.
    """
    metadata: dict[str, dict[str, Any]] = {}
    for idx, sample in enumerate(dataset):
        audio_info = sample.get("audio", {})
        metadata[str(idx)] = {
            "audio_id": sample.get("audio_id", sample.get("id", f"afrispeech_{idx:05d}")),
            "transcript": sample.get("transcript", sample.get("text", "")),
            "domain": sample.get("domain", ""),
            "accent": sample.get("accent", ""),
            "gender": sample.get("gender", ""),
            "age_group": sample.get("age_group", ""),
            "country": sample.get("country", ""),
            "speaker_id": sample.get("speaker_id", ""),
            "duration": sample.get("duration"),
            "sampling_rate": audio_info.get("sampling_rate", 16000) if isinstance(audio_info, dict) else None,
        }
    return metadata


def save_index(
    index: faiss.Index,
    index_path: Path,
    metadata_path: Path,
    dataset: Dataset,
) -> None:
    """Persist the phonetic FAISS index and its utterance metadata to disk.

    Serializes the built FAISS index with faiss.write_index() and writes a
    JSON metadata map that records, for every index row, enough information
    to trace back to the original dataset utterance. Parent directories are
    created as needed, and both output files are verified to exist after
    writing.

    Args:
        index: FAISS index from build_faiss_index().
        index_path: Output path for the serialized FAISS index file.
        metadata_path: Output path for the JSON utterance metadata file.
        dataset: Audio dataset in the same order as the embeddings added to
            the index; used to build the position-to-utterance metadata.

    Raises:
        RuntimeError: If the index or metadata file cannot be written, or
            if either output file does not exist after writing.
    """
    try:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_path))
    except Exception as e:
        raise RuntimeError(f"Failed to save phonetic FAISS index to '{index_path}': {e}") from e

    if not index_path.exists():
        raise RuntimeError(f"Phonetic FAISS index file was not created at '{index_path}'.")

    try:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(_build_utterance_metadata(dataset), f, indent=2)
    except Exception as e:
        raise RuntimeError(f"Failed to save utterance metadata to '{metadata_path}': {e}") from e

    if not metadata_path.exists():
        raise RuntimeError(f"Utterance metadata file was not created at '{metadata_path}'.")


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

