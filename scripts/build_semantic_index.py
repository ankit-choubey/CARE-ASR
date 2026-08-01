"""
CARE-ASR Semantic FAISS Index Builder (Task T2).

This script is a one-time offline pipeline that:
    1. Loads UMLS/RxNorm medical concept data from a source file or API.
    2. Encodes each concept into 768-dimensional embeddings using
       ClinicalBERT (emilyalsentzer/Bio_ClinicalBERT).
    3. Builds a FAISS vector index (IndexFlatIP) over the embeddings.
    4. Persists the FAISS index and a position-to-concept mapping to disk
       for runtime consumption by src/retrieval/semantic.py.

Usage:
    python scripts/build_semantic_index.py
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import faiss
import numpy as np
import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


class ConceptDict(TypedDict):
    """A single UMLS/RxNorm medical concept with its identifier and name.

    Attributes:
        concept_id: UMLS CUI or RxNorm RxCUI identifier (e.g., 'C0025598').
        concept_name: Preferred clinical concept label (e.g., 'Metformin').
    """

    concept_id: str
    concept_name: str


@dataclass(frozen=True)
class SemanticIndexConfig:
    """Aggregated configuration for the semantic index build pipeline.

    Attributes:
        clinical_bert_checkpoint: HuggingFace model identifier for ClinicalBERT.
        faiss_index_type: FAISS index type (e.g., 'IndexFlatIP').
        faiss_dimension: Embedding dimension for the FAISS index.
        faiss_index_path: Path to write the serialized FAISS index file.
        faiss_mapping_path: Path to write the CUI-to-position mapping JSON.
        encoding_batch_size: Number of concepts to encode per batch.
    """

    clinical_bert_checkpoint: str
    faiss_index_type: str
    faiss_dimension: int
    faiss_index_path: Path
    faiss_mapping_path: Path
    encoding_batch_size: int = 64


CONFIG_MODEL_PATH = Path("configs/model.yaml")
CONFIG_PIPELINE_PATH = Path("configs/pipeline.yaml")
CONFIG_RETRIEVAL_PATH = Path("configs/retrieval.yaml")


def _check_path(path: Path) -> None:
    """Raise FileNotFoundError if the given path does not exist."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")


def load_configs() -> SemanticIndexConfig:
    """Load and aggregate configuration from all three config YAML files.

    Reads configs/model.yaml, configs/pipeline.yaml, and configs/retrieval.yaml
    and returns a single SemanticIndexConfig dataclass with the settings needed
    for building the semantic FAISS index.

    Returns:
        SemanticIndexConfig: Aggregated configuration for the index build.

    Raises:
        FileNotFoundError: If any of the three config files is missing.
        yaml.YAMLError: If any config file contains invalid YAML.
        KeyError: If a required configuration key is missing.
    """
    for path in (CONFIG_MODEL_PATH, CONFIG_PIPELINE_PATH, CONFIG_RETRIEVAL_PATH):
        _check_path(path)

    with open(CONFIG_MODEL_PATH, encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    with open(CONFIG_PIPELINE_PATH, encoding="utf-8") as f:
        yaml.safe_load(f)  # parse to validate YAML is well-formed

    with open(CONFIG_RETRIEVAL_PATH, encoding="utf-8") as f:
        retrieval_cfg = yaml.safe_load(f)

    clinical_bert_checkpoint = model_cfg["models"]["clinical_bert"]["checkpoint"]
    faiss_index_type = retrieval_cfg["faiss"]["index_type"]
    faiss_dimension = retrieval_cfg["faiss"]["dimension"]
    faiss_index_path = Path(retrieval_cfg["faiss"]["index_file"])
    faiss_mapping_path = Path(retrieval_cfg["faiss"]["mapping_file"])
    encoding_batch_size = retrieval_cfg["faiss"]["encoding_batch_size"]

    return SemanticIndexConfig(
        clinical_bert_checkpoint=clinical_bert_checkpoint,
        faiss_index_type=faiss_index_type,
        faiss_dimension=faiss_dimension,
        faiss_index_path=faiss_index_path,
        faiss_mapping_path=faiss_mapping_path,
        encoding_batch_size=encoding_batch_size,
    )


def load_clinical_bert(
    config: SemanticIndexConfig,
) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
    """Load the ClinicalBERT model and tokenizer from HuggingFace.

    Uses the checkpoint specified in the configuration
    (emilyalsentzer/Bio_ClinicalBERT) and returns the model in evaluation mode
    on the optimal available device (CUDA if available, otherwise CPU).

    Args:
        config: Aggregated pipeline configuration containing the
            clinical_bert_checkpoint field.

    Returns:
        Tuple of (tokenizer, model) where:
            - tokenizer is a PreTrainedTokenizer for encoding text.
            - model is a PreTrainedModel set to evaluation mode.

    Raises:
        RuntimeError: If the checkpoint cannot be downloaded or loaded.
    """
    checkpoint = config.clinical_bert_checkpoint
    try:
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model: PreTrainedModel = AutoModel.from_pretrained(checkpoint)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load ClinicalBERT model from checkpoint " f"'{checkpoint}': {e}"
        ) from e
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return tokenizer, model


def load_concepts() -> list[ConceptDict]:
    """Load RxNorm medical concepts from the Hugging Face dataset.

    Fetches the nishanth-augustai/rxnorm_data dataset, filters to English
    (LAT == "ENG") non-suppressed (SUPPRESS == "N") records, and extracts
    only the concept ID and concept name fields.

    Returns:
        list[ConceptDict]: Concept records ordered as they appear in the
        dataset, each containing concept_id (RXCUI as str) and
        concept_name (STR).

    Raises:
        RuntimeError: If the dataset cannot be loaded from Hugging Face.
    """
    try:
        dataset = load_dataset(
            "nishanth-augustai/rxnorm_data",
            split="train",
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load dataset 'nishanth-augustai/rxnorm_data': {exc}"
        ) from exc

    filtered = dataset.filter(lambda row: row["LAT"] == "ENG" and row["SUPPRESS"] == "N")
    return [
        ConceptDict(
            concept_id=str(row["RXCUI"]),
            concept_name=row["STR"],
        )
        for row in filtered
    ]


def encode_concepts(
    config: SemanticIndexConfig,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    concepts: list[ConceptDict],
) -> np.ndarray:
    """Encode concepts into embeddings using ClinicalBERT.

    Processes concepts in batches of config.encoding_batch_size. Each concept
    name is tokenized, passed through the ClinicalBERT model, and converted
    to a fixed-length embedding via masked mean pooling (ignoring padding
    tokens) followed by L2 normalization.

    Args:
        config: Pipeline configuration containing encoding_batch_size
            and faiss_dimension.
        tokenizer: ClinicalBERT tokenizer for text preprocessing.
        model: ClinicalBERT model in evaluation mode.
        concepts: List of concept records to encode.

    Returns:
        np.ndarray: Float32 array of shape (N, config.faiss_dimension)
        where N is the number of concepts, and each row is the
        L2-normalized embedding of the corresponding concept.

    Raises:
        RuntimeError: If encoding fails at any step.
        ValueError: If the embedding dimension does not match
            config.faiss_dimension.
    """
    batch_size = config.encoding_batch_size
    all_embeddings: list[np.ndarray] = []

    try:
        for i in tqdm(
            range(0, len(concepts), batch_size),
            desc="Encoding concepts",
            unit="batch",
        ):
            batch = concepts[i : i + batch_size]
            texts = [concept["concept_name"] for concept in batch]

            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            device = next(model.parameters()).device

            inputs = {key: value.to(device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            last_hidden_state = outputs.last_hidden_state  # (B, seq_len, hidden)
            attention_mask = inputs["attention_mask"]  # (B, seq_len)

            # Expand mask to match hidden state dimensions
            mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())

            # Sum over valid (non-padding) token embeddings
            sum_embeddings = (last_hidden_state * mask).sum(dim=1)

            # Count valid tokens per sequence; clamp to avoid divide-by-zero
            valid_token_counts = attention_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)

            # Masked mean pooling
            mean_pooled = sum_embeddings / valid_token_counts

            # L2 normalize
            normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)

            # Convert to numpy float32 and collect
            all_embeddings.append(normalized.cpu().numpy().astype(np.float32))

    except Exception as exc:
        raise RuntimeError(f"Failed to encode concepts with ClinicalBERT: {exc}") from exc

    embeddings = np.concatenate(all_embeddings, axis=0)

    # Validate embedding dimension
    expected_dim = config.faiss_dimension
    actual_dim = embeddings.shape[1]
    if actual_dim != expected_dim:
        raise ValueError(
            f"Embedding dimension mismatch: expected {expected_dim}, "
            f"got {actual_dim}. Check config.faiss_dimension."
        )

    return embeddings


def build_faiss_index(
    embeddings: np.ndarray,
    config: SemanticIndexConfig,
) -> faiss.Index:
    """Create and train a FAISS vector index from the concept embeddings.

    Supports IndexFlatIP (inner product) as the FAISS index type.
    Validates input embeddings and verifies that all embeddings are
    added to the index successfully.

    Args:
        embeddings: Float32 array of shape (N, config.faiss_dimension)
            from encode_concepts().
        config: Pipeline configuration containing faiss_index_type and
            faiss_dimension.

    Returns:
        faiss.Index: Trained FAISS index populated with concept embeddings.

    Raises:
        ValueError: If the index type is unsupported, or if the embeddings
            array is not 2D float32 with the expected dimension.
        RuntimeError: If the FAISS index creation or population fails, or
            if the number of indexed vectors does not match the input count.
    """
    index_type = config.faiss_index_type
    dimension = config.faiss_dimension

    if index_type != "IndexFlatIP":
        raise ValueError(
            f"Unsupported FAISS index type: '{index_type}'. "
            f"Currently only 'IndexFlatIP' is supported."
        )

    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be a 2D array, got {embeddings.ndim}D.")

    if embeddings.dtype != np.float32:
        raise ValueError(f"Embeddings dtype must be float32, got {embeddings.dtype}.")

    if embeddings.shape[1] != dimension:
        raise ValueError(
            f"Embedding dimension mismatch: expected {dimension}, " f"got {embeddings.shape[1]}."
        )

    try:
        index = faiss.IndexFlatIP(config.faiss_dimension)
        index.add(embeddings)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to build FAISS index (type='IndexFlatIP', "
            f"dim={config.faiss_dimension}): {exc}"
        ) from exc

    if index.ntotal != embeddings.shape[0]:
        raise RuntimeError(
            f"FAISS index population mismatch: index contains "
            f"{index.ntotal} vectors, but {embeddings.shape[0]} "
            f"embeddings were provided."
        )

    return index


def save_index(
    index: faiss.Index,
    config: SemanticIndexConfig,
) -> None:
    """Serialize the trained FAISS index to disk.

    Writes the index binary file to the path configured in retrieval.yaml
    (data/indices/faiss_umls.index). Creates the output directory if it
    does not exist.

    Args:
        index: Trained FAISS index from build_faiss_index().
        config: Pipeline configuration containing faiss_index_path.

    Raises:
        RuntimeError: If the index file cannot be written or if the
            file does not exist after writing.
    """
    path = config.faiss_index_path

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(path))
    except Exception as e:
        raise RuntimeError(f"Failed to save FAISS index to '{path}': {e}") from e

    if not path.exists():
        raise RuntimeError(f"FAISS index file was not created at '{path}'.")


def save_mapping(concepts: list[ConceptDict], config: SemanticIndexConfig) -> None:
    """Save the position-to-concept mapping as a JSON file.

    Writes a JSON mapping where keys are sequential integer positions
    (matching FAISS index order) and values are ConceptDict entries
    containing concept_id and concept_name.

    Args:
        concepts: Concept records in the same order as embeddings passed
            to build_faiss_index().
        config: Pipeline configuration containing faiss_mapping_path.

    Raises:
        RuntimeError: If the mapping file cannot be written or if the
            file does not exist after writing.

    Output path is configured in retrieval.yaml
    (data/indices/cui_mapping.json).
    """
    path = config.faiss_mapping_path

    mapping = {str(idx): concept for idx, concept in enumerate(concepts)}

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2)
    except Exception as e:
        raise RuntimeError(f"Failed to save concept mapping to '{path}': {e}") from e

    if not path.exists():
        raise RuntimeError(f"Concept mapping file was not created at '{path}'.")


def main() -> None:
    """Orchestrate the full semantic index build pipeline.

    Execution order:
        1. load_configs() -> config
        2. load_clinical_bert(config) -> (tokenizer, model)
        3. load_concepts() -> concepts
        4. encode_concepts(config, tokenizer, model, concepts) -> embeddings
        5. build_faiss_index(embeddings, config) -> index
        6. save_index(index, config)
        7. save_mapping(concepts, config)

    A concise summary is printed upon successful completion.

    Raises:
        RuntimeError: If any step of the pipeline fails.
    """
    start_time = time.perf_counter()

    try:
        print("Loading configuration...")
        config = load_configs()
        print("✓ Configuration loaded")

        print("Loading ClinicalBERT model...")
        tokenizer, model = load_clinical_bert(config)
        print("✓ ClinicalBERT model loaded")

        print("Loading medical concepts...")
        concepts = load_concepts()
        print(f"✓ Loaded {len(concepts):,} concepts")

        print("Encoding concept embeddings...")
        embeddings = encode_concepts(
            config,
            tokenizer,
            model,
            concepts,
        )
        print("✓ Concept embeddings encoded")

        print("Building FAISS index...")
        index = build_faiss_index(
            embeddings,
            config,
        )
        print("✓ FAISS index built")

        print("Saving FAISS index...")
        save_index(
            index,
            config,
        )
        print("✓ FAISS index saved")

        print("Saving concept mapping...")
        save_mapping(
            concepts,
            config,
        )
        print("✓ Concept mapping saved")
    except Exception as e:
        raise RuntimeError(f"Semantic index build pipeline failed: {e}") from e

    elapsed = time.perf_counter() - start_time

    print("=" * 60)
    print("Semantic Index Build Complete")
    print("=" * 60)
    print(f"  Concepts loaded:       {len(concepts):,}")
    print(f"  Embedding shape:       {embeddings.shape}")
    print(f"  FAISS index size:      {index.ntotal:,} vectors")
    print(f"  Index output path:     {config.faiss_index_path}")
    print(f"  Mapping output path:   {config.faiss_mapping_path}")
    print(f"  Elapsed time:          {elapsed:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
