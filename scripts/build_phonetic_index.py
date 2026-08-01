"""
CARE-ASR Phonetic FAISS Index Builder (Task T6 scaffold).

This script is a one-time offline pipeline that will:
    1. Load configuration from the config YAML files.
    2. Load a HuBERT model for phonetic speech embedding extraction.
    3. Load the audio dataset to be indexed.
    4. Extract phonetic embeddings for each audio utterance.
    5. Build a FAISS vector index (IndexFlatIP) over the embeddings.
    6. Persist the FAISS index and a position-to-utterance metadata map
       to disk for runtime consumption by src/retrieval/phonetic.py.
    7. Print a final summary of the build.

Configuration loading, HuBERT model loading, audio dataset loading,
phonetic embedding extraction, FAISS index construction, and index
persistence are implemented; retrieval remains a TODO to be completed
during Task T6.

Usage:
    python scripts/build_phonetic_index.py

Run from the repository root (config paths are resolved relative to the
current working directory).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml
from transformers import FeatureExtractionMixin, HubertModel

# The `src` package is not installed (pyproject.toml only packages `care_asr*`),
# so add the repository root to sys.path. This lets the script run directly
# from the project root via `python scripts/build_phonetic_index.py` without
# manual PYTHONPATH configuration.
if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.phonetic import (  # noqa: E402
    HUBERT_CHECKPOINT,
    build_faiss_index,
    extract_embeddings,
    load_audio_dataset,
    load_hubert,
    save_index,
)


@dataclass(frozen=True)
class PhoneticIndexConfig:
    """Aggregated configuration for the phonetic index build pipeline.

    Attributes:
        hubert_checkpoint: HuggingFace model identifier for HuBERT.
        faiss_index_type: FAISS index type (e.g., 'IndexFlatIP').
        faiss_dimension: Embedding dimension for the FAISS index.
        faiss_index_path: Path to write the serialized FAISS index file.
        faiss_metadata_path: Path to write the index metadata JSON.
        extraction_batch_size: Number of audio utterances to embed per batch.
    """

    hubert_checkpoint: str
    faiss_index_type: str
    faiss_dimension: int
    faiss_index_path: Path
    faiss_metadata_path: Path
    extraction_batch_size: int = 32


CONFIG_MODEL_PATH = Path("configs/model.yaml")
CONFIG_RETRIEVAL_PATH = Path("configs/retrieval.yaml")
DEFAULT_FAISS_INDEX_PATH = Path("data/indices/faiss_phonetic.index")
DEFAULT_FAISS_METADATA_PATH = Path("data/indices/utterance_metadata.json")


def _check_path(path: Path) -> None:
    """Raise FileNotFoundError if the given path does not exist."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")


def load_config() -> PhoneticIndexConfig:
    """Load and aggregate configuration for the phonetic index build.

    Reads configs/model.yaml (HuBERT checkpoint) and configs/retrieval.yaml
    (FAISS index type, dimension, output paths, extraction batch size) and
    returns a single PhoneticIndexConfig dataclass with the settings needed
    for building the phonetic FAISS index. Keys that are not yet present in
    the YAML files fall back to project defaults so the pipeline remains
    runnable.

    Returns:
        PhoneticIndexConfig: Aggregated configuration for the index build.

    Raises:
        FileNotFoundError: If any of the config files is missing.
        yaml.YAMLError: If any config file contains invalid YAML.
    """
    for path in (CONFIG_MODEL_PATH, CONFIG_RETRIEVAL_PATH):
        _check_path(path)

    with open(CONFIG_MODEL_PATH, encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    with open(CONFIG_RETRIEVAL_PATH, encoding="utf-8") as f:
        retrieval_cfg = yaml.safe_load(f)

    models_cfg = model_cfg.get("models", {})
    hubert_cfg = models_cfg.get("hubert", {})
    faiss_cfg = retrieval_cfg.get("faiss", {})
    phonetic_cfg = retrieval_cfg.get("phonetic", {})

    hubert_checkpoint = hubert_cfg.get("checkpoint", HUBERT_CHECKPOINT)
    faiss_index_type = faiss_cfg.get("index_type", "IndexFlatIP")
    faiss_dimension = faiss_cfg.get("dimension", 768)
    faiss_index_path = Path(phonetic_cfg.get("index_file", DEFAULT_FAISS_INDEX_PATH))
    faiss_metadata_path = Path(phonetic_cfg.get("metadata_file", DEFAULT_FAISS_METADATA_PATH))
    extraction_batch_size = faiss_cfg.get("encoding_batch_size", 32)

    return PhoneticIndexConfig(
        hubert_checkpoint=hubert_checkpoint,
        faiss_index_type=faiss_index_type,
        faiss_dimension=faiss_dimension,
        faiss_index_path=faiss_index_path,
        faiss_metadata_path=faiss_metadata_path,
        extraction_batch_size=extraction_batch_size,
    )


def load_hubert_model(
    config: PhoneticIndexConfig,
) -> tuple[FeatureExtractionMixin, HubertModel]:
    """Load the HuBERT model and feature extractor for phonetic embedding extraction.

    Delegates to src.retrieval.phonetic.load_hubert(), which downloads the
    facebook/hubert-base-ls960 checkpoint, moves the model to the optimal
    available device (CUDA if available, otherwise CPU), and sets it to
    evaluation mode.

    Args:
        config: Aggregated pipeline configuration containing the
            hubert_checkpoint field.

    Returns:
        Tuple of (feature_extractor, model) where:
            - feature_extractor is the audio feature extractor used to
              preprocess raw waveforms for HuBERT.
            - model is the HuBERT model in evaluation mode.

    Raises:
        RuntimeError: If the HuBERT checkpoint cannot be downloaded or loaded.
    """
    print("Loading HuBERT model...")
    feature_extractor, model = load_hubert()
    print("✓ HuBERT model loaded")
    return feature_extractor, model


def main() -> None:
    """Orchestrate the phonetic index build pipeline through persistence.

    Execution order:
        1. load_config() -> config
        2. load_hubert_model(config) -> (feature_extractor, model)
        3. load_audio_dataset() -> dataset
        4. extract_embeddings(feature_extractor, model, dataset) -> embeddings
        5. build_faiss_index(embeddings) -> index
        6. save_index(index, config.faiss_index_path, config.faiss_metadata_path, dataset)

    The pipeline currently stops after persisting the phonetic FAISS index
    and its metadata; runtime retrieval remains a TODO to be completed
    during Task T6. Progress messages follow the style used by
    scripts/build_semantic_index.py.

    Raises:
        RuntimeError: If any step of the pipeline fails.
    """
    start_time = time.perf_counter()

    try:
        print("Loading configuration...")
        config = load_config()
        print("✓ Configuration loaded")

        feature_extractor, model = load_hubert_model(config)

        print("Loading audio dataset...")
        dataset = load_audio_dataset()
        print(f"✓ Audio dataset loaded: {len(dataset):,} samples")

        print("Loading phonetic embeddings...")
        embeddings = extract_embeddings(
            feature_extractor,
            model,
            dataset,
            batch_size=config.extraction_batch_size,
        )
        print(f"✓ Extracted embeddings: {embeddings.shape}")

        print("Loading phonetic FAISS index...")
        index = build_faiss_index(embeddings)
        print("✓ Phonetic FAISS index built")

        print("Saving phonetic FAISS index...")
        save_index(
            index,
            config.faiss_index_path,
            config.faiss_metadata_path,
            dataset,
        )
        print("✓ Phonetic FAISS index saved")
        print("Saving phonetic metadata...")
        print("✓ Metadata saved")

        # TODO(T6): Implement runtime retrieval over the persisted index.
    except Exception as e:
        raise RuntimeError(f"Phonetic index build pipeline failed: {e}") from e

    elapsed = time.perf_counter() - start_time

    print("=" * 60)
    print("Phonetic Index Build Complete")
    print("=" * 60)
    print(f"  Samples loaded:          {len(dataset):,}")
    print(f"  Embeddings shape:        {embeddings.shape}")
    print(f"  Indexed vectors:         {index.ntotal:,}")
    print(f"  Index output path:       {config.faiss_index_path}")
    print(f"  Metadata output path:    {config.faiss_metadata_path}")
    print(f"  Elapsed time:           {elapsed:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
