"""BioBERT Named Entity Recognition Extractor.

Why It Exists:
    Executes PyTorch token classification inference using HuggingFace BioBERT,
    extracting clinical entity spans and mapping predictions to official CARE-ASR
    categories (MED, COND, ANA, TTP, PHI).

Teammate Dependencies:
    - Ankit (ASR & Integration Lead): Provides raw transcript text and word alignment inputs.

Imported By:
    - `care_asr.validation.candidate_evaluator`
    - Pipeline integration entrypoints.

TODOs:
    - Implement fallback to CPU execution if CUDA Out-Of-Memory exception occurs.
"""

import logging
import math
import time
from typing import Any

try:
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer
except ImportError:
    torch = None  # type: ignore
    AutoModelForTokenClassification = None  # type: ignore
    AutoTokenizer = None  # type: ignore

from care_asr.config.settings import Settings, get_settings
from care_asr.contracts.asr_input import ASRTranscriptInput
from care_asr.utils.exceptions import InvalidCheckpointError, ModelInferenceError

logger = logging.getLogger(__name__)


class BioBertNERExtractor:
    """Manages BioBERT model initialization, device detection, and token classification.

    This class handles automatic device placement (CUDA vs CPU), evaluation mode configuration,
    and HuggingFace model loading based on settings loaded from config.yaml.

    Args:
        settings (Optional[Settings]): Settings instance. If None, loaded via `get_settings()`.
        tokenizer (Optional[Any]): Pre-initialized HuggingFace tokenizer (for testing/mocking).
        model (Optional[Any]): Pre-initialized HuggingFace model (for testing/mocking).
        auto_load (bool): If True and model/tokenizer are not provided, automatically calls `load_model()`.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        tokenizer: Any | None = None,
        model: Any | None = None,
        auto_load: bool = True,
    ) -> None:
        self.settings: Settings = settings or get_settings()

        yaml_config = self.settings.load_yaml_config()
        model_config = yaml_config.get("model", {})

        self.model_name: str = str(
            model_config.get("name_or_path", self.settings.biobert_model_name_or_path)
        )
        self.device: torch.device = self._detect_device(self.settings.torch_device)
        self.aggregation_strategy: str = str(
            model_config.get("confidence_aggregation_strategy", "mean")
        )
        self.taxonomy_mapping: dict[str, str] = yaml_config.get("taxonomy_mapping", {})

        self.tokenizer: Any | None = tokenizer
        self.model: Any | None = model
        self._unknown_label_warned: bool = False

        logger.info(
            f"Initializing BioBertNERExtractor - Target Model: '{self.model_name}', "
            f"Requested Device: '{self.settings.torch_device}', Resolved Device: '{self.device}'."
        )

        if auto_load and (self.tokenizer is None or self.model is None):
            self.load_model()

    def _detect_device(self, requested_device: str) -> Any:
        if torch is None:
            logger.warning(
                "PyTorch is not installed in the current environment. Defaulting device to 'cpu'."
            )
            return "cpu"

        if requested_device.lower() == "cuda":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"CUDA acceleration detected. GPU: {torch.cuda.get_device_name(0)}")
                return device
            else:
                logger.warning(
                    "CUDA requested in configuration but PyTorch reported CUDA is unavailable. Falling back to CPU."
                )
                return torch.device("cpu")

        return torch.device("cpu")

    def load_model(self) -> None:
        start_time = time.perf_counter()
        logger.info(f"Starting BioBERT model loading pipeline for '{self.model_name}'...")

        if AutoTokenizer is None or AutoModelForTokenClassification is None:
            logger.error(
                "HuggingFace 'transformers' package is missing. Install requirements.txt to load models."
            )
            raise ModelInferenceError("HuggingFace 'transformers' package is not installed.")

        try:
            logger.info(f"Loading tokenizer for '{self.model_name}'...")
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info(f"Tokenizer loaded successfully. Vocabulary size: {len(self.tokenizer)}")

            logger.info(f"Loading AutoModelForTokenClassification for '{self.model_name}'...")
            if self.model is None:
                self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
            logger.info("Model weights loaded successfully.")

            logger.info(f"Moving PyTorch model to compute device '{self.device}'...")
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model successfully moved to '{self.device}' and switched to eval() mode.")

            self._validate_checkpoint_labels()

            elapsed_sec = time.perf_counter() - start_time
            logger.info(
                f"BioBERT initialization complete. "
                f"Model: '{self.model_name}', Device: '{self.device}', Total Time: {elapsed_sec:.3f}s."
            )

        except InvalidCheckpointError:
            raise
        except Exception as e:
            logger.error(
                f"Failed to initialize BioBERT model '{self.model_name}': {e}", exc_info=True
            )
            raise ModelInferenceError(
                f"Failed to load BioBERT tokenizer or model weights for '{self.model_name}': {e}"
            ) from e

    def _validate_checkpoint_labels(self) -> None:
        """Validates that all model id2label values are correctly mapped in the taxonomy config."""
        id2label = getattr(self.model.config, "id2label", {})
        for _label_id, label in id2label.items():
            if not label or label.upper() in ("O", "OUTSIDE", "[CLS]", "[SEP]", "[PAD]"):
                continue

            clean_label = label
            if clean_label.startswith(("B-", "I-", "U-", "L-")):
                clean_label = clean_label[2:]

            clean_label = clean_label.upper()

            if clean_label in ("MED", "COND", "ANA", "TTP", "PHI"):
                continue

            if clean_label not in self.taxonomy_mapping:
                raise InvalidCheckpointError(
                    f"Unsupported label in checkpoint id2label: '{label}' (clean: '{clean_label}')"
                )

    def _map_label_to_category(self, raw_label: str) -> str | None:
        if not raw_label or raw_label.upper() in ("O", "OUTSIDE", "[CLS]", "[SEP]", "[PAD]"):
            return None

        clean_label = raw_label
        if clean_label.startswith(("B-", "I-", "U-", "L-")):
            clean_label = clean_label[2:]

        clean_label = clean_label.upper()

        if clean_label in ("MED", "COND", "ANA", "TTP", "PHI"):
            return clean_label

        category = self.taxonomy_mapping.get(clean_label)
        if category is None:
            if not self._unknown_label_warned:
                logger.warning(
                    f"Unknown label encountered during inference: '{raw_label}'. Skipping."
                )
                self._unknown_label_warned = True
            return None

        return category

    def extract_entities(self, asr_input: ASRTranscriptInput) -> list[dict[str, Any]]:
        if self.model is None or self.tokenizer is None:
            logger.error(
                "Attempted to run extract_entities() without initializing model or tokenizer."
            )
            raise ModelInferenceError(
                "BioBERT model and tokenizer must be loaded before running inference."
            )

        raw_transcript = asr_input.raw_transcript
        if not raw_transcript or not raw_transcript.strip():
            logger.info(
                f"Empty transcript received for ID '{asr_input.transcript_id}'. Returning empty entity list."
            )
            return []

        start_time = time.perf_counter()
        logger.info(
            f"Starting NER extraction for transcript ID '{asr_input.transcript_id}' "
            f"({len(raw_transcript)} chars, {len(asr_input.words)} words)..."
        )

        try:
            # 1. Tokenize Transcript with Sliding Window Overlap
            encoding = self._tokenize_text(raw_transcript)
            input_ids_tensor = encoding["input_ids"].to(self.device)
            attention_mask_tensor = encoding["attention_mask"].to(self.device)
            offset_mapping_tensor = encoding["offset_mapping"]

            # 2. Execute PyTorch Inference
            pred_ids, conf_scores = self._run_model(input_ids_tensor, attention_mask_tensor)

            id2label = getattr(self.model.config, "id2label", {})
            all_extracted_entities = []

            # Process each sliding window chunk
            num_chunks = input_ids_tensor.size(0)
            for chunk_idx in range(num_chunks):
                chunk_input_ids = input_ids_tensor[chunk_idx].cpu().tolist()
                chunk_pred_ids = pred_ids[chunk_idx].cpu().tolist()
                chunk_conf_scores = conf_scores[chunk_idx].cpu().tolist()
                chunk_offsets = offset_mapping_tensor[chunk_idx].tolist()

                chunk_entities = self._decode_predictions(
                    chunk_input_ids,
                    chunk_pred_ids,
                    chunk_conf_scores,
                    chunk_offsets,
                    id2label,
                    raw_transcript,
                )
                all_extracted_entities.extend(chunk_entities)

            # 3. Merge entities across overlapping chunks
            final_entities = self._merge_entities(all_extracted_entities)

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.info(
                f"NER extraction complete for ID '{asr_input.transcript_id}' in {elapsed_ms:.2f}ms. "
                f"Extracted {len(final_entities)} entities."
            )

            return final_entities

        except Exception as e:
            logger.error(
                f"Inference error during NER extraction for transcript ID '{asr_input.transcript_id}': {e}",
                exc_info=True,
            )
            raise ModelInferenceError(
                f"Failed to execute BioBERT NER extraction for transcript ID '{asr_input.transcript_id}': {e}"
            ) from e

    def _tokenize_text(self, raw_transcript: str) -> dict[str, Any]:
        logger.debug("Tokenizing transcript with sliding-window chunking...")
        return self.tokenizer(
            raw_transcript,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            stride=128,  # Overlap window
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
        )

    def _run_model(self, input_ids: Any, attention_mask: Any) -> tuple[Any, Any]:
        logger.debug("Executing BioBERT PyTorch model forward pass...")
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # Shape: (num_chunks, sequence_length, num_labels)
            probabilities = torch.softmax(logits, dim=-1)
            confidences, predictions = torch.max(probabilities, dim=-1)
        return predictions, confidences

    def _decode_predictions(
        self,
        chunk_input_ids: list[int],
        chunk_pred_ids: list[int],
        chunk_conf_scores: list[float],
        chunk_offsets: list[tuple[int, int]],
        id2label: dict,
        raw_transcript: str,
    ) -> list[dict[str, Any]]:
        extracted_entities: list[dict[str, Any]] = []
        current_entity: dict[str, Any] | None = None

        for idx, (_token_id, pred_id, conf_score, offsets) in enumerate(
            zip(chunk_input_ids, chunk_pred_ids, chunk_conf_scores, chunk_offsets, strict=False)
        ):
            start_char, end_char = offsets

            # Ignore special tokens with zero offset range (e.g. [CLS], [SEP], [PAD])
            if start_char == 0 and end_char == 0:
                if current_entity:
                    self._finalize_entity(current_entity, raw_transcript, extracted_entities)
                    current_entity = None
                continue

            raw_label = id2label.get(pred_id, "O") if isinstance(id2label, dict) else "O"
            category = self._map_label_to_category(raw_label)

            if category is None:
                # Outside token ("O")
                if current_entity:
                    self._finalize_entity(current_entity, raw_transcript, extracted_entities)
                    current_entity = None
                continue

            is_b_tag = raw_label.startswith("B-") or raw_label.startswith("U-")

            # Decision logic for merging or starting new entity span
            if current_entity is None:
                current_entity = self._create_entity_dict(
                    raw_label, category, start_char, end_char, idx, conf_score
                )
            elif (
                is_b_tag
                or category != current_entity["category"]
                or start_char > current_entity["end_char"] + 2
            ):
                # New entity boundary detected
                self._finalize_entity(current_entity, raw_transcript, extracted_entities)
                current_entity = self._create_entity_dict(
                    raw_label, category, start_char, end_char, idx, conf_score
                )
            else:
                # Continuation of current entity span (I-tag or subword token)
                current_entity["end_char"] = max(current_entity["end_char"], end_char)
                current_entity["token_indices"].append(idx)
                current_entity["confidences"].append(conf_score)

        # Finalize trailing entity if present
        if current_entity:
            self._finalize_entity(current_entity, raw_transcript, extracted_entities)

        return extracted_entities

    def _create_entity_dict(
        self,
        raw_label: str,
        category: str,
        start_char: int,
        end_char: int,
        token_idx: int,
        conf_score: float,
    ) -> dict[str, Any]:
        return {
            "model_label": raw_label,
            "category": category,
            "start_char": start_char,
            "end_char": end_char,
            "token_indices": [token_idx],
            "confidences": [conf_score],
        }

    def _compute_entity_confidence(self, confidences: list[float]) -> float:
        if not confidences:
            return 0.0

        if self.aggregation_strategy == "min":
            return min(confidences)
        elif self.aggregation_strategy == "geom_mean":
            # Geometric mean using log to avoid underflow
            return math.exp(sum(math.log(max(c, 1e-9)) for c in confidences) / len(confidences))
        else:  # default to mean
            return sum(confidences) / len(confidences)

    def _finalize_entity(
        self, entity_dict: dict[str, Any], raw_transcript: str, target_list: list[dict[str, Any]]
    ) -> None:
        start = entity_dict["start_char"]
        end = entity_dict["end_char"]
        entity_text = raw_transcript[start:end].strip()

        if entity_text:
            confidences = entity_dict.pop("confidences")
            agg_conf = self._compute_entity_confidence(confidences)

            entity_dict["entity_text"] = entity_text
            entity_dict["confidence"] = round(agg_conf, 4)
            target_list.append(entity_dict)

    def _merge_entities(self, all_entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Deduplicates entities extracted across multiple sliding window chunks based on span match."""
        merged: dict[tuple[int, int], dict[str, Any]] = {}
        for ent in all_entities:
            key = (ent["start_char"], ent["end_char"])

            # If span is seen again, keep the one with higher confidence
            if key not in merged or ent["confidence"] > merged[key]["confidence"]:
                merged[key] = ent

        # Return sorted by character offsets
        return sorted(merged.values(), key=lambda x: x["start_char"])
