# CARE-ASR API Reference Specification

This document provides function signatures, expected inputs, outputs, and exception behaviors for all core Python modules in the `care_asr` package.

---

## 1. Module: `care_asr.transcriber.whisper`

### Function: `transcribe()`

```python
def transcribe(
    audio: AudioInput,
    model_name: str = "whisper-medium",
    device: str = "cuda",
    beam_size: int = 5,
    temperature: float = 0.0
) -> Transcript:
    """Transcribes input audio stream using OpenAI Whisper ASR model and extracts token logit distributions.

    Args:
        audio: AudioInput object containing WAV bytes or file path.
        model_name: Name of Whisper model checkpoint ('whisper-medium' or 'whisper-large-v3').
        device: Hardware device for inference ('cuda' or 'cpu').
        beam_size: Beam search width (Default: 5).
        temperature: Decoding temperature (Default: 0.0 for greedy decoding).

    Returns:
        Transcript: Fully populated Transcript object containing raw text, tokens, and 2D logits tensor.

    Raises:
        FileNotFoundError: If audio.file_path does not exist on disk.
        ValueError: If sample rate is not 16000 Hz.
        RuntimeError: If CUDA out-of-memory error occurs during decoding.
    """
```

---

## 2. Module: `care_asr.confidence.tsallis`

### Function: `compute_confidence()`

```python
def compute_confidence(
    transcript: Transcript,
    q: float = 0.5,
    tau: float = 0.45
) -> ConfidenceScore:
    """Computes Tsallis non-extensive entropy across token logit probability distributions.

    Args:
        transcript: Transcript object containing raw token logits.
        q: Tsallis entropy non-extensivity parameter (Default: 0.5).
        tau: Entropy cutoff threshold for setting uncertainty flags (Default: 0.45).

    Returns:
        ConfidenceScore: Object containing token/word entropy values and boolean flags.

    Raises:
        ValueError: If q == 1.0 (Tsallis entropy undefined at q=1; use Shannon entropy instead).
        ValueError: If logits tensor is 0-dimensional or empty.
    """
```

---

## 3. Module: `care_asr.ner.tagger`

### Function: `extract_entities()`

```python
def extract_entities(
    transcript: Transcript,
    confidence: ConfidenceScore,
    model_path: str = "emilyalsentzer/Bio_ClinicalBERT"
) -> List[EntitySpan]:
    """Identifies and classifies medical entities in the transcript text using ClinicalBERT.

    Args:
        transcript: Transcript object containing raw text and word alignments.
        confidence: ConfidenceScore object containing per-word uncertainty flags.
        model_path: HuggingFace model path for clinical NER tagger.

    Returns:
        List[EntitySpan]: List of detected clinical entity spans (MED, COND, ANA, TTP).

    Raises:
        RuntimeError: If HuggingFace model checkpoint fails to load.
        IndexError: If character offsets exceed transcript length.
    """
```

---

## 4. Module: `care_asr.retrieval.semantic`

### Function: `retrieve_semantic()`

```python
def retrieve_semantic(
    entity: EntitySpan,
    sentence_context: str,
    top_k: int = 10,
    index_path: str = "data/indices/faiss_umls.index"
) -> List[RetrievalCandidate]:
    """Queries dense FAISS index using ClinicalBERT embeddings of the entity and sentence context.

    Args:
        entity: EntitySpan object containing target text and category.
        sentence_context: Surrounding sentence string to guide context embedding.
        top_k: Number of nearest neighbor candidates to retrieve (Default: 10).
        index_path: Path to serialized FAISS index file.

    Returns:
        List[RetrievalCandidate]: Candidate terms retrieved via semantic similarity.

    Raises:
        FileNotFoundError: If index_path is missing.
        RuntimeError: If FAISS query vector dimension mismatches index dimension (768).
    """
```

---

## 5. Module: `care_asr.retrieval.phonetic`

### Function: `retrieve_phonetic()`

```python
def retrieve_phonetic(
    entity: EntitySpan,
    top_k: int = 10,
    dict_path: str = "data/indices/medical_vocab.json"
) -> List[RetrievalCandidate]:
    """Retrieves candidates based on Double Metaphone phonetic encoding and Levenshtein distance.

    Args:
        entity: EntitySpan object containing misrecognized entity string.
        top_k: Number of phonetic candidates to retrieve (Default: 10).
        dict_path: Path to medical vocabulary dictionary file.

    Returns:
        List[RetrievalCandidate]: Candidate terms retrieved via phonetic matching.

    Raises:
        FileNotFoundError: If dictionary file is missing.
    """
```

---

## 6. Module: `care_asr.fusion.rrf`

### Function: `fuse_candidates()`

```python
def fuse_candidates(
    semantic_candidates: List[RetrievalCandidate],
    phonetic_candidates: List[RetrievalCandidate],
    rrf_k: int = 60,
    top_n: int = 5
) -> List[FusionCandidate]:
    """Combines semantic and phonetic candidates into a single ranked list using Reciprocal Rank Fusion.

    Args:
        semantic_candidates: Candidates from Module 4.
        phonetic_candidates: Candidates from Module 5.
        rrf_k: RRF smoothing constant (Default: 60).
        top_n: Number of top candidates to return (Default: 5).

    Returns:
        List[FusionCandidate]: Unified candidate list sorted by descending RRF score.
    """
```

---

## 7. Module: `care_asr.correction.llm`

### Function: `generate_correction()`

```python
def generate_correction(
    context_sentence: str,
    entity: EntitySpan,
    candidates: List[FusionCandidate],
    model_endpoint: str = "http://localhost:11434"
) -> str:
    """Prompts Llama-3.1-8B-Instruct to select the best replacement concept or return UNSURE.

    Args:
        context_sentence: Original sentence containing the uncertain entity.
        entity: Target EntitySpan object.
        candidates: Top fusion candidate terms.
        model_endpoint: Ollama/vLLM HTTP server API endpoint.

    Returns:
        str: Proposed replacement term or 'UNSURE'.

    Raises:
        ConnectionError: If LLM service endpoint is unreachable.
        TimeoutError: If LLM generation exceeds timeout limit (5.0s).
    """
```

---

## 8. Module: `care_asr.safety.gate`

### Function: `validate_correction()`

```python
def validate_correction(
    original_entity: EntitySpan,
    proposed_term: str,
    selected_candidate: Optional[FusionCandidate] = None,
    max_edit_ratio: float = 0.45
) -> CorrectionResult:
    """Executes medical safety checks (Levenshtein distance, category preservation) on proposed edits.

    Args:
        original_entity: Original EntitySpan object.
        proposed_term: Term proposed by LLM engine.
        selected_candidate: Corresponding FusionCandidate object (if any).
        max_edit_ratio: Maximum allowable normalized edit distance ratio (Default: 0.45).

    Returns:
        CorrectionResult: Approved correction outcome object.
    """
```

---

## 9. Module: `care_asr.core.pipeline`

### Function: `run_pipeline()`

```python
def run_pipeline(
    audio: AudioInput,
    config_path: str = "configs/pipeline.yaml"
) -> PipelineOutput:
    """Executes the full CARE-ASR pipeline from audio input to final clinical transcript.

    Args:
        audio: AudioInput object.
        config_path: Path to pipeline YAML configuration file.

    Returns:
        PipelineOutput: Complete execution response containing corrected transcript and telemetry.

    Raises:
        Exception: Captures and logs any module failure, returning uncorrected transcript as fallback.
    """
```
