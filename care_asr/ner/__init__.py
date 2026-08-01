"""BioBERT Named Entity Recognition (NER) Subpackage for CARE-ASR Module.

Why It Exists:
    Performs HuggingFace BioBERT token classification, subtoken aggregation,
    and subtoken-to-character span alignment against Whisper ASR word timestamps.

Teammate Dependencies:
    - Ankit (ASR & Integration Lead): Provides raw transcript text and word alignments;
      consumes extracted clinical entity spans.

Imported By:
    - `care_asr.validation.candidate_evaluator`
    - Integration pipeline entrypoints.

TODOs:
    - Add batch tokenization optimization for long medical audio transcripts.
"""

from care_asr.ner.extractor import BioBertNERExtractor
from care_asr.ner.span_aligner import SpanAligner

__all__ = ["BioBertNERExtractor", "SpanAligner"]
