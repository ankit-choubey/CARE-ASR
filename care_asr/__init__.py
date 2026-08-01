"""CARE-ASR: Confidence-Aware Retrieval-Augmented Clinical Entity Recovery.

BioBERT NER, Medical Entity Schema, Retrieval Candidate Validation, Thresholding & Error Analysis Module.

Why It Exists:
    Acts as the core intelligence, entity extraction, thresholding, candidate validation,
    and audit evaluation package for CARE-ASR.

Teammate Dependencies:
    - Ankit (ASR & Integration Lead): Consumes validated candidates and NER spans.
    - Divya (FAISS & Retrieval Lead): Provides FAISS candidates mapped to entity queries.
    - Mahi (Testing & QA Lead): Consumes error analysis audit reports and unit tests.

Imported By:
    - External pipeline integration scripts (`Ankit`).
    - Quality assurance test runners (`Mahi`).

TODOs:
    - Integrate BioBERT token classification weights.
    - Connect FAISS candidate evaluation pipelines.
"""

import logging

__version__ = "2.0.0"
__author__ = "Lead Architect"

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
