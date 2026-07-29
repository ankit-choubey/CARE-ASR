"""Offline Error Analysis & Audit Evaluation Subpackage for CARE-ASR Module.

Why It Exists:
    Calculates precision, recall, F1 gains, and root-cause failure taxonomy counts across
    the 5 official CARE-ASR categories (MED, COND, ANA, TTP, PHI).

Teammate Dependencies:
    - Mahi (Testing & QA Lead): Consumes `ErrorAnalysisAuditOutput` for benchmark audit runs.

Imported By:
    - Automated test suites and offline benchmark evaluation scripts.

TODOs:
    - Add HTML visual error matrix report generator for Mahi's final demo presentation.
"""

from care_asr.evaluation.metrics_calculator import ErrorAnalysisEngine
from care_asr.evaluation.taxonomy_classifier import FailureTaxonomyClassifier

__all__ = ["ErrorAnalysisEngine", "FailureTaxonomyClassifier"]
