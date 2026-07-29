"""Unit Tests for ErrorAnalysisEngine & FailureTaxonomyClassifier.

Why It Exists:
    Verifies offline audit metrics calculation stubs and taxonomy classification schemas.

Teammate Dependencies:
    - Mahi (Testing & QA Lead): Executes pytest suite for regression testing.

Imported By:
    - `pytest` runner.

TODOs:
    - Add tests for category metrics validation across MED, COND, ANA, TTP, and PHI.
"""

import logging
import pytest

from care_asr.evaluation.metrics_calculator import ErrorAnalysisEngine
from care_asr.evaluation.taxonomy_classifier import FailureTaxonomyClassifier

logger = logging.getLogger(__name__)


def test_error_analysis_engine_initialization() -> None:
    """Tests ErrorAnalysisEngine instantiation."""
    engine = ErrorAnalysisEngine()
    assert engine is not None


def test_failure_taxonomy_classifier_interface() -> None:
    """Tests FailureTaxonomyClassifier static methods."""
    assert hasattr(FailureTaxonomyClassifier, "classify_failure")
    assert hasattr(FailureTaxonomyClassifier, "aggregate_taxonomy")
