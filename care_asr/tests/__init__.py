"""Test Suite Subpackage for CARE-ASR Module.

Why It Exists:
    Contains unit and contract tests verifying component behavior, schema validation,
    and thresholding logic prior to handing over module components to Mahi.

Teammate Dependencies:
    - Mahi (Testing Lead): Runs pytest test suites in CI/CD pipeline.

Imported By:
    - `pytest` runner.

TODOs:
    - Add mock JSON fixture loaders for offline testing without GPUs.
"""
