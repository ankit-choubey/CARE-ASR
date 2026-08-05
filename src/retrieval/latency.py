"""
Latency instrumentation helpers for the CARE-ASR retrieval pipeline (T12).

Provides a single reusable timing implementation (LatencyStats) shared by the
pipeline instrumentation and the latency benchmark. All durations are recorded
in milliseconds; no timing logic is duplicated anywhere else.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager


class LatencyStats:
    """Collects wall-clock latency samples under named labels.

    Supports manual timing (``start``/``stop``), direct recording (``record``),
    and a context-manager form (``timed``) so every measurement path shares one
    implementation. Samples are stored in milliseconds.

    Attributes:
        _start_times (dict[str, float]): Active timer start points.
        _samples (defaultdict[str, list[float]]): Recorded samples per label.
    """

    _start_times: dict[str, float]
    _samples: defaultdict[str, list[float]]

    def __init__(self) -> None:
        self._start_times = {}
        self._samples = defaultdict(list)

    def start(self, name: str) -> None:
        """Starts a manual timer for ``name``."""
        self._start_times[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        """Stops the timer for ``name`` and records the elapsed milliseconds.

        Args:
            name (str): Timer label previously started with ``start``.

        Returns:
            float: Elapsed milliseconds recorded for ``name``.

        Raises:
            ValueError: If no timer is active for ``name``.
        """
        start_time = self._start_times.pop(name, None)
        if start_time is None:
            raise ValueError(f"No active timer for '{name}'. Call start('{name}') first.")
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self.record(name, elapsed_ms)
        return elapsed_ms

    def record(self, name: str, elapsed_ms: float) -> None:
        """Records an externally measured duration (milliseconds) under ``name``."""
        self._samples[name].append(float(elapsed_ms))

    @contextmanager
    def timed(self, name: str) -> Iterator[None]:
        """Context manager that times the wrapped block and records it under ``name``.

        The block is timed even when it raises; the exception then propagates.
        """
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def values(self, name: str) -> list[float]:
        """Returns a copy of all recorded samples for ``name`` (empty when none)."""
        return list(self._samples.get(name, []))

    def names(self) -> list[str]:
        """Returns the recorded label names in insertion order."""
        return list(self._samples.keys())

    def summary(self) -> dict[str, float]:
        """Returns the mean latency (ms) recorded under each label."""
        return {name: self._mean(samples) for name, samples in self._samples.items()}

    @staticmethod
    def _mean(values: list[float]) -> float:
        """Returns the arithmetic mean of ``values`` (0.0 when empty)."""
        return sum(values) / len(values) if values else 0.0
