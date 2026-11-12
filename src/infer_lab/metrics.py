"""Turn per-request timings into the latency and throughput numbers we report.

YOU write the functions below; tests/test_metrics.py defines what "correct" means.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .loadgen import RequestResult


@dataclass(frozen=True)
class Summary:
    num_ok: int
    num_failed: int
    output_tokens: int  # summed over successful requests
    mean_prompt_tokens: float | None
    mean_output_tokens: float | None
    ttft_p50_ms: float | None
    ttft_p99_ms: float | None
    itl_p50_ms: float | None
    tokens_per_s: float | None
    req_per_s: float | None


def ttft_ms(r: RequestResult) -> float:
    """TODO(you): time from sending the request to its first streamed chunk, in ms."""
    raise NotImplementedError


def itl_ms(r: RequestResult) -> list[float]:
    """TODO(you): gaps between consecutive chunks in ms (n chunks -> n-1 gaps).

    If a runtime packs several tokens into one chunk this is per-chunk, not per-token (DECISIONS D-008).
    """
    raise NotImplementedError


def percentile(values: Sequence[float], p: float) -> float:
    """TODO(you): p in [0, 100], linear interpolation between closest ranks (numpy's default)."""
    raise NotImplementedError


def cv(values: Sequence[float]) -> float:
    """TODO(you): coefficient of variation = sample std (ddof=1) / mean.

    The runner re-runs a concurrency level when the repeats' tokens/s CV exceeds 5%.
    """
    raise NotImplementedError


def summarize(results: Sequence[RequestResult], wall_time_s: float) -> Summary:
    """TODO(you): aggregate one load run.

    - latency stats use successful requests only; ITL p50 pools the gaps of all of them
    - tokens_per_s = successful output tokens / wall_time_s; req_per_s = successful requests / wall_time_s
    - with zero successful requests, every metric is None ("not measured")
    """
    raise NotImplementedError
