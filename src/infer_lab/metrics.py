"""Turn per-request timings into the latency and throughput numbers we report."""

from __future__ import annotations

import math
import statistics
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
    """Time from sending the request to its first streamed chunk, in ms."""
    return (r.chunk_times_s[0] - r.start_s) * 1000


def itl_ms(r: RequestResult) -> list[float]:
    """Gaps between consecutive chunks in ms (n chunks -> n-1 gaps).

    If a runtime packs several tokens into one chunk this is per-chunk, not per-token (docs/methodology.md D-008).
    """
    t = r.chunk_times_s
    return [(b - a) * 1000 for a, b in zip(t, t[1:])]


def percentile(values: Sequence[float], p: float) -> float:
    """p in [0, 100], linear interpolation between closest ranks (numpy's default)."""
    xs = sorted(values)
    rank = (len(xs) - 1) * p / 100  # fractional index into the sorted list
    lo = math.floor(rank)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (rank - lo)


def cv(values: Sequence[float]) -> float:
    """Coefficient of variation = sample std (ddof=1) / mean.

    The runner re-runs a concurrency level when the repeats' tokens/s CV exceeds 5%.
    """
    return statistics.stdev(values) / statistics.fmean(values)


def summarize(results: Sequence[RequestResult], wall_time_s: float) -> Summary:
    """Aggregate one load run.

    - latency stats use successful requests only; ITL p50 pools the gaps of all of them
    - tokens_per_s = successful output tokens / wall_time_s; req_per_s = successful requests / wall_time_s
    - with zero successful requests, every metric is None ("not measured")
    """
    ok = [r for r in results if r.ok]
    num_failed = len(results) - len(ok)
    if not ok:
        return Summary(0, num_failed, 0, None, None, None, None, None, None, None)

    output_tokens = sum(r.output_tokens for r in ok)
    ttfts = [ttft_ms(r) for r in ok]
    gaps = [g for r in ok for g in itl_ms(r)]
    return Summary(
        num_ok=len(ok),
        num_failed=num_failed,
        output_tokens=output_tokens,
        mean_prompt_tokens=statistics.fmean(r.prompt_tokens for r in ok),
        mean_output_tokens=output_tokens / len(ok),
        ttft_p50_ms=percentile(ttfts, 50),
        ttft_p99_ms=percentile(ttfts, 99),
        itl_p50_ms=percentile(gaps, 50) if gaps else None,
        tokens_per_s=output_tokens / wall_time_s,
        req_per_s=len(ok) / wall_time_s,
    )
