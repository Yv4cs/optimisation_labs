import pytest

from infer_lab.loadgen import RequestResult
from infer_lab.metrics import cv, itl_ms, percentile, summarize, ttft_ms


def _req(i, start, chunks, out_tokens, prompt_tokens=10, error=None):
    return RequestResult(
        request_id=i,
        prompt_tokens=prompt_tokens,
        output_tokens=out_tokens,
        start_s=start,
        chunk_times_s=chunks,
        end_s=chunks[-1] if chunks else start,
        error=error,
    )


def test_ttft_is_time_to_first_chunk():
    assert ttft_ms(_req(0, 1.0, [1.1, 1.15, 1.2], 3)) == pytest.approx(100.0)


def test_itl_is_gap_between_chunks():
    assert itl_ms(_req(0, 0.0, [0.1, 0.15, 0.25], 3)) == pytest.approx([50.0, 100.0])


def test_itl_single_chunk_has_no_gaps():
    assert itl_ms(_req(0, 0.0, [0.1], 1)) == []


def test_percentile_linear_interpolation():
    values = list(range(1, 101))
    assert percentile(values, 50) == pytest.approx(50.5)
    assert percentile(values, 99) == pytest.approx(99.01)


def test_cv_uses_sample_std():
    assert cv([10, 10, 10]) == 0
    assert cv([9, 10, 11]) == pytest.approx(0.1)


def test_summarize_uses_successful_requests_only():
    ok1 = _req(0, 0.0, [0.1, 0.2], 2, prompt_tokens=10)
    ok2 = _req(1, 0.0, [0.3, 0.4, 0.5], 3, prompt_tokens=20)
    failed = _req(2, 0.0, [], 0, error="timeout")
    s = summarize([ok1, ok2, failed], wall_time_s=1.0)

    assert (s.num_ok, s.num_failed) == (2, 1)
    assert s.output_tokens == 5
    assert s.tokens_per_s == pytest.approx(5.0)
    assert s.req_per_s == pytest.approx(2.0)
    assert s.mean_prompt_tokens == pytest.approx(15.0)
    assert s.mean_output_tokens == pytest.approx(2.5)
    assert s.ttft_p50_ms == pytest.approx(200.0)  # median of 100 and 300
    assert s.itl_p50_ms == pytest.approx(100.0)  # pooled gaps: [100] + [100, 100]


def test_summarize_with_no_success_is_not_measured():
    s = summarize([_req(0, 0.0, [], 0, error="refused")], wall_time_s=1.0)
    assert s.num_ok == 0
    assert s.ttft_p50_ms is None
    assert s.tokens_per_s is None
