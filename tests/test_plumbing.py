from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from infer_lab.config import load_experiment
from infer_lab.schema import RunRecord, append_run, load_runs
from infer_lab.server import ServerGroup, build_command

EXAMPLE = Path(__file__).parent.parent / "configs/experiments/t4_qwen3b_fp16.yaml"


def _record(**overrides):
    base = dict(
        run_id="r1",
        timestamp=datetime(2026, 11, 13, tzinfo=timezone.utc),
        milestone="M1",
        experiment="exp",
        repeat_idx=0,
        git_commit="abc",
        gpu="Tesla T4",
        gpu_count=1,
        interconnect="PCIe",
        driver="550",
        cuda="12.4",
        runtime="vllm",
        runtime_version="x",
        model="m",
        precision="fp16",
        tp=1,
        dp=1,
        concurrency=1,
    )
    return RunRecord(**{**base, **overrides})


def test_record_roundtrip(tmp_path):
    path = tmp_path / "runs.jsonl"
    append_run(_record(tokens_per_s=12.5), path)
    append_run(_record(run_id="r2"), path)
    runs = load_runs(path)
    assert [r.run_id for r in runs] == ["r1", "r2"]
    assert runs[0].tokens_per_s == 12.5
    assert runs[1].tokens_per_s is None  # not measured stays None


def test_record_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        _record(tokens_per_second=1.0)


def test_example_experiment_loads():
    exp = load_experiment(EXAMPLE)
    assert exp.gpu_count == 1
    assert exp.num_requests(1) == 32
    assert exp.num_requests(256) == 1024


def test_vllm_command_on_t4():
    cmd = build_command(load_experiment(EXAMPLE), 8000)
    assert cmd[:3] == ["vllm", "serve", "Qwen/Qwen2.5-3B-Instruct"]
    assert "float16" in cmd
    assert "--no-enable-prefix-caching" in cmd


def test_sglang_command_maps_flags():
    exp = load_experiment(EXAMPLE).model_copy(update={"runtime": "sglang", "tp": 2, "max_num_seqs": 64})
    cmd = build_command(exp, 8000)
    assert "--disable-radix-cache" in cmd
    assert cmd[cmd.index("--tp-size") + 1] == "2"
    assert cmd[cmd.index("--max-running-requests") + 1] == "64"


def test_data_parallel_gets_one_url_per_replica():
    exp = load_experiment(EXAMPLE).model_copy(update={"dp": 2})
    assert ServerGroup(exp, Path("unused")).urls == ["http://127.0.0.1:8000", "http://127.0.0.1:8001"]
