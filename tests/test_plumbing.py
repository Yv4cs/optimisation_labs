from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from infer_lab.config import load_experiment
from infer_lab.schema import RunRecord, append_run, load_runs
from infer_lab.server import ServerGroup, build_command, trtllm_options

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


def test_trtllm_command_and_options():
    exp = load_experiment(EXAMPLE.parent / "h100_qwen3-8b_trtllm_fp8.yaml").model_copy(update={"max_num_seqs": 64})
    cmd = build_command(exp, 8000, Path("opts.yaml"))
    assert cmd[:3] == ["trtllm-serve", "serve", "nvidia/Qwen3-8B-FP8"]
    assert cmd[cmd.index("--extra_llm_api_options") + 1] == "opts.yaml"
    assert cmd[cmd.index("--max_batch_size") + 1] == "64"
    assert trtllm_options(exp) == {"kv_cache_config": {"enable_block_reuse": False, "dtype": "fp8"}}
    bf16 = exp.model_copy(update={"precision": "bf16", "prefix_caching": True})
    assert trtllm_options(bf16) == {"kv_cache_config": {"enable_block_reuse": True}}
    with pytest.raises(ValueError):
        trtllm_options(exp.model_copy(update={"precision": "fp16"}))


@pytest.mark.parametrize("runtime, kv_dtype", [("vllm", "fp8"), ("sglang", "fp8_e4m3")])
def test_fp8_reads_the_checkpoint_and_stores_kv_in_fp8(runtime, kv_dtype):
    exp = load_experiment(EXAMPLE.parent / f"h100_qwen3-8b_{runtime}_fp8.yaml")
    cmd = build_command(exp, 8000)
    assert "--quantization" not in cmd
    assert cmd[cmd.index("--kv-cache-dtype") + 1] == kv_dtype


def test_data_parallel_gets_one_url_per_replica():
    exp = load_experiment(EXAMPLE).model_copy(update={"dp": 2})
    assert ServerGroup(exp, Path("unused")).urls == ["http://127.0.0.1:8000", "http://127.0.0.1:8001"]


def test_shared_prefix_needs_a_length_and_is_tagged():
    exp = load_experiment(EXAMPLE)
    with pytest.raises(ValidationError):
        exp.dataset.model_validate({"kind": "shared_prefix"})
    spec = exp.dataset.model_validate({"kind": "shared_prefix", "input_len": 1024})
    assert "shared_prefix=1024" in exp.model_copy(update={"dataset": spec}).features()


@pytest.mark.parametrize("path", sorted(EXAMPLE.parent.glob("*.yaml")), ids=lambda p: p.name)
def test_every_experiment_config_loads_and_builds(path):
    exp = load_experiment(path)
    cmd = build_command(exp, exp.port)
    assert exp.model in cmd
    assert ServerGroup(exp, Path("unused")).urls[0].endswith(str(exp.port))
