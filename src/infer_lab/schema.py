"""Validated record for one measured run; one JSON line per record in results/runs.jsonl.

A field left as None means "not measured". Never fill it with an estimate.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Runtime = Literal["vllm", "sglang", "trtllm"]


class RunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    # identity and provenance
    run_id: str
    timestamp: datetime
    milestone: str
    experiment: str
    repeat_idx: int = Field(ge=0)  # mean ± std is computed across repeats at report time
    git_commit: str

    # hardware and software
    gpu: str
    gpu_count: int = Field(ge=1)
    interconnect: str
    driver: str
    cuda: str
    runtime: Runtime
    runtime_version: str
    model: str
    model_revision: str | None = None  # Hugging Face commit sha ("model hash")
    precision: str
    tp: int = Field(ge=1)
    dp: int = Field(ge=1)
    features: list[str] = Field(default_factory=list)

    # workload (measured means, in tokens)
    concurrency: int = Field(ge=1)
    input_len: float | None = None
    output_len: float | None = None

    # results
    ttft_p50_ms: float | None = None
    ttft_p99_ms: float | None = None
    itl_p50_ms: float | None = None
    tokens_per_s: float | None = None
    req_per_s: float | None = None
    gpu_util_pct: float | None = None
    vram_gb: float | None = None
    avg_power_w: float | None = None
    joules_per_token: float | None = None
    accuracy: dict[str, float] = Field(default_factory=dict)
    cost_usd: float | None = None
    notes: str = ""


def append_run(record: RunRecord, path: Path | str = "results/runs.jsonl") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")


def load_runs(path: Path | str = "results/runs.jsonl") -> list[RunRecord]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [RunRecord.model_validate_json(line) for line in f if line.strip()]
