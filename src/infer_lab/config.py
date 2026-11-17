"""Experiment definition, loaded from configs/experiments/*.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Precision = Literal["fp16", "bf16", "fp8", "awq-int4"]


class DatasetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["sharegpt", "long_context", "shared_prefix"]
    # long_context: prompt length in tokens; shared_prefix: length of the prefix every request shares
    input_len: int | None = None
    output_len: int = 256  # every request generates exactly this many tokens (ignore_eos)
    seed: int = 0

    @model_validator(mode="after")
    def _length_given_when_needed(self) -> DatasetSpec:
        if self.kind != "sharegpt" and self.input_len is None:
            raise ValueError(f"dataset.input_len is required for kind={self.kind}")
        return self


class Experiment(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    name: str
    milestone: str
    runtime: Literal["vllm", "sglang", "trtllm"]
    model: str
    precision: Precision
    tp: int = Field(default=1, ge=1)
    dp: int = Field(default=1, ge=1)  # independent replicas, one server each
    max_num_seqs: int | None = None
    prefix_caching: bool = False
    max_model_len: int | None = None
    extra_server_args: list[str] = Field(default_factory=list)
    runtime_version: str | None = None  # set when the server runs outside this Python env
    interconnect: str = "PCIe"
    pod_hourly_usd: float = 0.0  # Kaggle is free
    port: int = 8000

    concurrency: list[int]
    dataset: DatasetSpec
    min_requests: int = 32
    requests_per_slot: int = 4
    warmup_requests: int = 16
    repeats: int = Field(default=3, ge=1)
    max_cv: float = 0.05

    @property
    def gpu_count(self) -> int:
        return self.tp * self.dp

    def features(self) -> list[str]:
        out = []
        if self.prefix_caching:
            out.append("prefix_caching")
        if self.max_num_seqs is not None:
            out.append(f"max_num_seqs={self.max_num_seqs}")
        if self.dataset.kind == "long_context":
            out.append(f"long_context={self.dataset.input_len}")
        if self.dataset.kind == "shared_prefix":
            out.append(f"shared_prefix={self.dataset.input_len}")
        if self.extra_server_args:
            out.append("args=" + " ".join(self.extra_server_args))
        return out

    def num_requests(self, concurrency: int) -> int:
        """Enough requests that the server spends most of the run at the target concurrency."""
        return max(self.min_requests, self.requests_per_slot * concurrency)


def load_experiment(path: Path | str) -> Experiment:
    with open(path, encoding="utf-8") as f:
        return Experiment.model_validate(yaml.safe_load(f))
