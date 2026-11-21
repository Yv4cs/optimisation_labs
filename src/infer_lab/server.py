"""Launch inference servers (one per data-parallel replica), wait until healthy, stop cleanly."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import yaml

from .config import Experiment

# Quantized checkpoints (AWQ, ModelOpt FP8) carry their quantization config and every runtime reads it, so
# no --quantization flag (docs/methodology.md D-010, D-011). "fp8" also stores the KV-cache in FP8 on every runtime: the ModelOpt
# checkpoint ships KV scales, and leaving it to each engine's default would make the KV dtype differ by engine.
_PRECISION_ARGS = {
    "vllm": {
        "fp16": ["--dtype", "float16"],
        "bf16": ["--dtype", "bfloat16"],
        "fp8": ["--kv-cache-dtype", "fp8"],
        "awq-int4": ["--dtype", "float16"],
    },
    "sglang": {
        "fp16": ["--dtype", "float16"],
        "bf16": ["--dtype", "bfloat16"],
        "fp8": ["--kv-cache-dtype", "fp8_e4m3"],
        "awq-int4": ["--dtype", "float16"],
    },
}


def trtllm_options(exp: Experiment) -> dict:
    """LLM API options for trtllm-serve, which takes KV-cache settings only from a YAML file."""
    if exp.precision not in ("bf16", "fp8"):
        raise ValueError(f"trtllm runs are planned for bf16/fp8 on H100 only, got {exp.precision}")
    kv_cache = {"enable_block_reuse": exp.prefix_caching}  # block reuse is TRT-LLM's prefix caching
    if exp.precision == "fp8":
        kv_cache["dtype"] = "fp8"
    return {"kv_cache_config": kv_cache}


def build_command(exp: Experiment, port: int, options_path: Path = Path("trtllm_options.yaml")) -> list[str]:
    """`options_path` is only used by trtllm; ServerGroup writes trtllm_options() there before launching."""
    if exp.runtime == "trtllm":
        cmd = ["trtllm-serve", "serve", exp.model, "--host", "127.0.0.1", "--port", str(port)]
        cmd += ["--tp_size", str(exp.tp), "--extra_llm_api_options", str(options_path)]
        if exp.max_num_seqs is not None:
            cmd += ["--max_batch_size", str(exp.max_num_seqs)]
        if exp.max_model_len is not None:
            cmd += ["--max_seq_len", str(exp.max_model_len)]
        return cmd + exp.extra_server_args
    if exp.runtime == "vllm":
        cmd = ["vllm", "serve", exp.model, "--host", "127.0.0.1", "--port", str(port)]
        cmd += ["--tensor-parallel-size", str(exp.tp)]
        cmd += _PRECISION_ARGS["vllm"][exp.precision]
        cmd.append("--enable-prefix-caching" if exp.prefix_caching else "--no-enable-prefix-caching")
        if exp.max_num_seqs is not None:
            cmd += ["--max-num-seqs", str(exp.max_num_seqs)]
        if exp.max_model_len is not None:
            cmd += ["--max-model-len", str(exp.max_model_len)]
    else:  # sglang
        cmd = [sys.executable, "-m", "sglang.launch_server", "--model-path", exp.model]
        cmd += ["--host", "127.0.0.1", "--port", str(port), "--tp-size", str(exp.tp)]
        cmd += _PRECISION_ARGS["sglang"][exp.precision]
        if not exp.prefix_caching:
            cmd.append("--disable-radix-cache")
        if exp.max_num_seqs is not None:
            cmd += ["--max-running-requests", str(exp.max_num_seqs)]
        if exp.max_model_len is not None:
            cmd += ["--context-length", str(exp.max_model_len)]
    return cmd + exp.extra_server_args


class ServerGroup:
    """Context manager: `dp` servers, each pinned to its own `tp` GPUs."""

    def __init__(self, exp: Experiment, log_dir: Path, ready_timeout_s: float = 1200):
        self.exp = exp
        self.log_dir = log_dir
        self.ready_timeout_s = ready_timeout_s
        self._procs: list[tuple[subprocess.Popen, object]] = []

    @property
    def urls(self) -> list[str]:
        return [f"http://127.0.0.1:{self.exp.port + i}" for i in range(self.exp.dp)]

    def __enter__(self) -> ServerGroup:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        try:
            for i in range(self.exp.dp):
                gpus = ",".join(str(g) for g in range(i * self.exp.tp, (i + 1) * self.exp.tp))
                # PCI order makes CUDA device ids match NVML ids used by the power sampler.
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpus, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
                log = open(self.log_dir / f"server_{i}.log", "w")
                options_path = self.log_dir / f"trtllm_options_{i}.yaml"
                if self.exp.runtime == "trtllm":
                    options_path.write_text(yaml.safe_dump(trtllm_options(self.exp)))
                proc = subprocess.Popen(
                    build_command(self.exp, self.exp.port + i, options_path),
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,  # own process group, so stop() also kills workers
                )
                self._procs.append((proc, log))
            for i, ((proc, _), url) in enumerate(zip(self._procs, self.urls)):
                self._wait_ready(i, proc, url)
        except BaseException:
            self.stop()
            raise
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def _wait_ready(self, i: int, proc: subprocess.Popen, url: str) -> None:
        deadline = time.monotonic() + self.ready_timeout_s
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"server {i} exited with code {proc.returncode}:\n{self._log_tail(i)}")
            try:
                if httpx.get(f"{url}/health", timeout=5).status_code == 200:
                    print(f"server {i} ready at {url}", flush=True)
                    return
            except httpx.HTTPError:
                pass
            time.sleep(5)
        raise TimeoutError(f"server {i} not ready after {self.ready_timeout_s}s:\n{self._log_tail(i)}")

    def _log_tail(self, i: int, lines: int = 40) -> str:
        text = (self.log_dir / f"server_{i}.log").read_text(errors="replace")
        return "\n".join(text.splitlines()[-lines:])

    def stop(self) -> None:
        for proc, _ in self._procs:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        for proc, log in self._procs:
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            log.close()
        self._procs = []
