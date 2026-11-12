"""Launch inference servers (one per data-parallel replica), wait until healthy, stop cleanly."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

from .config import Experiment

# AWQ checkpoints carry their quantization config; the runtime picks the fastest AWQ kernel for the GPU.
_PRECISION_ARGS = {
    "fp16": ["--dtype", "float16"],
    "bf16": ["--dtype", "bfloat16"],
    "fp8": ["--quantization", "fp8"],
    "awq-int4": ["--dtype", "float16"],
}


def build_command(exp: Experiment, port: int) -> list[str]:
    if exp.runtime == "vllm":
        cmd = ["vllm", "serve", exp.model, "--host", "127.0.0.1", "--port", str(port)]
        cmd += ["--tensor-parallel-size", str(exp.tp)]
        cmd += _PRECISION_ARGS[exp.precision]
        cmd.append("--enable-prefix-caching" if exp.prefix_caching else "--no-enable-prefix-caching")
        if exp.max_num_seqs is not None:
            cmd += ["--max-num-seqs", str(exp.max_num_seqs)]
        if exp.max_model_len is not None:
            cmd += ["--max-model-len", str(exp.max_model_len)]
    else:  # sglang
        cmd = [sys.executable, "-m", "sglang.launch_server", "--model-path", exp.model]
        cmd += ["--host", "127.0.0.1", "--port", str(port), "--tp-size", str(exp.tp)]
        cmd += _PRECISION_ARGS[exp.precision]
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
                proc = subprocess.Popen(
                    build_command(self.exp, self.exp.port + i),
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
