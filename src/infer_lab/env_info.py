"""Provenance for every run: GPU, driver, CUDA, runtime version, model revision, git commit."""

from __future__ import annotations

import json
import platform
import subprocess
from importlib import metadata

_RUNTIME_PACKAGES = {"vllm": "vllm", "sglang": "sglang", "trtllm": "tensorrt_llm"}


def git_commit() -> str:
    """HEAD sha, suffixed with -dirty if tracked code (not results/) has uncommitted changes."""
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        dirty = _run(["git", "status", "--porcelain", "--untracked-files=no", "--", ".", ":!results"])
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha + ("-dirty" if dirty else "")


def gpu_info() -> dict[str, str | int]:
    try:
        import pynvml

        pynvml.nvmlInit()
    except Exception:
        return {"gpu": "none", "gpu_count_visible": 0, "driver": "none", "cuda_driver": "none"}
    try:
        count = pynvml.nvmlDeviceGetCount()
        name = pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0)) if count else "none"
        driver = pynvml.nvmlSystemGetDriverVersion()
        cuda = pynvml.nvmlSystemGetCudaDriverVersion()  # e.g. 12040 -> "12.4"
    finally:
        pynvml.nvmlShutdown()
    return {
        "gpu": _str(name),
        "gpu_count_visible": count,
        "driver": _str(driver),
        "cuda_driver": f"{cuda // 1000}.{cuda % 1000 // 10}",
    }


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def cuda_runtime() -> str:
    """CUDA version PyTorch was built against (what kernels actually use)."""
    try:
        import torch
    except ImportError:
        return "unknown"
    return torch.version.cuda or "unknown"


def model_revision(repo_id: str) -> str | None:
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(repo_id).sha
    except Exception:
        return None


def collect(runtime: str | None = None, model: str | None = None) -> dict:
    info = {
        "git_commit": git_commit(),
        **gpu_info(),
        "cuda": cuda_runtime(),
        "python": platform.python_version(),
        "torch": package_version("torch"),
    }
    if runtime is not None:
        info["runtime_version"] = package_version(_RUNTIME_PACKAGES[runtime])
    else:
        info.update({rt: package_version(pkg) for rt, pkg in _RUNTIME_PACKAGES.items()})
    if model is not None:
        info["model_revision"] = model_revision(model)
    return info


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()


def _str(value: str | bytes) -> str:
    return value.decode() if isinstance(value, bytes) else value


if __name__ == "__main__":
    print(json.dumps(collect(), indent=2))
