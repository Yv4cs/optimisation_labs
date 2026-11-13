"""Background NVML sampler (power, utilisation, memory) and energy integration."""

from __future__ import annotations

import csv
import threading
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import astuple, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    t_s: float  # time.perf_counter()
    gpu: int  # NVML device index
    power_w: float
    util_pct: float
    mem_used_gb: float  # GB = 1e9 bytes


class GpuSampler:
    """`with GpuSampler([0, 1]) as s: ...` polls every `interval_s` until the block exits."""

    def __init__(self, gpu_indices: Iterable[int], interval_s: float = 0.1):
        self.gpu_indices = list(gpu_indices)
        self.interval_s = interval_s
        self.samples: list[Sample] = []
        self.energy_counter_j: float | None = None  # NVML's own energy counter, for cross-checking
        self._stop = threading.Event()

    def __enter__(self) -> GpuSampler:
        import pynvml

        self._nvml = pynvml
        pynvml.nvmlInit()
        self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in self.gpu_indices]
        self._start_mj = self._total_energy_mj()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        end_mj = self._total_energy_mj()
        if self._start_mj is not None and end_mj is not None:
            self.energy_counter_j = (end_mj - self._start_mj) / 1000
        self._nvml.nvmlShutdown()

    def _loop(self) -> None:
        nvml = self._nvml
        while not self._stop.is_set():
            t = time.perf_counter()
            for idx, h in zip(self.gpu_indices, self._handles):
                self.samples.append(
                    Sample(
                        t_s=t,
                        gpu=idx,
                        power_w=nvml.nvmlDeviceGetPowerUsage(h) / 1000,
                        util_pct=float(nvml.nvmlDeviceGetUtilizationRates(h).gpu),
                        mem_used_gb=nvml.nvmlDeviceGetMemoryInfo(h).used / 1e9,
                    )
                )
            self._stop.wait(self.interval_s)

    def _total_energy_mj(self) -> float | None:
        try:
            return float(sum(self._nvml.nvmlDeviceGetTotalEnergyConsumption(h) for h in self._handles))
        except self._nvml.NVMLError:
            return None  # not supported on every GPU

    @property
    def duration_s(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        return self.samples[-1].t_s - self.samples[0].t_s

    def peak_vram_gb(self) -> float | None:
        """Peak of memory summed across GPUs. Includes the runtime's up-front KV-cache reservation."""
        per_tick: dict[float, float] = defaultdict(float)
        for s in self.samples:
            per_tick[s.t_s] += s.mem_used_gb
        return max(per_tick.values(), default=None)

    def write_csv(self, path: Path) -> None:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([fld.name for fld in fields(Sample)])
            writer.writerows(astuple(s) for s in self.samples)


def energy_joules(samples: Sequence[Sample]) -> float:
    """Total energy in joules across all GPUs.

    Power is a rate (W = J/s); energy is the area under the power-vs-time curve.
    Integrate each GPU separately (sort by time, trapezoid rule), then sum the GPUs.
    A GPU with fewer than two samples contributes 0.
    """
    by_gpu: dict[int, list[Sample]] = defaultdict(list)
    for s in samples:
        by_gpu[s.gpu].append(s)

    total = 0.0
    for gpu_samples in by_gpu.values():
        gpu_samples.sort(key=lambda s: s.t_s)
        for a, b in zip(gpu_samples, gpu_samples[1:]):
            total += (a.power_w + b.power_w) / 2 * (b.t_s - a.t_s)
    return total
