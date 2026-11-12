"""Run one experiment: start servers, warm up, then for each concurrency level run N repeats.

Every repeat becomes one line in runs.jsonl plus raw files in results/raw/<experiment>-<stamp>/<run_id>/.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from . import env_info
from .config import Experiment, load_experiment
from .datasets import Prompt, load_prompts
from .loadgen import RequestResult, run_load
from .metrics import cv, summarize
from .power import GpuSampler, energy_joules
from .schema import RunRecord, append_run
from .server import ServerGroup

MAX_ATTEMPTS = 2  # one re-run of a concurrency level when repeats disagree (CV > max_cv)


def run_experiment(exp: Experiment, results_dir: Path, smoke: bool = False) -> None:
    levels = exp.concurrency[:1] if smoke else exp.concurrency
    repeats = 1 if smoke else exp.repeats
    attempts = 1 if smoke else MAX_ATTEMPTS
    per_level = [min(exp.num_requests(c), 8) if smoke else exp.num_requests(c) for c in levels]

    # Fresh prompts for every warm-up and repeat, so prefix caching never reuses an earlier repeat.
    pool_size = exp.warmup_requests + attempts * repeats * sum(per_level)
    max_prompt_tokens = (exp.max_model_len or 4096) - exp.dataset.output_len
    pool = iter(load_prompts(exp.dataset, exp.model, pool_size, max_prompt_tokens))

    env = env_info.collect(exp.runtime, exp.model)
    if exp.runtime_version:
        env["runtime_version"] = exp.runtime_version
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_root = results_dir / "raw" / f"{exp.name}-{stamp}"

    with ServerGroup(exp, raw_root) as servers:
        print(f"warm-up: {exp.warmup_requests} requests", flush=True)
        asyncio.run(
            run_load(servers.urls, exp.model, _take(pool, exp.warmup_requests), exp.warmup_requests, exp.dataset.output_len)
        )
        for concurrency, n in zip(levels, per_level):
            for attempt in range(attempts):
                records = [
                    _measure(exp, env, servers.urls, _take(pool, n), concurrency, k, raw_root) for k in range(repeats)
                ]
                tps = [r.tokens_per_s for r in records if r.tokens_per_s is not None]
                level_cv = cv(tps) if len(tps) > 1 else 0.0
                accepted = len(tps) == repeats and level_cv <= exp.max_cv
                for r in records:
                    r.notes = f"cv_tokens_per_s={level_cv:.3f}" + ("" if accepted else "; discarded")
                    append_run(r, results_dir / "runs.jsonl")
                print(
                    f"c={concurrency} attempt={attempt} tokens/s={[round(x, 1) for x in tps]} "
                    f"cv={level_cv:.3f} {'ok' if accepted else 'DISCARDED'}",
                    flush=True,
                )
                if accepted:
                    break
            else:
                if not smoke:
                    print(f"c={concurrency}: repeats still disagree after {attempts} attempts; investigate", flush=True)


def _measure(
    exp: Experiment,
    env: dict,
    urls: list[str],
    prompts: list[Prompt],
    concurrency: int,
    repeat_idx: int,
    raw_root: Path,
) -> RunRecord:
    run_id = f"{exp.name}-c{concurrency}-r{repeat_idx}-{uuid.uuid4().hex[:6]}"
    with GpuSampler(range(exp.gpu_count)) as sampler:
        results, wall_s = asyncio.run(run_load(urls, exp.model, prompts, concurrency, exp.dataset.output_len))
    s = summarize(results, wall_s)
    energy_j = energy_joules(sampler.samples)

    run_dir = raw_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_requests_csv(results, run_dir / "requests.csv")
    sampler.write_csv(run_dir / "power.csv")
    provenance = {**env, "energy_trapz_j": energy_j, "energy_counter_j": sampler.energy_counter_j}
    (run_dir / "env.json").write_text(json.dumps(provenance, indent=2))

    util = [x.util_pct for x in sampler.samples]
    return RunRecord(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc),
        milestone=exp.milestone,
        experiment=exp.name,
        repeat_idx=repeat_idx,
        git_commit=env["git_commit"],
        gpu=env["gpu"],
        gpu_count=exp.gpu_count,
        interconnect=exp.interconnect,
        driver=env["driver"],
        cuda=env["cuda"],
        runtime=exp.runtime,
        runtime_version=env["runtime_version"],
        model=exp.model,
        model_revision=env.get("model_revision"),
        precision=exp.precision,
        tp=exp.tp,
        dp=exp.dp,
        features=exp.features(),
        concurrency=concurrency,
        input_len=s.mean_prompt_tokens,
        output_len=s.mean_output_tokens,
        ttft_p50_ms=s.ttft_p50_ms,
        ttft_p99_ms=s.ttft_p99_ms,
        itl_p50_ms=s.itl_p50_ms,
        tokens_per_s=s.tokens_per_s,
        req_per_s=s.req_per_s,
        gpu_util_pct=statistics.fmean(util) if util else None,
        vram_gb=sampler.peak_vram_gb(),
        avg_power_w=energy_j / sampler.duration_s if sampler.duration_s > 0 else None,
        joules_per_token=energy_j / s.output_tokens if s.output_tokens else None,
        cost_usd=exp.pod_hourly_usd * wall_s / 3600,
    )


def _take(pool: Iterator[Prompt], n: int) -> list[Prompt]:
    return [next(pool) for _ in range(n)]


def _write_requests_csv(results: list[RequestResult], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["request_id", "error", "prompt_tokens", "output_tokens", "start_s", "end_s", "chunk_times_s"])
        for r in results:
            writer.writerow(
                [r.request_id, r.error or "", r.prompt_tokens, r.output_tokens, r.start_s, r.end_s, json.dumps(r.chunk_times_s)]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", required=True, help="path to configs/experiments/*.yaml")
    parser.add_argument("--results", default="results")
    parser.add_argument("--smoke", action="store_true", help="plumbing check; logs to <results>/smoke/")
    args = parser.parse_args()
    results_dir = Path(args.results) / "smoke" if args.smoke else Path(args.results)
    run_experiment(load_experiment(args.exp), results_dir, smoke=args.smoke)


if __name__ == "__main__":
    main()
