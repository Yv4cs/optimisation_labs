"""Async load generator: N simulated users streaming completions from an OpenAI-compatible server."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field

import httpx

from .datasets import Prompt


@dataclass
class RequestResult:
    request_id: int
    prompt_tokens: int
    output_tokens: int = 0
    start_s: float = 0.0  # time.perf_counter() just before the request is sent
    chunk_times_s: list[float] = field(default_factory=list)  # perf_counter() at each chunk carrying text
    end_s: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


async def send_request(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: Prompt,
    max_tokens: int,
    request_id: int,
) -> RequestResult:
    """Send one streaming completion request and time it. Never raises: failures go in `error`.

    The reply is Server-Sent Events: lines like `data: {...json...}`, ending with `data: [DONE]`.
    A chunk counts towards chunk_times_s only if it carries text; output_tokens is the server's own
    count from the final usage chunk.
    """
    result = RequestResult(request_id=request_id, prompt_tokens=prompt.prompt_tokens)
    body = {
        "model": model,
        "prompt": prompt.text,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "ignore_eos": True,
        "stream_options": {"include_usage": True},
    }
    result.start_s = time.perf_counter()
    try:
        async with client.stream("POST", f"{base_url}/v1/completions", json=body) as resp:
            if resp.status_code != 200:
                await resp.aread()
                result.error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                return result
            async for line in resp.aiter_lines():
                now = time.perf_counter()  # read the clock before parsing, so JSON work is not timed
                if not line.startswith("data:"):
                    continue  # blank separator lines between events
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                choices = chunk.get("choices") or []
                if choices and choices[0].get("text"):
                    result.chunk_times_s.append(now)
                if chunk.get("usage"):
                    result.output_tokens = chunk["usage"]["completion_tokens"]
        # Without these the request cannot be measured; counting it as a success would skew the numbers.
        if not result.chunk_times_s:
            result.error = "stream ended without any text"
        elif result.output_tokens == 0:
            result.error = "stream ended without a usage chunk"
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"[:200]
    finally:
        result.end_s = time.perf_counter()
    return result


async def run_load(
    base_urls: list[str],
    model: str,
    prompts: list[Prompt],
    concurrency: int,
    max_tokens: int,
) -> tuple[list[RequestResult], float]:
    """Send every prompt with at most `concurrency` requests in flight; return (results, wall_time_s).

    The semaphore models N users who each send their next request as soon as the previous one
    finishes, so the server sees a steady N in flight instead of one burst of len(prompts).
    Requests go round-robin over base_urls (one URL per data-parallel replica).
    """
    semaphore = asyncio.Semaphore(concurrency)
    # More connections than in-flight requests, so no request ever waits for a free connection.
    limits = httpx.Limits(max_connections=concurrency + 16, max_keepalive_connections=concurrency + 16)

    async with httpx.AsyncClient(timeout=None, limits=limits) as client:

        async def one_user_turn(i: int, prompt: Prompt) -> RequestResult:
            async with semaphore:
                url = base_urls[i % len(base_urls)]
                return await send_request(client, url, model, prompt, max_tokens, i)

        t0 = time.perf_counter()
        results = await asyncio.gather(*(one_user_turn(i, p) for i, p in enumerate(prompts)))
        wall_time_s = time.perf_counter() - t0

    return list(results), wall_time_s
