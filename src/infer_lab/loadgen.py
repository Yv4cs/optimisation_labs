"""Async load generator: N simulated users streaming completions from an OpenAI-compatible server.

YOU write send_request and run_load. tests/ and runner.py rely only on the signatures below.
"""

from __future__ import annotations

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
    """TODO(you): send one streaming completion request and time it.

    POST {base_url}/v1/completions with JSON:
        {"model": model, "prompt": prompt.text, "max_tokens": max_tokens, "temperature": 0,
         "stream": True, "ignore_eos": True, "stream_options": {"include_usage": True}}

    The reply is Server-Sent Events: lines like `data: {...json...}`, ending with `data: [DONE]`.

    Record:
    - start_s right before sending
    - one entry in chunk_times_s for every chunk whose choices[0]["text"] is non-empty
    - output_tokens from the final usage chunk (usage.completion_tokens), the server's own count
    - end_s when the stream finishes
    - on a non-200 status or any exception: set `error` to a short message, never raise

    Hints: `async with client.stream("POST", url, json=body) as resp:` and
    `async for line in resp.aiter_lines():`
    """
    raise NotImplementedError


async def run_load(
    base_urls: list[str],
    model: str,
    prompts: list[Prompt],
    concurrency: int,
    max_tokens: int,
) -> tuple[list[RequestResult], float]:
    """TODO(you): send every prompt with at most `concurrency` requests in flight.

    Returns (results, wall_time_s).
    - one httpx.AsyncClient for the whole run (timeout=None, connection limits above `concurrency`)
    - asyncio.Semaphore(concurrency) caps in-flight requests
    - spread requests round-robin over base_urls (one URL per data-parallel replica)
    - wall_time_s: from just before the first request is sent until the last one finishes

    Think: why does a semaphore model "N users" better than sending all requests at once?
    """
    raise NotImplementedError
