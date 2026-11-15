import asyncio
import json

import httpx

from infer_lab.datasets import Prompt
from infer_lab.loadgen import send_request

PROMPT = Prompt("Hello", 1)


def _sse(*events):
    lines = [f"data: {json.dumps(e)}\n\n" for e in events] + ["data: [DONE]\n\n"]
    return "".join(lines).encode()


def _send(handler):
    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await send_request(client, "http://test", "m", PROMPT, max_tokens=3, request_id=7)

    return asyncio.run(go())


def test_stream_is_timed_and_usage_is_read():
    body = _sse(
        {"choices": [{"text": "a"}]},
        {"choices": [{"text": ""}]},  # empty text: not a chunk we time
        {"choices": [{"text": "b"}]},
        {"choices": [], "usage": {"completion_tokens": 3}},
    )
    r = _send(lambda req: httpx.Response(200, content=body))

    assert r.ok and r.request_id == 7
    assert r.output_tokens == 3
    assert len(r.chunk_times_s) == 2
    assert r.start_s <= r.chunk_times_s[0] <= r.chunk_times_s[1] <= r.end_s


def test_request_body_is_controlled_workload():
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        return httpx.Response(200, content=_sse({"choices": [{"text": "a"}], "usage": {"completion_tokens": 1}}))

    _send(handler)
    assert seen["temperature"] == 0 and seen["ignore_eos"] is True and seen["stream"] is True


def test_http_error_is_recorded_not_raised():
    r = _send(lambda req: httpx.Response(503, text="overloaded"))
    assert not r.ok and "503" in r.error


def test_connection_error_is_recorded_not_raised():
    def handler(req):
        raise httpx.ConnectError("refused")

    r = _send(handler)
    assert not r.ok and "ConnectError" in r.error


def test_missing_usage_is_a_failure():
    r = _send(lambda req: httpx.Response(200, content=_sse({"choices": [{"text": "a"}]})))
    assert not r.ok
