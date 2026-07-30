"""提供商重试策略测试。"""

from contextlib import AbstractAsyncContextManager

import pytest

from pi.ai.providers import openai
from pi.ai.providers.retry import RetryableProviderError, run_with_retries
from pi.ai.streaming import EventStream
from pi.ai.types import Model


class _ResponseContext(AbstractAsyncContextManager):
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False


class _FakeResponse:
    def __init__(self, status_code, lines=None, headers=None, body=b""):
        self.status_code = status_code
        self._lines = lines or []
        self.headers = headers or {}
        self._body = body

    async def aread(self):
        return self._body

    async def aiter_lines(self):
        for line in self._lines:
            yield line


async def test_openai_retries_rate_limit_before_streaming(monkeypatch):
    responses = [
        _FakeResponse(429, headers={"Retry-After": "0"}, body=b"busy"),
        _FakeResponse(
            200,
            lines=[
                'data: {"choices":[{"delta":{"reasoning_content":"think",'
                '"content":"ok"},"finish_reason":"stop"}]}',
                "data: [DONE]",
            ],
        ),
    ]

    class FakeClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            return False

        def stream(self, method, url, headers, json):
            return _ResponseContext(responses.pop(0))

    monkeypatch.setattr(openai.httpx, "AsyncClient", FakeClient)
    stream = EventStream()
    model = Model(id="test", name="Test", api="test", provider="openai")

    await openai._stream_openai(
        "https://example.test",
        {},
        {},
        model,
        stream,
        max_retries=1,
        timeout=1,
    )
    events = [event async for event in stream]

    errors = [event.error.error_message for event in events if event.type == "error"]
    assert not errors
    assert [event.type for event in events] == [
        "start",
        "retry",
        "thinking_delta",
        "text_delta",
        "done",
    ]
    assert events[1].attempt == 1
    assert events[-1].message.content[0].thinking == "think"
    assert events[-1].message.content[1].text == "ok"
    assert not responses


async def test_retry_stops_after_partial_output():
    attempts = 0
    stream = EventStream()

    async def operation():
        nonlocal attempts
        attempts += 1
        raise RetryableProviderError("temporary", "0")

    with pytest.raises(RetryableProviderError):
        await run_with_retries(operation, lambda: True, stream, max_retries=3)

    assert attempts == 1
