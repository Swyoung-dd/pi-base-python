"""Anthropic SSE 解析集成测试。"""

from contextlib import AbstractAsyncContextManager

from pi.ai.providers import anthropic
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
    status_code = 200
    headers = {}

    async def aiter_lines(self):
        lines = [
            'data: {"type":"message_start","message":{"usage":{"input_tokens":3}}}',
            'data: {"type":"content_block_delta","delta":{"type":"thinking_delta",'
            '"thinking":"plan"}}',
            'data: {"type":"content_block_delta","delta":{"type":"text_delta",'
            '"text":"done"}}',
            'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
            '"usage":{"output_tokens":2}}',
            'data: {"type":"message_stop"}',
        ]
        for line in lines:
            yield line


async def test_anthropic_sse_produces_thinking_text_and_usage(monkeypatch):
    class FakeClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            return False

        def stream(self, method, url, headers, json):
            return _ResponseContext(_FakeResponse())

    monkeypatch.setattr(anthropic.httpx, "AsyncClient", FakeClient)
    stream = EventStream()
    model = Model(id="claude-test", name="Test", api="test", provider="anthropic")

    await anthropic._stream_anthropic(
        "https://example.test",
        {},
        {},
        model,
        stream,
        max_retries=0,
        timeout=1,
    )
    events = [event async for event in stream]

    assert [event.type for event in events] == [
        "start",
        "thinking_delta",
        "text_delta",
        "done",
    ]
    message = events[-1].message
    assert message.content[0].thinking == "plan"
    assert message.content[1].text == "done"
    assert message.usage.total_tokens == 5
