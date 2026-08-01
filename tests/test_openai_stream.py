"""OpenAI SSE 解析集成测试。"""

from contextlib import AbstractAsyncContextManager

from pi.ai.providers import openai
from pi.ai.streaming import EventStream
from pi.ai.types import Model, StopReason, ThinkingContent


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

    def __init__(self, lines):
        self.lines = lines

    async def aiter_lines(self):
        for line in self.lines:
            yield line


def _install_fake_client(monkeypatch, lines):
    class FakeClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            return False

        def stream(self, method, url, headers, json):
            return _ResponseContext(_FakeResponse(lines))

    monkeypatch.setattr(openai.httpx, "AsyncClient", FakeClient)


async def _run_stream():
    stream = EventStream()
    model = Model(id="gpt-test", name="Test", api="test", provider="openai")
    await openai._stream_openai(
        "https://example.test",
        {},
        {},
        model,
        stream,
        max_retries=0,
        timeout=1,
    )
    return [event async for event in stream]


async def test_openai_sse_tracks_cached_and_reasoning_tokens(monkeypatch):
    _install_fake_client(
        monkeypatch,
        [
            'data: {"choices":[{"delta":{"content":"done"},"finish_reason":"stop"}]}',
            'data: {"choices":[],"usage":{"prompt_tokens":10,'
            '"completion_tokens":6,"total_tokens":16,'
            '"prompt_tokens_details":{"cached_tokens":4},'
            '"completion_tokens_details":{"reasoning_tokens":2}}}',
            "data: [DONE]",
        ],
    )

    events = await _run_stream()

    message = events[-1].message
    assert message.usage.input == 6
    assert message.usage.cache_read == 4
    assert message.usage.output == 6
    assert message.usage.reasoning == 2
    assert message.usage.total_tokens == 16


async def test_openai_sse_emits_reasoning_content(monkeypatch):
    _install_fake_client(
        monkeypatch,
        [
            'data: {"choices":[{"delta":{"reasoning_content":"think"},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ],
    )

    events = await _run_stream()

    assert [event.type for event in events].count("thinking_delta") == 1
    assert isinstance(events[-1].message.content[0], ThinkingContent)
    assert events[-1].message.content[0].thinking == "think"


async def test_openai_rejects_invalid_tool_arguments(monkeypatch):
    _install_fake_client(
        monkeypatch,
        [
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            '"id":"call-1","function":{"name":"read","arguments":"{"}}]},'
            '"finish_reason":"tool_calls"}]}',
            "data: [DONE]",
        ],
    )

    events = await _run_stream()

    assert events[-1].type == "error"
    assert events[-1].error.stop_reason == StopReason.ERROR
    assert "invalid arguments for tool read" in events[-1].error.error_message


async def test_openai_surfaces_stream_errors(monkeypatch):
    _install_fake_client(
        monkeypatch,
        ['data: {"error":{"message":"provider failed"}}'],
    )

    events = await _run_stream()

    assert events[-1].type == "error"
    assert events[-1].error.error_message == "provider failed"
