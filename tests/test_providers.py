"""提供商注册与兼容端点测试。"""

from pi.ai.providers import get_provider, list_providers, openai
from pi.ai.providers.deepseek import DeepSeekProvider
from pi.ai.streaming import DoneEvent
from pi.ai.types import (
    AssistantMessage,
    Context,
    Model,
    StopReason,
    StreamOptions,
    TextContent,
)
from pi.coding_agent import runtime


def test_builtin_provider_ecosystem_is_registered():
    assert {
        "anthropic",
        "deepseek",
        "groq",
        "lmstudio",
        "mistral",
        "ollama",
        "openai",
        "openrouter",
        "xai",
    }.issubset(list_providers())
    assert isinstance(get_provider("deepseek"), DeepSeekProvider)


async def test_openai_compatible_provider_merges_headers_and_allows_local_auth(
    monkeypatch,
):
    captured = {}

    async def fake_worker(url, headers, payload, model, stream, max_retries, timeout):
        captured["url"] = url
        captured["headers"] = headers
        message = AssistantMessage(
            content=[TextContent(text="ok")],
            api=model.api,
            provider=model.provider,
            model=model.id,
            stop_reason=StopReason.STOP,
            timestamp=1,
        )
        await stream.push(DoneEvent(message=message))
        await stream.end(message)

    monkeypatch.setattr(openai, "_stream_openai", fake_worker)
    provider = openai.OpenAIProvider("local", requires_api_key=False)
    model = Model(
        id="local-model",
        name="Local",
        api="openai-chat-completions",
        provider="local",
        base_url="http://localhost:11434/v1",
        headers={"X-Model": "model", "X-Remove": "remove"},
    )
    options = StreamOptions(
        headers={"X-Call": "call", "X-Remove": None},
        max_retries=0,
    )

    stream = await provider.stream(model, Context(), options)
    events = [event async for event in stream]

    assert events[-1].type == "done"
    assert captured["url"] == "http://localhost:11434/v1/chat/completions"
    assert captured["headers"] == {
        "Content-Type": "application/json",
        "X-Model": "model",
        "X-Call": "call",
    }


async def test_stream_function_resolves_provider_for_current_model(monkeypatch):
    calls = []

    class FakeProvider:
        def __init__(self, provider_id):
            self.provider_id = provider_id

        async def stream(self, model, context, options):
            calls.append((self.provider_id, model.id))
            return self.provider_id

    providers = {
        "first": FakeProvider("first"),
        "second": FakeProvider("second"),
    }
    monkeypatch.setattr(runtime, "get_provider", providers.get)
    stream_fn = runtime.make_stream_fn()
    first = Model(id="one", name="One", api="test", provider="first")
    second = Model(id="two", name="Two", api="test", provider="second")

    assert await stream_fn(first, Context(), None) == "first"
    assert await stream_fn(second, Context(), None) == "second"
    assert calls == [("first", "one"), ("second", "two")]
