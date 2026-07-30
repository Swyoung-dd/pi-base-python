"""提供商注册与兼容端点测试。"""

from pi.ai.providers import list_providers, openai
from pi.ai.streaming import DoneEvent
from pi.ai.types import (
    AssistantMessage,
    Context,
    Model,
    StopReason,
    StreamOptions,
    TextContent,
)


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
