"""模型推理配置测试。"""

import pytest

from pi.ai.providers import anthropic, openai
from pi.ai.streaming import DoneEvent
from pi.ai.types import (
    AssistantMessage,
    Context,
    Model,
    ModelThinkingLevel,
    StopReason,
    StreamOptions,
    TextContent,
)


async def _finish_stream(model, stream):
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


@pytest.mark.parametrize(
    ("provider", "module", "expected"),
    [
        (openai.OpenAIProvider(), openai, {"reasoning_effort": "high"}),
        (
            anthropic.AnthropicProvider(),
            anthropic,
            {"thinking": {"type": "enabled", "budget_tokens": 8192}},
        ),
    ],
)
async def test_reasoning_options_are_added_to_provider_payload(
    monkeypatch,
    provider,
    module,
    expected,
):
    captured = {}

    async def fake_worker(url, headers, payload, model, stream, max_retries, timeout):
        captured.update(payload)
        await _finish_stream(model, stream)

    worker_name = "_stream_openai" if module is openai else "_stream_anthropic"
    monkeypatch.setattr(module, worker_name, fake_worker)
    model = Model(
        id="reasoning-model",
        name="Reasoning",
        api="test",
        provider=provider.provider_id,
        reasoning=True,
    )
    options = StreamOptions(
        api_key="test-key",
        thinking_level=ModelThinkingLevel.HIGH,
    )

    stream = await provider.stream(model, Context(), options)
    events = [event async for event in stream]

    assert events[-1].type == "done"
    for key, value in expected.items():
        assert captured[key] == value
