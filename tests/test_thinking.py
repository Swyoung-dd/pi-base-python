"""模型推理配置测试。"""

import pytest

from pi.ai.providers import anthropic, openai
from pi.ai.providers.deepseek import DeepSeekProvider
from pi.ai.streaming import DoneEvent
from pi.ai.types import (
    AssistantMessage,
    Context,
    Model,
    ModelThinkingLevel,
    StopReason,
    StreamOptions,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
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


@pytest.mark.parametrize(
    ("level", "expected_thinking", "expected_effort"),
    [
        (ModelThinkingLevel.OFF, {"type": "disabled"}, None),
        (ModelThinkingLevel.HIGH, {"type": "enabled"}, "high"),
        (ModelThinkingLevel.MAX, {"type": "enabled"}, "max"),
    ],
)
async def test_deepseek_v4_thinking_payload(
    monkeypatch,
    level,
    expected_thinking,
    expected_effort,
):
    captured = {}

    async def fake_worker(url, headers, payload, model, stream, max_retries, timeout):
        captured.update(payload)
        await _finish_stream(model, stream)

    monkeypatch.setattr(openai, "_stream_openai", fake_worker)
    provider = DeepSeekProvider()
    model = Model(
        id="deepseek-v4-flash",
        name="DeepSeek V4 Flash",
        api="openai-chat-completions",
        provider="deepseek",
        base_url="https://api.deepseek.com",
        reasoning=True,
    )

    stream = await provider.stream(
        model,
        Context(),
        StreamOptions(api_key="test-key", thinking_level=level),
    )
    events = [event async for event in stream]

    assert events[-1].type == "done"
    assert captured["thinking"] == expected_thinking
    assert captured.get("reasoning_effort") == expected_effort


def test_deepseek_preserves_reasoning_content_for_tool_results():
    provider = DeepSeekProvider()
    context = Context(
        messages=[
            AssistantMessage(
                content=[
                    ThinkingContent(thinking="需要读取文件"),
                    ToolCall(id="call-1", name="read", arguments={"path": "README.md"}),
                ],
                api="openai-chat-completions",
                provider="deepseek",
                model="deepseek-v4-flash",
                stop_reason=StopReason.TOOL_USE,
                timestamp=1,
            ),
            ToolResultMessage(
                tool_call_id="call-1",
                tool_name="read",
                content=[TextContent(text="content")],
                timestamp=2,
            ),
        ]
    )

    messages = provider.convert_messages(context)

    assert messages[0]["reasoning_content"] == "需要读取文件"
    assert messages[0]["tool_calls"][0]["id"] == "call-1"
    assert messages[1]["tool_call_id"] == "call-1"
