"""机器可读打印输出测试。"""

import json

from pi.agent.types import (
    AgentAssistantMessage,
    AgentEndEvent,
    TextDeltaUpdateEvent,
    create_user_message,
)
from pi.ai.types import TextContent, Usage
from pi.coding_agent.output import PrintRenderer


async def test_json_output_emits_single_result_with_usage(capsys):
    renderer = PrintRenderer("json")
    messages = [
        create_user_message("hello"),
        AgentAssistantMessage(
            content=[TextContent(text="ok")],
            api="test",
            provider="test",
            model="test-model",
            usage=Usage(input=3, output=2, total_tokens=5),
            stop_reason="stop",
            timestamp=1,
        ),
    ]

    await renderer(AgentEndEvent(messages=messages))
    result = json.loads(capsys.readouterr().out)

    assert result["type"] == "agent_result"
    assert result["messages"][1]["content"][0]["text"] == "ok"
    assert result["usage"]["total_tokens"] == 5
    assert result["error"] is None


async def test_jsonl_output_emits_each_event(capsys):
    renderer = PrintRenderer("jsonl")

    await renderer(TextDeltaUpdateEvent(delta="hello"))
    event = json.loads(capsys.readouterr().out)

    assert event == {"type": "text_delta", "delta": "hello", "content_index": 0}


async def test_text_renderer_is_awaitable(capsys):
    renderer = PrintRenderer("text")
    await renderer(TextDeltaUpdateEvent(delta="hello"))
    assert capsys.readouterr().out == "hello"
