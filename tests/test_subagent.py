"""subagent 工具与 agent 循环 max_turns 上限的测试。"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from pi.agent.agent import Agent, AgentOptions
from pi.agent.agent_loop import run_agent_loop
from pi.agent.tools.base import ToolContext
from pi.agent.types import (
    AgentContext,
    AgentMessage,
    AgentTool,
    AgentToolCall,
    AgentToolResult,
    AgentToolResultMessage,
    create_user_message,
)
from pi.ai.streaming import DoneEvent, EventStream, StartEvent, TextDeltaEvent, ToolCallEndEvent
from pi.ai.types import (
    AssistantMessage,
    Model,
    StopReason,
    TextContent,
    ToolCall,
    Usage,
)
from pi.coding_agent.tools.subagent import (
    MAX_SUBAGENT_DEPTH,
    create_subagent_tool,
    execute,
)


def make_model(model_id: str = "test-model") -> Model:
    return Model(
        id=model_id,
        name="Test model",
        api="openai-chat-completions",
        provider="openai",
        base_url="http://localhost",
        context_window=128000,
        max_tokens=4096,
    )


def make_scripted_stream(script: list[Any]) -> tuple[Any, dict[str, int]]:
    """返回按调用次数重放 `script` 的 stream_fn。

    每个元素是字符串（最终文本）或 ToolCall；多余调用重复最后一个元素。
    state 字典记录 stream_fn 的调用次数。
    """
    state: dict[str, int] = {"calls": 0}

    async def stream_fn(model: Model, context, options):
        index = min(state["calls"], len(script) - 1)
        state["calls"] += 1
        payload = script[index]
        stream = EventStream()

        async def produce() -> None:
            timestamp = int(time.time() * 1000)
            if isinstance(payload, ToolCall):
                blocks: list[Any] = [payload]
                stop_reason = StopReason.TOOL_USE
            else:
                blocks = [TextContent(text=payload)]
                stop_reason = StopReason.STOP
            message = AssistantMessage(
                content=blocks,
                api="openai-chat-completions",
                provider="openai",
                model="test-model",
                usage=Usage(),
                stop_reason=stop_reason,
                timestamp=timestamp,
            )
            await stream.push(StartEvent(partial=message))
            for block in blocks:
                if isinstance(block, TextContent):
                    await stream.push(TextDeltaEvent(delta=block.text))
            for block in blocks:
                if isinstance(block, ToolCall):
                    await stream.push(ToolCallEndEvent(tool_call=block))
            await stream.push(
                DoneEvent(
                    reason="toolUse" if stop_reason == StopReason.TOOL_USE else "stop",
                    message=message,
                )
            )
            await stream.end(message)

        task = asyncio.create_task(produce())
        stream.set_producer_task(task)
        return stream

    return stream_fn, state


def _subagent_results(agent: Agent) -> list[AgentToolResultMessage]:
    return [
        message
        for message in agent.state.messages
        if isinstance(message, AgentToolResultMessage) and message.tool_name == "subagent"
    ]


def _text_of(message: AgentMessage) -> str:
    return "\n".join(
        block.text for block in message.content if isinstance(block, TextContent) and block.text
    )


async def test_agent_registers_itself_in_tool_context() -> None:
    tool_context = ToolContext()
    agent = Agent(
        AgentOptions(
            model=make_model(),
            tools=[create_subagent_tool()],
            tool_context=tool_context,
        )
    )
    assert tool_context.state["agent"] is agent


async def test_subagent_end_to_end() -> None:
    subagent_call = ToolCall(
        id="p1",
        name="subagent",
        arguments={"task": "Summarize the repo", "tools": "none"},
    )
    stream_fn, state = make_scripted_stream(
        [subagent_call, "child done", "parent done"]
    )
    agent = Agent(
        AgentOptions(
            model=make_model(),
            tools=[create_subagent_tool()],
            stream_fn=stream_fn,
            tool_context=ToolContext(),
        )
    )
    await agent.prompt("go")

    results = _subagent_results(agent)
    assert len(results) == 1
    assert _text_of(results[0]) == "child done"
    assert state["calls"] == 3
    last_assistant = next(
        message
        for message in reversed(agent.state.messages)
        if message.role == "assistant" and _text_of(message)
    )
    assert _text_of(last_assistant) == "parent done"


async def test_subagent_max_turns() -> None:
    subagent_call = ToolCall(
        id="p1",
        name="subagent",
        arguments={"task": "investigate", "tools": "analysis", "max_turns": 2},
    )
    read_call = ToolCall(id="r1", name="read", arguments={"path": "missing.py"})
    stream_fn, state = make_scripted_stream(
        [subagent_call, read_call, read_call, "parent done"]
    )
    agent = Agent(
        AgentOptions(
            model=make_model(),
            tools=[create_subagent_tool()],
            stream_fn=stream_fn,
            tool_context=ToolContext(),
        )
    )
    await agent.prompt("go")

    results = _subagent_results(agent)
    assert len(results) == 1
    assert "without a final answer" in _text_of(results[0])
    # 父 agent 1 轮 + 子 agent 2 轮；子 agent 在第 3 次模型调用前停止。
    assert state["calls"] == 4


async def test_subagent_depth_limit() -> None:
    parent = Agent(
        AgentOptions(
            model=make_model(),
            tools=[create_subagent_tool()],
            stream_fn=make_scripted_stream(["unused"])[0],
        )
    )
    ctx = ToolContext()
    ctx.state["agent"] = parent
    ctx.state["subagent_depth"] = MAX_SUBAGENT_DEPTH
    call = AgentToolCall(id="d1", name="subagent", arguments={"task": "x"})

    result = await execute(call, ctx)

    assert result.is_error
    assert "depth limit" in result.content[0].text


async def test_agent_loop_max_turns() -> None:
    async def stub_execute(call: AgentToolCall, _ctx: ToolContext | None) -> AgentToolResult:
        return AgentToolResult(
            tool_call_id=call.id,
            tool_name="stub",
            content=[TextContent(text="ok")],
        )

    stub = AgentTool(
        name="stub",
        description="Stub tool",
        parameters={"type": "object", "properties": {}},
        execute=stub_execute,
    )
    stream_fn, state = make_scripted_stream(
        [ToolCall(id="c1", name="stub", arguments={})]
    )
    model = make_model()

    async def sink(event) -> None:
        pass

    result = await run_agent_loop(
        prompts=[create_user_message("run")],
        context=AgentContext(system_prompt="", messages=[], tools=[stub]),
        model=model,
        stream_fn=stream_fn,
        sink=sink,
        max_turns=1,
    )

    assert state["calls"] == 1
    assert any(
        isinstance(message, AgentToolResultMessage) and message.tool_name == "stub"
        for message in result.messages
    )
