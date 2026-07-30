"""工具并行调度测试。"""

import asyncio

from pi.agent.agent import Agent, AgentOptions
from pi.agent.types import AgentTool, AgentToolCall, AgentToolResult
from pi.ai.streaming import DoneEvent, EventStream
from pi.ai.types import AssistantMessage, Model, StopReason, TextContent, ToolCall


async def test_parallel_mode_starts_all_tools_before_waiting_for_results():
    model = Model(id="test", name="Test", api="test", provider="test")
    release = asyncio.Event()
    both_started = asyncio.Event()
    started: list[str] = []
    provider_calls = 0

    async def execute(call: AgentToolCall, context) -> AgentToolResult:
        started.append(call.name)
        if len(started) == 2:
            both_started.set()
        await release.wait()
        return AgentToolResult(
            tool_call_id=call.id,
            tool_name=call.name,
            content=[TextContent(text=call.name)],
        )

    tools = [
        AgentTool(
            name=name,
            description=name,
            parameters={"type": "object", "properties": {}},
            execute=execute,
        )
        for name in ("first", "second")
    ]

    async def stream_fn(current_model, context, options):
        nonlocal provider_calls
        provider_calls += 1
        stream = EventStream()
        if provider_calls == 1:
            content = [
                ToolCall(id="call-1", name="first", arguments={}),
                ToolCall(id="call-2", name="second", arguments={}),
            ]
            reason = StopReason.TOOL_USE
        else:
            content = [TextContent(text="done")]
            reason = StopReason.STOP
        response = AssistantMessage(
            content=content,
            api=current_model.api,
            provider=current_model.provider,
            model=current_model.id,
            stop_reason=reason,
            timestamp=provider_calls,
        )
        await stream.push(DoneEvent(message=response))
        await stream.end(response)
        return stream

    agent = Agent(AgentOptions(model=model, stream_fn=stream_fn, tools=tools))
    prompt_task = asyncio.create_task(agent.prompt("run"))
    await asyncio.wait_for(both_started.wait(), timeout=1)
    release.set()
    await asyncio.wait_for(prompt_task, timeout=1)

    tool_results = [message for message in agent.state.messages if message.role == "toolResult"]
    assert started == ["first", "second"]
    assert [message.tool_name for message in tool_results] == ["first", "second"]
    assert provider_calls == 2
