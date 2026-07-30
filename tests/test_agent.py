"""Agent 运行时测试。"""

import asyncio

import pytest

from pi.agent.agent import Agent, AgentOptions
from pi.agent.types import (
    AgentTool,
    AgentToolCall,
    AgentToolResult,
    create_user_message,
)
from pi.ai.streaming import DoneEvent, EventStream
from pi.ai.types import AssistantMessage, Model, StopReason, TextContent


@pytest.mark.asyncio
async def test_create_user_message():
    msg = create_user_message("hello")
    assert msg.role == "user"
    assert msg.content == "hello"
    assert msg.timestamp > 0


@pytest.mark.asyncio
async def test_create_user_message_with_images():
    from pi.ai.types import ImageContent

    img = ImageContent(data="base64data", mime_type="image/png")
    msg = create_user_message("hello", [img])
    assert msg.role == "user"
    assert isinstance(msg.content, list)
    assert len(msg.content) == 2


@pytest.mark.asyncio
async def test_tool_execution():
    async def execute(call: AgentToolCall, ctx) -> AgentToolResult:
        return AgentToolResult(
            tool_call_id=call.id,
            tool_name=call.name,
            content=[TextContent(text=f"Result for {call.name}")],
        )

    tool = AgentTool(
        name="test_tool",
        description="A test tool",
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )

    call = AgentToolCall(id="call_1", name="test_tool", arguments={})
    result = await tool.execute(call, None)
    assert result.tool_name == "test_tool"
    assert result.content[0].text == "Result for test_tool"
    assert not result.is_error


@pytest.mark.asyncio
async def test_prompt_only_appears_once_in_provider_context():
    contexts = []
    event_types: list[str] = []
    model = Model(id="test", name="Test", api="test", provider="test")

    async def stream_fn(current_model, context, options):
        contexts.append(context)
        stream = EventStream()
        response = AssistantMessage(
            content=[TextContent(text="ok")],
            api=current_model.api,
            provider=current_model.provider,
            model=current_model.id,
            stop_reason=StopReason.STOP,
            timestamp=1,
        )
        await stream.push(DoneEvent(message=response))
        await stream.end(response)
        return stream

    async def listener(event):
        event_types.append(event.type)

    agent = Agent(AgentOptions(model=model, stream_fn=stream_fn))
    agent.subscribe(listener)
    await agent.prompt("hello")

    assert len(contexts) == 1
    assert len(contexts[0].messages) == 1
    assert contexts[0].messages[0].content == "hello"
    assert [message.role for message in agent.state.messages] == ["user", "assistant"]
    assert event_types.count("message_start") == 1


async def test_agent_passes_generation_options_to_provider():
    seen_options = []
    model = Model(
        id="test",
        name="Test",
        api="test",
        provider="test",
        context_window=10_000,
        max_tokens=2_000,
    )

    async def stream_fn(current_model, context, options):
        seen_options.append(options)
        stream = EventStream()
        response = AssistantMessage(
            content=[TextContent(text="ok")],
            api=current_model.api,
            provider=current_model.provider,
            model=current_model.id,
            stop_reason=StopReason.STOP,
            timestamp=1,
        )
        await stream.push(DoneEvent(message=response))
        await stream.end(response)
        return stream

    agent = Agent(
        AgentOptions(
            model=model,
            stream_fn=stream_fn,
            temperature=0.25,
            max_tokens=512,
        )
    )

    await agent.prompt("hello")

    assert seen_options[0].temperature == 0.25
    assert seen_options[0].max_tokens == 512
    assert agent.context_token_limit == 9_488


@pytest.mark.asyncio
async def test_abort_cancels_provider_task_and_records_terminal_message():
    started = asyncio.Event()
    producer_cancelled = asyncio.Event()
    model = Model(id="test", name="Test", api="test", provider="test")

    async def stream_fn(current_model, context, options):
        stream = EventStream()

        async def produce():
            started.set()
            try:
                await asyncio.Future()
            finally:
                producer_cancelled.set()

        stream.set_producer_task(asyncio.create_task(produce()))
        return stream

    agent = Agent(AgentOptions(model=model, stream_fn=stream_fn))
    prompt_task = asyncio.create_task(agent.prompt("wait"))
    await asyncio.wait_for(started.wait(), timeout=1)
    agent.abort()
    await asyncio.wait_for(prompt_task, timeout=1)

    assert producer_cancelled.is_set()
    assert not agent.is_busy
    assert agent.state.messages[-1].stop_reason == "aborted"
    assert agent.state.messages[-1].error_message == "Request aborted"
