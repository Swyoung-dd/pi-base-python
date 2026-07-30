"""上下文压缩测试。"""

from pi.agent.agent import Agent, AgentOptions
from pi.agent.compaction import compact_messages, estimate_messages_tokens
from pi.agent.types import AgentAssistantMessage, create_user_message
from pi.ai.streaming import DoneEvent, EventStream
from pi.ai.types import AssistantMessage, Model, StopReason, TextContent


def _assistant(text: str, timestamp: int) -> AgentAssistantMessage:
    return AgentAssistantMessage(
        content=[TextContent(text=text)],
        api="test",
        provider="test",
        model="test-model",
        stop_reason="stop",
        timestamp=timestamp,
    )


def test_compaction_preserves_recent_turn_and_full_input():
    messages = []
    for index in range(6):
        messages.extend(
            [
                create_user_message(f"question-{index} " + "x" * 80),
                _assistant(f"answer-{index} " + "y" * 80, index),
            ]
        )

    result = compact_messages(messages, target_tokens=90)

    assert result.dropped_messages > 0
    assert result.messages[0].content.startswith("[已压缩的早期对话]")
    assert result.messages[-2:] == messages[-2:]
    assert result.compacted_tokens < result.original_tokens
    assert estimate_messages_tokens(messages) == result.original_tokens


async def test_agent_compacts_provider_view_without_mutating_history():
    model = Model(id="test-model", name="Test", api="test", provider="test")
    seen_contexts = []
    event_types: list[str] = []

    async def stream_fn(current_model, context, options):
        seen_contexts.append(context)
        stream = EventStream()
        response = AssistantMessage(
            content=[TextContent(text="ok")],
            api=current_model.api,
            provider=current_model.provider,
            model=current_model.id,
            stop_reason=StopReason.STOP,
            timestamp=99,
        )
        await stream.push(DoneEvent(message=response))
        await stream.end(response)
        return stream

    async def listener(event):
        event_types.append(event.type)

    agent = Agent(
        AgentOptions(
            model=model,
            stream_fn=stream_fn,
            context_token_limit=80,
            compact_to_tokens=60,
        )
    )
    for index in range(5):
        agent.state.messages.extend(
            [
                create_user_message(f"old-{index} " + "x" * 80),
                _assistant(f"reply-{index} " + "y" * 80, index),
            ]
        )
    original_count = len(agent.state.messages)
    agent.subscribe(listener)

    await agent.prompt("new prompt")

    assert seen_contexts[0].messages[0].content.startswith("[已压缩的早期对话]")
    assert len(agent.state.messages) == original_count + 2
    assert "context_compacted" in event_types
