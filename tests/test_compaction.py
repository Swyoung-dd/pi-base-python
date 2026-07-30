"""上下文压缩测试。"""

from pi.agent.agent import Agent, AgentOptions
from pi.agent.compaction import (
    apply_compaction_summary,
    compact_messages,
    estimate_context_tokens,
    estimate_context_tokens_with_overhead,
    estimate_message_tokens,
    estimate_messages_tokens,
    prepare_compaction,
)
from pi.agent.session import JsonlStorage
from pi.agent.types import AgentAssistantMessage, create_user_message
from pi.ai.streaming import DoneEvent, EventStream
from pi.ai.types import AssistantMessage, Model, StopReason, TextContent, Usage


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


def test_context_estimate_uses_last_valid_usage_and_trailing_messages():
    measured = _assistant("measured response", 1)
    measured.usage = Usage(input=80, output=20, total_tokens=100)
    aborted = _assistant("aborted response", 2)
    aborted.stop_reason = "aborted"
    aborted.usage = Usage(input=900, output=100, total_tokens=1_000)
    trailing = create_user_message("next question")

    result = estimate_context_tokens([measured, aborted, trailing])

    expected_trailing = estimate_message_tokens(aborted) + estimate_message_tokens(trailing)
    assert result.tokens == 100 + expected_trailing
    assert result.usage_tokens == 100
    assert result.trailing_tokens == expected_trailing
    assert result.last_usage_index == 0


def test_context_estimate_falls_back_to_all_messages_without_usage():
    messages = [create_user_message("first"), _assistant("second", 1)]

    result = estimate_context_tokens(messages)

    assert result.tokens == estimate_messages_tokens(messages)
    assert result.usage_tokens == 0
    assert result.last_usage_index is None


def test_context_estimate_ignores_usage_before_compaction_summary():
    messages = [
        create_user_message("old " + "x" * 100),
        _assistant("old response " + "y" * 100, 1),
        create_user_message("recent"),
        _assistant("recent response", 2),
    ]
    messages[-1].usage = Usage(input=900, output=100, total_tokens=1_000)
    plan = prepare_compaction(messages, target_tokens=40, original_tokens=1_000)

    compacted = apply_compaction_summary(plan, "summary")
    result = estimate_context_tokens(compacted.messages)

    assert result.usage_tokens == 0
    assert result.tokens == estimate_messages_tokens(compacted.messages)


def test_context_estimate_includes_system_prompt_before_first_provider_usage():
    messages = [create_user_message("short prompt")]

    tokens = estimate_context_tokens_with_overhead(
        messages,
        system_prompt="system rules " * 100,
        tools=[],
    )

    assert tokens > estimate_messages_tokens(messages) + 200


async def test_agent_persists_model_generated_compaction_checkpoint(tmp_path):
    model = Model(id="test-model", name="Test", api="test", provider="test")
    seen_contexts = []
    event_types: list[str] = []
    storage = JsonlStorage(tmp_path / "session.jsonl")
    for index in range(5):
        await storage.append_message(create_user_message(f"old-{index} " + "x" * 80))
        await storage.append_message(_assistant(f"reply-{index} " + "y" * 80, index))

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
            session_storage=storage,
            context_token_limit=80,
            compact_to_tokens=60,
        )
    )
    await agent.restore()
    original_count = len(agent.state.messages)
    agent.subscribe(listener)

    await agent.prompt("new prompt")

    assert seen_contexts[0].system_prompt.startswith("你负责压缩编码 agent")
    assert seen_contexts[1].messages[0].content.startswith("[已压缩的早期对话]\nok")
    assert len(agent.state.messages) < original_count + 2
    assert "context_compacted" in event_types

    restored = JsonlStorage(tmp_path / "session.jsonl")
    restored_messages = await restored.get_context_messages()
    assert restored_messages == agent.state.messages
    assert (await restored.get_branch())[-1].type == "compaction"
