"""会话持久化与恢复测试。"""

from pi.agent.agent import Agent, AgentOptions
from pi.agent.session import JsonlStorage
from pi.agent.types import AgentAssistantMessage, create_user_message
from pi.ai.streaming import DoneEvent, EventStream
from pi.ai.types import AssistantMessage, Model, StopReason, TextContent


async def test_jsonl_storage_round_trip_and_trailing_record_recovery(tmp_path):
    path = tmp_path / "session.jsonl"
    storage = JsonlStorage(path)
    await storage.append_message(create_user_message("你好"))
    await storage.append_message(
        AgentAssistantMessage(
            content=[TextContent(text="世界")],
            api="test",
            provider="test",
            model="test-model",
            stop_reason="stop",
            timestamp=1,
        )
    )

    restored = JsonlStorage(path)
    branch = await restored.get_branch()
    assert [entry.message.role for entry in branch] == ["user", "assistant"]
    assert branch[0].message.content == "你好"
    assert branch[1].message.content[0].text == "世界"

    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"id":"partial"')

    recovered = JsonlStorage(path)
    assert len(await recovered.get_branch()) == 2


async def test_agent_restores_and_persists_session(tmp_path):
    path = tmp_path / "session.jsonl"
    storage = JsonlStorage(path)
    await storage.append_message(create_user_message("first"))
    model = Model(id="test-model", name="Test", api="test", provider="test")
    seen_contexts = []

    async def stream_fn(current_model, context, options):
        seen_contexts.append(context)
        stream = EventStream()
        response = AssistantMessage(
            content=[TextContent(text="done")],
            api=current_model.api,
            provider=current_model.provider,
            model=current_model.id,
            stop_reason=StopReason.STOP,
            timestamp=2,
        )
        await stream.push(DoneEvent(message=response))
        await stream.end(response)
        return stream

    agent = Agent(
        AgentOptions(
            model=model,
            stream_fn=stream_fn,
            session_storage=JsonlStorage(path),
        )
    )
    await agent.prompt("second")

    assert [message.content for message in seen_contexts[0].messages] == ["first", "second"]
    assert [message.role for message in agent.state.messages] == ["user", "user", "assistant"]

    restored = JsonlStorage(path)
    branch = await restored.get_branch()
    assert [entry.message.role for entry in branch] == ["user", "user", "assistant"]


async def test_jsonl_storage_persists_branches_and_model_selection(tmp_path):
    path = tmp_path / "session.jsonl"
    storage = JsonlStorage(path)
    root_id = await storage.append_message(create_user_message("root"))
    await storage.append_model_change("openai", "gpt-old")
    await storage.append_message(create_user_message("old branch"))

    await storage.branch_from(root_id)
    await storage.append_model_change("anthropic", "claude-new")
    await storage.append_message(create_user_message("new branch"))

    restored = JsonlStorage(path)

    assert [message.content for message in await restored.get_context_messages()] == [
        "root",
        "new branch",
    ]
    assert await restored.get_model_selection() == ("anthropic", "claude-new")
    assert len(await restored.get_entries()) == 6
