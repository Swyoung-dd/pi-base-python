"""交互终端状态与命令测试。"""

from unittest.mock import Mock

import pytest
from prompt_toolkit.formatted_text import fragment_list_to_text, to_formatted_text
from rich.console import Console, Group
from rich.panel import Panel

from pi.agent.session import JsonlStorage
from pi.agent.types import (
    AgentAssistantMessage,
    MessageEndEvent,
    MessageStartEvent,
    TextDeltaUpdateEvent,
    ThinkingDeltaUpdateEvent,
    TurnEndEvent,
    create_user_message,
)
from pi.ai.models import list_models
from pi.ai.oauth import CredentialStore, resolve_stored_api_key, save_api_key
from pi.ai.types import ModelThinkingLevel, TextContent, ThinkingContent, ToolCall, Usage
from pi.coding_agent.extensions import ExtensionContext
from pi.coding_agent.prompt_templates import PromptTemplate
from pi.coding_agent.themes import Theme
from pi.tui.interactive import InteractiveSession, _format_tokens, _SafeFileHistory


async def _unused_stream(model, context, options):
    raise AssertionError("stream should not be called")


def _toolbar_text(session: InteractiveSession) -> str:
    return fragment_list_to_text(to_formatted_text(session._bottom_toolbar()))


async def test_model_command_updates_toolbar(tmp_path, monkeypatch):
    models = list_models()
    initial = models[0]
    replacement = models[1]
    store = CredentialStore(tmp_path / "auth.json")
    await save_api_key(replacement.provider, "existing-key", store)

    async def prompt_api_key(message):
        return "new-key"

    monkeypatch.setattr("pi.coding_agent.model_auth._prompt_api_key", prompt_api_key)
    selected_models = []
    session = InteractiveSession(
        model=initial,
        system_prompt="",
        tools=[],
        stream_fn=_unused_stream,
        session_id="test-session",
        history_file=tmp_path / "history",
        credential_store=store,
        on_model_selected=selected_models.append,
    )

    handled, should_exit = await session._handle_command(
        f"/model {replacement.provider}/{replacement.id}"
    )

    assert handled and not should_exit
    session._console = Console(width=120)
    assert replacement.name in _toolbar_text(session)
    assert "test-ses" in _toolbar_text(session)
    assert selected_models == [replacement]
    assert await resolve_stored_api_key(replacement.provider, store) == "new-key"


async def test_thinking_command_updates_toolbar_and_persists_selection(tmp_path):
    model = next(candidate for candidate in list_models() if candidate.reasoning)
    selected_levels = []
    session = InteractiveSession(
        model=model,
        system_prompt="",
        tools=[],
        stream_fn=_unused_stream,
        history_file=tmp_path / "history",
        on_thinking_selected=selected_levels.append,
    )

    handled, should_exit = await session._handle_command("/thinking high")

    assert handled and not should_exit
    assert session._agent.thinking_level.value == "high"
    assert "thinking high" in _toolbar_text(session)
    assert [level.value for level in selected_levels] == ["high"]


async def test_thinking_command_uses_keyboard_selector(tmp_path, monkeypatch):
    model = next(candidate for candidate in list_models() if candidate.reasoning)

    async def choose(title, options, **kwargs):
        assert title == "Thinking level"
        return ModelThinkingLevel.HIGH

    monkeypatch.setattr("pi.tui.interactive.select_option", choose)
    session = InteractiveSession(
        model=model,
        system_prompt="",
        tools=[],
        stream_fn=_unused_stream,
        history_file=tmp_path / "history",
    )

    handled, should_exit = await session._handle_command("/thinking")

    assert handled and not should_exit
    assert session._agent.thinking_level is ModelThinkingLevel.HIGH


async def test_model_command_is_restored_from_session(tmp_path, monkeypatch):
    models = list_models()
    initial = models[0]
    replacement = models[1]
    store = CredentialStore(tmp_path / "auth.json")
    await save_api_key(replacement.provider, "existing-key", store)
    monkeypatch.setattr(
        "pi.coding_agent.model_auth._prompt_api_key",
        lambda message: _empty_api_key(),
    )
    storage = JsonlStorage(tmp_path / "session.jsonl")
    await storage.append_message(create_user_message("hello"))
    session = InteractiveSession(
        model=initial,
        system_prompt="",
        tools=[],
        stream_fn=_unused_stream,
        session_id="test-session",
        session_storage=storage,
        history_file=tmp_path / "history",
        credential_store=store,
    )

    await session._handle_command(f"/model {replacement.provider}/{replacement.id}")

    restored = InteractiveSession(
        model=initial,
        system_prompt="",
        tools=[],
        stream_fn=_unused_stream,
        session_id="test-session",
        session_storage=JsonlStorage(tmp_path / "session.jsonl"),
        history_file=tmp_path / "history",
        credential_store=store,
    )
    await restored._agent.restore()
    await restored._restore_session_model()

    assert replacement.name in _toolbar_text(restored)


async def test_new_session_records_current_model(tmp_path):
    model = list_models()[0]
    sessions_dir = tmp_path / "sessions"
    session = InteractiveSession(
        model=model,
        system_prompt="",
        tools=[],
        stream_fn=_unused_stream,
        sessions_dir=sessions_dir,
        history_file=tmp_path / "history",
    )

    handled, should_exit = await session._handle_command("/new")

    assert handled and not should_exit
    assert session._session_storage is not None
    assert await session._session_storage.get_model_selection() == (
        model.provider,
        model.id,
    )


async def _empty_api_key():
    return ""


def test_toolbar_shows_context_usage_and_percentage(tmp_path):
    model = list_models()[0]
    session = InteractiveSession(
        model=model,
        system_prompt="",
        tools=[],
        stream_fn=_unused_stream,
        history_file=tmp_path / "history",
    )
    session._agent.state.messages.append(
        AgentAssistantMessage(
            content=[TextContent(text="answer")],
            usage=Usage(input=9_000, output=1_000, total_tokens=10_000),
            stop_reason="stop",
        )
    )

    toolbar = _toolbar_text(session)

    assert f"ctx 10k/{_format_tokens(model.context_window)}" in toolbar
    assert f"{10_000 / model.context_window * 100:.1f}%" in toolbar


def test_toolbar_prioritizes_core_status_on_narrow_terminal(tmp_path):
    session = InteractiveSession(
        model=list_models()[0],
        system_prompt="",
        tools=[],
        stream_fn=_unused_stream,
        session_id="long-session-id",
        history_file=tmp_path / "history",
    )
    session._console = Console(width=60)

    toolbar = _toolbar_text(session)

    assert len(toolbar.splitlines()[0]) <= 60
    assert "ctx " in toolbar
    assert "session" not in toolbar
    assert str(tmp_path) not in toolbar


def test_interactive_session_accepts_custom_theme(tmp_path):
    theme = Theme(name="test", primary="cyan", muted="white")

    session = InteractiveSession(
        model=list_models()[0],
        system_prompt="",
        tools=[],
        stream_fn=_unused_stream,
        theme=theme,
        history_file=tmp_path / "history",
    )

    assert session._theme is theme


def test_busy_session_queues_prompt_as_steering(tmp_path, monkeypatch):
    session = InteractiveSession(
        model=list_models()[0],
        system_prompt="",
        tools=[],
        stream_fn=_unused_stream,
        history_file=tmp_path / "history",
    )
    queued = []
    session._agent._idle_event.clear()
    monkeypatch.setattr(session._agent, "steer", queued.append)

    session._submit_agent_prompt("correction")

    assert len(queued) == 1
    assert queued[0].content == "correction"


async def test_prompt_template_command_submits_expanded_prompt(tmp_path, monkeypatch):
    template = PromptTemplate(
        name="review",
        description="Review files",
        content="Review $ARGUMENTS",
        file_path=tmp_path / "review.md",
    )
    session = InteractiveSession(
        model=list_models()[0],
        system_prompt="",
        tools=[],
        stream_fn=_unused_stream,
        prompt_templates=[template],
        history_file=tmp_path / "history",
    )
    prompts = []
    monkeypatch.setattr(session, "_submit_agent_prompt", prompts.append)

    handled, should_exit = await session._handle_command("/review src/pi")

    assert handled and not should_exit
    assert prompts == ["Review src/pi"]


def test_extension_commands_cannot_shadow_builtin_commands(tmp_path):
    context = ExtensionContext()
    context.add_command("model", lambda argument, agent: None)

    with pytest.raises(ValueError, match="Extension command conflicts: model"):
        InteractiveSession(
            model=list_models()[0],
            system_prompt="",
            tools=[],
            stream_fn=_unused_stream,
            commands=context.commands,
            extension_context=context,
            history_file=tmp_path / "history",
        )


async def test_model_command_selects_model_and_prompts_for_api_key(tmp_path, monkeypatch):
    models = list_models()
    selected_model = next(model for model in models if model.provider == "deepseek")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("PIY_API_KEY_DEEPSEEK", raising=False)

    async def choose(title, options, **kwargs):
        assert title == "Select model"
        return selected_model

    monkeypatch.setattr("pi.tui.interactive.select_option", choose)
    prompt_messages = []

    async def prompt_api_key(message):
        prompt_messages.append(message)
        return "deepseek-secret"

    monkeypatch.setattr("pi.coding_agent.model_auth._prompt_api_key", prompt_api_key)
    store = CredentialStore(tmp_path / "auth.json")
    session = InteractiveSession(
        model=models[0],
        system_prompt="",
        tools=[],
        stream_fn=_unused_stream,
        history_file=tmp_path / "history",
        credential_store=store,
    )

    handled, should_exit = await session._handle_command("/model")

    assert handled and not should_exit
    assert selected_model.name in _toolbar_text(session)
    assert await resolve_stored_api_key("deepseek", store) == "deepseek-secret"
    assert prompt_messages == ["deepseek API key: "]


def test_prompt_uses_framed_input_and_placeholder(tmp_path):
    session = InteractiveSession(
        model=list_models()[0],
        system_prompt="",
        tools=[],
        stream_fn=_unused_stream,
        history_file=tmp_path / "history",
    )

    assert session._prompt_session.show_frame is True
    assert session._prompt_session.placeholder


def test_file_history_does_not_store_likely_api_keys(tmp_path):
    path = tmp_path / "history"
    history = _SafeFileHistory(str(path))

    history.store_string("normal prompt")
    history.store_string("sk-" + "a" * 32)

    content = path.read_text(encoding="utf-8")
    assert "normal prompt" in content
    assert "sk-" not in content


async def test_tool_only_message_renders_grouped_tree(tmp_path):
    session = InteractiveSession(
        model=list_models()[0],
        system_prompt="",
        tools=[],
        stream_fn=_unused_stream,
        history_file=tmp_path / "history",
    )
    session._console.print = Mock()
    message = AgentAssistantMessage(
        content=[ToolCall(id="call-1", name="read", arguments={"path": "README.md"})]
    )

    await session._on_event(MessageStartEvent(message=message))
    await session._on_event(MessageEndEvent(message=message))

    session._console.print.assert_called_once()
    renderable = session._console.print.call_args.args[0]
    assert isinstance(renderable, Group)
    console = Console(record=True, width=80)
    console.print(renderable)
    output = console.export_text()
    assert "Read" in output
    assert "README.md" in output


async def test_non_streaming_text_message_is_rendered_once(tmp_path):
    session = InteractiveSession(
        model=list_models()[0],
        system_prompt="",
        tools=[],
        stream_fn=_unused_stream,
        history_file=tmp_path / "history",
    )
    session._console.print = Mock()
    message = AgentAssistantMessage(content=[TextContent(text="answer")])

    await session._on_event(MessageStartEvent(message=message))
    await session._on_event(MessageEndEvent(message=message))

    session._console.print.assert_called_once()
    assert isinstance(session._console.print.call_args.args[0], Group)


async def test_streaming_answer_is_appended_to_body_without_toolbar_preview(tmp_path):
    session = InteractiveSession(
        model=list_models()[0],
        system_prompt="",
        tools=[],
        stream_fn=_unused_stream,
        history_file=tmp_path / "history",
    )
    session._console.print = Mock()
    message = AgentAssistantMessage(content=[TextContent(text="final answer")])

    await session._on_event(MessageStartEvent(message=message))
    await session._on_event(TextDeltaUpdateEvent(delta="partial"))

    assert "partial" not in _toolbar_text(session)
    session._console.print.assert_called_once_with(
        "partial",
        end="",
        markup=False,
        highlight=False,
    )

    await session._on_event(MessageEndEvent(message=message))

    assert session._console.print.call_count == 2
    session._console.print.assert_called_with()


async def test_thinking_precedes_streaming_answer_in_body(tmp_path):
    session = InteractiveSession(
        model=list_models()[0],
        system_prompt="",
        tools=[],
        stream_fn=_unused_stream,
        history_file=tmp_path / "history",
    )
    session._console = Console(record=True, width=80)
    message = AgentAssistantMessage(
        content=[
            ThinkingContent(thinking="checking project files"),
            TextContent(text="final answer"),
        ]
    )

    await session._on_event(MessageStartEvent(message=message))
    await session._on_event(ThinkingDeltaUpdateEvent(delta="checking project files"))
    await session._on_event(TextDeltaUpdateEvent(delta="final answer"))
    await session._on_event(MessageEndEvent(message=message))

    output = session._console.export_text()
    assert output.index("checking project files") < output.index("final answer")
    assert output.count("final answer") == 1


async def test_thinking_is_rendered_in_body_before_tool_calls(tmp_path):
    session = InteractiveSession(
        model=list_models()[0],
        system_prompt="",
        tools=[],
        stream_fn=_unused_stream,
        history_file=tmp_path / "history",
    )
    session._console.print = Mock()
    message = AgentAssistantMessage(
        content=[
            ThinkingContent(thinking="checking project files"),
            ToolCall(id="call-1", name="read", arguments={"path": "README.md"}),
        ]
    )

    await session._on_event(ThinkingDeltaUpdateEvent(delta="checking project files"))
    assert "checking project files" not in _toolbar_text(session)

    await session._on_event(MessageEndEvent(message=message))

    renderable = session._console.print.call_args.args[0]
    console = Console(record=True, width=80)
    console.print(renderable)
    output = console.export_text()
    assert output.index("checking project files") < output.index("Read")
    assert output.index("Read") < output.index("README.md")


def test_welcome_panel_shows_model_actions_and_recent_sessions(tmp_path):
    session = InteractiveSession(
        model=list_models()[0],
        system_prompt="",
        tools=[],
        stream_fn=_unused_stream,
        cwd=tmp_path,
        history_file=tmp_path / "history",
    )
    session._console = Console(record=True, width=120)

    panel = session._welcome_panel("0.1.0", [])
    session._console.print(panel)
    output = session._console.export_text()

    assert isinstance(panel, Panel)
    assert "piY v0.1.0" in output
    assert "Quick actions" in output
    assert "/model" in output
    assert "No recent sessions" in output
    assert f"workspace {tmp_path.name}" in output


def test_welcome_panel_fits_narrow_terminal(tmp_path):
    session = InteractiveSession(
        model=list_models()[0],
        system_prompt="",
        tools=[],
        stream_fn=_unused_stream,
        history_file=tmp_path / "history",
    )
    session._console = Console(width=60)

    panel = session._welcome_panel("0.1.0", [])

    assert panel.width == 58


async def test_turn_usage_is_aggregated_before_display(tmp_path):
    session = InteractiveSession(
        model=list_models()[0],
        system_prompt="",
        tools=[],
        stream_fn=_unused_stream,
        history_file=tmp_path / "history",
    )
    session._console.print = Mock()

    await session._on_event(
        TurnEndEvent(
            message=AgentAssistantMessage(
                usage=Usage(input=100, output=20, total_tokens=120)
            )
        )
    )
    await session._on_event(
        TurnEndEvent(
            message=AgentAssistantMessage(
                usage=Usage(input=200, output=30, cache_read=50, total_tokens=230)
            )
        )
    )

    session._console.print.assert_not_called()
    session._print_request_usage()
    session._console.print.assert_called_once_with(
        "300 in / 50 out / 350 total tokens / 50 cached",
        style=session._theme.muted,
    )
