"""交互终端状态与命令测试。"""

from pi.agent.types import AgentAssistantMessage
from pi.ai.models import list_models
from pi.ai.oauth import CredentialStore, resolve_stored_api_key, save_api_key
from pi.ai.types import TextContent, Usage
from pi.tui.interactive import InteractiveSession, _format_tokens, _SafeFileHistory


async def _unused_stream(model, context, options):
    raise AssertionError("stream should not be called")


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
    assert f"{replacement.provider}/{replacement.id}" in session._bottom_toolbar()
    assert "test-session" in session._bottom_toolbar()
    assert selected_models == [replacement]
    assert await resolve_stored_api_key(replacement.provider, store) == "new-key"


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

    toolbar = session._bottom_toolbar()

    assert f"ctx 10k/{_format_tokens(model.context_window)}" in toolbar
    assert f"({10_000 / model.context_window * 100:.1f}%)" in toolbar


async def test_model_command_selects_model_and_prompts_for_api_key(tmp_path, monkeypatch):
    models = list_models()
    selected_model = next(model for model in models if model.provider == "deepseek")
    selected_index = models.index(selected_model) + 1
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("PIY_API_KEY_DEEPSEEK", raising=False)
    monkeypatch.setattr(
        "pi.tui.interactive.click.prompt",
        lambda *args, **kwargs: selected_index,
    )
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
    assert f"deepseek/{selected_model.id}" in session._bottom_toolbar()
    assert await resolve_stored_api_key("deepseek", store) == "deepseek-secret"
    assert prompt_messages == ["deepseek API key: "]


def test_file_history_does_not_store_likely_api_keys(tmp_path):
    path = tmp_path / "history"
    history = _SafeFileHistory(str(path))

    history.store_string("normal prompt")
    history.store_string("sk-" + "a" * 32)

    content = path.read_text(encoding="utf-8")
    assert "normal prompt" in content
    assert "sk-" not in content
