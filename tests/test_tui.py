"""交互终端状态与命令测试。"""

from pi.ai.models import list_models
from pi.ai.oauth import CredentialStore, resolve_stored_api_key, save_api_key
from pi.tui.interactive import InteractiveSession


async def _unused_stream(model, context, options):
    raise AssertionError("stream should not be called")


async def test_model_command_updates_toolbar(tmp_path):
    models = list_models()
    initial = models[0]
    replacement = models[1]
    store = CredentialStore(tmp_path / "auth.json")
    await save_api_key(replacement.provider, "existing-key", store)
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


async def test_model_command_selects_model_and_prompts_for_api_key(tmp_path, monkeypatch):
    models = list_models()
    selected_model = next(model for model in models if model.provider == "deepseek")
    selected_index = models.index(selected_model) + 1
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("PIY_API_KEY_DEEPSEEK", raising=False)
    responses = iter([selected_index, "deepseek-secret"])
    prompt_calls = []

    def prompt(*args, **kwargs):
        prompt_calls.append((args, kwargs))
        return next(responses)

    monkeypatch.setattr("pi.tui.interactive.click.prompt", prompt)
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
    assert prompt_calls[1][1]["hide_input"] is True
