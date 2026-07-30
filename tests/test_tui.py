"""交互终端状态与命令测试。"""

from pi.ai.models import list_models
from pi.tui.interactive import InteractiveSession


async def _unused_stream(model, context, options):
    raise AssertionError("stream should not be called")


async def test_model_command_updates_toolbar(tmp_path):
    models = list_models()
    initial = models[0]
    replacement = models[1]
    session = InteractiveSession(
        model=initial,
        system_prompt="",
        tools=[],
        stream_fn=_unused_stream,
        session_id="test-session",
        history_file=tmp_path / "history",
    )

    handled, should_exit = await session._handle_command(
        f"/model {replacement.provider}/{replacement.id}"
    )

    assert handled and not should_exit
    assert f"{replacement.provider}/{replacement.id}" in session._bottom_toolbar()
    assert "test-session" in session._bottom_toolbar()
