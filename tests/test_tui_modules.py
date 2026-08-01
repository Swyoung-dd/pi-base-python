"""拆分后的 TUI 模块边界测试。"""

import pytest

from pi.tui.commands import build_command_names, validate_command_names
from pi.tui.formatting import format_tool_display, split_complete_markdown
from pi.tui.interactive import InteractiveSession as FacadeInteractiveSession
from pi.tui.rendering import MessageRenderState
from pi.tui.session import InteractiveSession


def test_interactive_module_remains_a_compatibility_facade() -> None:
    assert FacadeInteractiveSession is InteractiveSession


def test_dynamic_commands_are_included_in_completion() -> None:
    commands = build_command_names(
        {"deploy"},
        {"release"},
        {"review"},
    )

    assert "/deploy" in commands
    assert "/skill:release" in commands
    assert "/review" in commands
    assert "/model" in commands
    assert commands == sorted(commands)


def test_dynamic_commands_cannot_shadow_registered_names() -> None:
    with pytest.raises(ValueError, match="Extension command conflicts: model"):
        validate_command_names({"model"}, set())
    with pytest.raises(ValueError, match="Prompt template command conflicts: deploy"):
        validate_command_names({"deploy"}, {"deploy"})


def test_markdown_split_waits_for_a_complete_fenced_block() -> None:
    complete, pending = split_complete_markdown("```python\nprint('piY')\n\n")
    assert complete == ""
    assert pending

    complete, pending = split_complete_markdown(f"{pending}```\n\n")
    assert complete.endswith("```\n\n")
    assert pending == ""


def test_tool_display_and_render_state_are_independent_of_session() -> None:
    assert format_tool_display("read", {"path": "README.md"}) == "read: README.md"
    state = MessageRenderState(
        current_text="answer",
        current_thinking="reasoning",
        rendered_text="answer",
        stream_buffer="pending",
        thinking_printed=True,
    )

    state.reset()

    assert state == MessageRenderState()
