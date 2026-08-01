"""拆分后的 TUI 模块边界测试。"""

import pytest
from rich.console import Console

from pi.agent.types import AgentToolResult, ToolExecutionEndEvent
from pi.ai.types import TextContent
from pi.coding_agent.themes import Theme
from pi.tui.commands import build_command_names, validate_command_names
from pi.tui.formatting import format_tool_display, split_complete_markdown
from pi.tui.interactive import InteractiveSession as FacadeInteractiveSession
from pi.tui.rendering import AgentEventRenderer, MessageRenderState, summarize_tool_error
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


def test_long_tool_error_keeps_diagnostic_and_tail() -> None:
    lines = [
        "setup noise",
        "PermissionError: [WinError 5] Access is denied",
        *(f"detail {index}" for index in range(10)),
        "ERROR tests/test_tui.py::test_toolbar - PermissionError",
        "1 error in 0.48s",
        "Exit code: 1",
    ]

    preview = summarize_tool_error("\n".join(lines))

    assert preview.exit_code == "1"
    assert preview.lead == "PermissionError: [WinError 5] Access is denied"
    assert preview.tail[-2:] == (
        "ERROR tests/test_tui.py::test_toolbar - PermissionError",
        "1 error in 0.48s",
    )
    assert len(preview.tail) + 1 == 8
    assert preview.hidden_lines == 6


def test_tool_error_renderer_is_compact_and_treats_output_as_plain_text() -> None:
    console = Console(record=True, width=100, color_system=None)
    result = AgentToolResult(
        tool_call_id="call-1",
        tool_name="bash",
        content=[TextContent(text="STDERR:\nPermissionError: [WinError 5]\nExit code: 1")],
        is_error=True,
    )

    AgentEventRenderer().render(
        ToolExecutionEndEvent(
            tool_call_id="call-1",
            tool_name="bash",
            result=result,
        ),
        console,
        Theme(),
    )

    output = console.export_text()
    assert "! Bash failed (exit 1)" in output
    assert "PermissionError: [WinError 5]" in output
    assert "STDERR:" not in output
    assert "Exit code: 1" not in output
