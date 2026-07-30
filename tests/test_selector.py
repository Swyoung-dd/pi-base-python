"""终端键盘选择器测试。"""

from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from pi.tui.selector import select_option


async def test_selector_uses_arrow_keys_and_enter() -> None:
    with create_pipe_input() as pipe_input:
        pipe_input.send_text("\x1b[B\r")

        selected = await select_option(
            "Select",
            [("first", "First"), ("second", "Second")],
            input=pipe_input,
            output=DummyOutput(),
        )

    assert selected == "second"


async def test_selector_uses_default_and_escape_cancels() -> None:
    with create_pipe_input() as pipe_input:
        pipe_input.send_text("\r")
        selected = await select_option(
            "Select",
            [("first", "First"), ("second", "Second")],
            default="second",
            input=pipe_input,
            output=DummyOutput(),
        )

    assert selected == "second"

    with create_pipe_input() as pipe_input:
        pipe_input.send_text("\x03")
        cancelled = await select_option(
            "Select",
            [("first", "First")],
            input=pipe_input,
            output=DummyOutput(),
        )

    assert cancelled is None
