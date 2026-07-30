"""终端内的紧凑键盘选择器。"""

from __future__ import annotations

from collections.abc import Sequence

from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.input import Input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.output import Output
from prompt_toolkit.styles import Style

_SELECTOR_STYLE = Style.from_dict(
    {
        "selector.title": "bold #60a5fa",
        "selector.item": "#9ca3af",
        "selector.selected": "reverse bold",
    }
)


async def select_option[T](
    title: str,
    options: Sequence[tuple[T, str]],
    *,
    default: T | None = None,
    input: Input | None = None,
    output: Output | None = None,
) -> T | None:
    """使用方向键选择选项，回车确认，Esc 或 Ctrl+C 取消。"""
    if not options:
        raise ValueError("选择器至少需要一个选项")

    selected_index = next(
        (index for index, (value, _) in enumerate(options) if value == default),
        0,
    )
    bindings = KeyBindings()

    def render_options() -> StyleAndTextTuples:
        fragments: StyleAndTextTuples = [("class:selector.title", f" {title}\n")]
        for index, (_, label) in enumerate(options):
            if index == selected_index:
                fragments.append(("class:selector.selected", f" > {label} "))
            else:
                fragments.append(("class:selector.item", f"   {label} "))
            if index < len(options) - 1:
                fragments.append(("", "\n"))
        return fragments

    @bindings.add("up")
    def move_up(event: KeyPressEvent) -> None:
        nonlocal selected_index
        selected_index = (selected_index - 1) % len(options)
        event.app.invalidate()

    @bindings.add("down")
    def move_down(event: KeyPressEvent) -> None:
        nonlocal selected_index
        selected_index = (selected_index + 1) % len(options)
        event.app.invalidate()

    @bindings.add("home")
    def move_first(event: KeyPressEvent) -> None:
        nonlocal selected_index
        selected_index = 0
        event.app.invalidate()

    @bindings.add("end")
    def move_last(event: KeyPressEvent) -> None:
        nonlocal selected_index
        selected_index = len(options) - 1
        event.app.invalidate()

    @bindings.add("enter")
    def confirm(event: KeyPressEvent) -> None:
        event.app.exit(result=options[selected_index][0])

    @bindings.add("escape", eager=True)
    @bindings.add("c-c")
    def cancel(event: KeyPressEvent) -> None:
        event.app.exit(result=None)

    control = FormattedTextControl(render_options, focusable=True)
    application: Application[T | None] = Application(
        layout=Layout(
            Window(
                content=control,
                height=Dimension.exact(len(options) + 1),
                always_hide_cursor=True,
                dont_extend_width=True,
            ),
            focused_element=control,
        ),
        key_bindings=bindings,
        style=_SELECTOR_STYLE,
        full_screen=False,
        erase_when_done=True,
        input=input,
        output=output,
    )
    return await application.run_async()
