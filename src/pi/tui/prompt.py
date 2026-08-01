"""prompt-toolkit 输入框、状态栏与补全装配。"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.formatted_text.utils import fragment_list_width
from prompt_toolkit.input import DummyInput
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.styles import Style

from pi.agent.agent import Agent
from pi.tui.formatting import format_tokens
from pi.tui.history import SafeFileHistory, SafeInMemoryHistory

PROMPT_STYLE_RULES = {
    "input.prompt": "bold #e5e7eb",
    "input.placeholder": "italic #6b7280",
    "input.motion": "bold #67e8f9",
    "input.border": "#22d3ee",
    "input.queue.label": "bold #fbbf24",
    "input.queue.text": "#d1d5db",
    "input.queue.count": "#9ca3af",
    "input.brand": "bold #60a5fa",
    "input.separator": "#6b7280",
    "input.ready": "bold #a3e635",
    "input.working": "bold #fbbf24",
    "status.brand": "bold bg:#2563eb #ffffff",
    "status.model": "bold #22d3ee",
    "status.thinking": "#c084fc",
    "status.context": "#a3e635",
    "status.meta": "#9ca3af",
    "status.separator": "#525252",
}
PROMPT_STYLE_READY = Style.from_dict({**PROMPT_STYLE_RULES, "input.border": "#22d3ee"})
PROMPT_STYLE_WORKING = Style.from_dict({**PROMPT_STYLE_RULES, "input.border": "#f59e0b"})
READY_ANIMATION = ("·", "•", "·", " ")
WORKING_ANIMATION = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
ANIMATION_FPS = 8


def create_prompt_session(
    message,
    command_names: list[str],
    history_file: Path | None,
    erase_input_when: Callable[[str], bool] | None = None,
) -> PromptSession[str]:
    """创建带安全历史、命令补全和非交互回退的输入会话。"""
    history = SafeFileHistory(str(history_file)) if history_file else SafeInMemoryHistory()
    interactive_terminal = sys.stdin.isatty() and sys.stdout.isatty()
    session: PromptSession[str] = PromptSession(
        message=message,
        history=history,
        auto_suggest=AutoSuggestFromHistory(),
        completer=WordCompleter(sorted(command_names), sentence=True),
        complete_while_typing=True,
        placeholder=[("class:input.placeholder", "Type a message or /command")],
        show_frame=False,
        style=PROMPT_STYLE_READY,
        refresh_interval=1 / ANIMATION_FPS,
        input=None if interactive_terminal else DummyInput(),
        output=None if interactive_terminal else DummyOutput(),
    )
    if erase_input_when is not None:
        original_accept = session.default_buffer.accept_handler
        if original_accept is not None:

            def accept(buffer: Buffer) -> bool:
                session.app.erase_when_done = erase_input_when(buffer.text)
                return original_accept(buffer)

            session.default_buffer.accept_handler = accept
    return session


def build_input_prompt(
    status: StyleAndTextTuples,
    *,
    console_width: int,
    request_active: bool,
    timestamp: float,
    pending_follow_ups: Sequence[str] = (),
) -> StyleAndTextTuples:
    """构建状态轨道、待处理 follow-up 和开放式输入行。"""
    header: StyleAndTextTuples = [("", "\n"), ("class:input.border", "╭─"), *status]
    fill_width = max(1, console_width - fragment_list_width(header) - 1)
    header.append(("class:input.border", f"{'─' * fill_width}╮\n"))
    if pending_follow_ups:
        _append_follow_up_row(header, pending_follow_ups, console_width)
    frames = WORKING_ANIMATION if request_active else READY_ANIMATION
    frame = frames[int(timestamp * ANIMATION_FPS) % len(frames)]
    header.extend(
        [
            ("class:input.border", "╰─"),
            ("class:input.motion", f" {frame} "),
            ("class:input.prompt", "❯ "),
        ]
    )
    return header


def _append_follow_up_row(
    fragments: StyleAndTextTuples,
    pending_follow_ups: Sequence[str],
    console_width: int,
) -> None:
    """追加一行有界队列摘要，避免长文本改变输入区宽度。"""
    prefix = "│  ↳ follow-up  "
    count = f"  +{len(pending_follow_ups) - 1} more" if len(pending_follow_ups) > 1 else ""
    content_width = max(0, console_width - fragment_list_width([("", prefix + count + "│")]))
    text = _truncate_display_text(pending_follow_ups[0], content_width)
    used_width = fragment_list_width([("", prefix + text + count + "│")])
    padding = " " * max(0, console_width - used_width)
    fragments.extend(
        [
            ("class:input.border", "│  "),
            ("class:input.queue.label", "↳ follow-up  "),
            ("class:input.queue.text", text),
            ("class:input.queue.count", count),
            ("class:input.border", f"{padding}│\n"),
        ]
    )


def _truncate_display_text(text: str, max_width: int) -> str:
    """按终端显示宽度压缩单行文本。"""
    normalized = " ".join(text.split())
    if max_width <= 0:
        return ""
    if fragment_list_width([("", normalized)]) <= max_width:
        return normalized
    ellipsis = "…"
    ellipsis_width = fragment_list_width([("", ellipsis)])
    if max_width < ellipsis_width:
        return ""
    width = 0
    kept: list[str] = []
    for character in normalized:
        character_width = fragment_list_width([("", character)])
        if width + character_width + ellipsis_width > max_width:
            break
        kept.append(character)
        width += character_width
    return f"{''.join(kept)}{ellipsis}"


def build_input_status(
    agent: Agent,
    *,
    request_active: bool,
    session_id: str | None,
    cwd: Path,
    git_status: tuple[str, int] | None,
    console_width: int,
) -> StyleAndTextTuples:
    """按终端宽度生成输入框顶部状态轨道。"""
    model = agent.state.model
    model_name = model.name if model else "No model"
    session = session_id[:8] if session_id else "memory"
    context_usage = agent.get_context_usage()
    fragments: StyleAndTextTuples = [
        ("class:status.brand", " piY"),
        ("class:input.working" if request_active else "class:input.ready", " ●"),
    ]
    budget = max(16, console_width - 4)
    available_model_width = max(8, budget - fragment_list_width(fragments) - 2)
    if fragment_list_width([("", model_name)]) > available_model_width:
        model_name = f"{model_name[: available_model_width - 1]}…"

    items: dict[str, tuple[str, str]] = {
        "model": ("class:status.model", f"◇ {model_name}"),
        "session": ("class:status.meta", f"session {session}"),
        "cwd": ("class:status.meta", f"▱ {cwd}"),
    }
    if model is not None and model.reasoning:
        items["thinking"] = (
            "class:status.thinking",
            f"▣ {agent.thinking_level.value}",
        )
    if git_status is not None:
        branch, changes = git_status
        dirty = f" +{changes}" if changes else ""
        items["git"] = ("class:status.meta", f"○ {branch}{dirty}")
    if context_usage is not None:
        items["context"] = (
            "class:status.context",
            f"ctx {format_tokens(context_usage.tokens)}/"
            f"{format_tokens(context_usage.context_window)} "
            f"{context_usage.percent:.1f}%",
        )

    selected = {"model"}
    used_width = fragment_list_width(fragments) + 2 + fragment_list_width([items["model"]])
    for key in ("context", "thinking", "git", "cwd", "session"):
        item = items.get(key)
        if item is None:
            continue
        item_width = 2 + fragment_list_width([item])
        if used_width + item_width <= budget:
            selected.add(key)
            used_width += item_width

    for key in ("model", "thinking", "cwd", "git", "context", "session"):
        if key in selected:
            fragments.append(("class:status.separator", "  "))
            fragments.append(items[key])
    return fragments
