"""prompt-toolkit 输入框、状态栏与补全装配。"""

from __future__ import annotations

import sys
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
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
PROMPT_STYLE_WORKING = Style.from_dict(
    {**PROMPT_STYLE_RULES, "input.border": "#f59e0b"}
)
READY_ANIMATION = ("·", "•", "·", " ")
WORKING_ANIMATION = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
ANIMATION_FPS = 8


def create_prompt_session(
    message,
    command_names: list[str],
    history_file: Path | None,
) -> PromptSession[str]:
    """创建带安全历史、命令补全和非交互回退的输入会话。"""
    history = SafeFileHistory(str(history_file)) if history_file else SafeInMemoryHistory()
    interactive_terminal = sys.stdin.isatty() and sys.stdout.isatty()
    return PromptSession(
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


def build_input_prompt(
    status: StyleAndTextTuples,
    *,
    console_width: int,
    request_active: bool,
    timestamp: float,
) -> StyleAndTextTuples:
    """构建状态轨道和开放式输入行。"""
    header: StyleAndTextTuples = [("class:input.border", "╭─"), *status]
    fill_width = max(1, console_width - fragment_list_width(header) - 1)
    header.append(("class:input.border", f"{'─' * fill_width}╮\n"))
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

