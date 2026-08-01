"""Rich 欢迎界面与 Agent 事件渲染。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from rich import box
from rich.console import Console, Group, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from pi.agent.types import AgentEvent
from pi.ai.types import Model, ModelThinkingLevel, TextContent, ThinkingContent, ToolCall, Usage
from pi.coding_agent.sessions import SessionInfo
from pi.coding_agent.themes import Theme
from pi.tui.formatting import format_tool_target, split_complete_markdown

PIY_LOGO = """       _ __   __
 _ __ (_)\\ \\ / /
| '_ \\| | \\ V /
| |_) | |  | |
| .__/|_|  |_|
|_|"""

_TOOL_LABELS = {
    "bash": "Bash",
    "edit": "Edit",
    "find": "Find",
    "grep": "Grep",
    "ls": "List",
    "read": "Read",
    "subagent": "Subagent",
    "write": "Write",
}

_TOOL_ERROR_PREVIEW_LINES = 8
_EXIT_CODE_RE = re.compile(r"^\s*Exit code:\s*(-?\d+)\s*$", re.IGNORECASE)
_DIAGNOSTIC_RE = re.compile(
    r"(?:\b(?:fatal|failed|error)\b|\b[\w.]+(?:Error|Exception)\b)",
    re.IGNORECASE,
)


@dataclass
class MessageRenderState:
    """跨流事件保存尚未完整渲染的消息片段。"""

    current_text: str = ""
    current_thinking: str = ""
    rendered_text: str = ""
    stream_buffer: str = ""
    thinking_printed: bool = False

    def reset(self) -> None:
        self.current_text = ""
        self.current_thinking = ""
        self.rendered_text = ""
        self.stream_buffer = ""
        self.thinking_printed = False


@dataclass(frozen=True)
class RenderOutcome:
    """渲染事件后需要由会话层执行的副作用。"""

    refresh_prompt: bool = False
    usage: Usage | None = None


@dataclass(frozen=True)
class ToolErrorPreview:
    """工具失败结果的紧凑展示数据。"""

    exit_code: str | None
    lead: str | None
    tail: tuple[str, ...]
    hidden_lines: int = 0


def summarize_tool_error(text: str, max_lines: int = _TOOL_ERROR_PREVIEW_LINES) -> ToolErrorPreview:
    """提取退出码、关键异常和末尾摘要，避免长错误淹没对话。"""
    exit_code = None
    lines: list[str] = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        line = raw_line.rstrip()
        match = _EXIT_CODE_RE.match(line)
        if match:
            exit_code = match.group(1)
        elif line.strip() and line.strip().upper() != "STDERR:":
            lines.append(line)

    if len(lines) <= max_lines:
        return ToolErrorPreview(exit_code, None, tuple(lines))

    tail_size = max(1, max_lines - 1)
    prefix = lines[:-tail_size]
    lead = next(
        (
            line
            for line in reversed(prefix)
            if _DIAGNOSTIC_RE.search(line) and "warning" not in line.lower()
        ),
        None,
    )
    if lead is None:
        tail_size = max_lines
    tail = tuple(lines[-tail_size:])
    hidden_lines = len(lines) - len(tail) - (1 if lead is not None else 0)
    return ToolErrorPreview(exit_code, lead, tail, hidden_lines)


def build_tool_error_renderable(
    tool_name: str,
    text: str,
    theme: Theme,
) -> RenderableType:
    """构建层级清晰且不会解析工具输出为 Rich markup 的失败摘要。"""
    preview = summarize_tool_error(text)
    label = _TOOL_LABELS.get(tool_name, tool_name)
    title = Text.assemble(("! ", theme.error), (f"{label} failed", theme.error))
    if preview.exit_code is not None:
        title.append(f" (exit {preview.exit_code})", style=theme.muted)

    details = Text(style=theme.muted)
    if preview.lead is not None:
        details.append(f"  {preview.lead}\n")
    if preview.hidden_lines:
        details.append(f"  ... {preview.hidden_lines} earlier lines hidden\n")
    for index, line in enumerate(preview.tail):
        details.append(f"  {line}")
        if index < len(preview.tail) - 1:
            details.append("\n")
    return Group(title, details) if details.plain else title


def build_tool_trees(tool_calls: list[ToolCall], theme: Theme) -> list[Tree]:
    """把同一轮工具调用按工具名称分组为紧凑树形结构。"""
    groups: dict[str, list[ToolCall]] = {}
    for call in tool_calls:
        groups.setdefault(call.name, []).append(call)

    trees = []
    for tool_name, calls in groups.items():
        count = f" ({len(calls)})" if len(calls) > 1 else ""
        title = Text.assemble(
            ("> ", theme.success),
            (_TOOL_LABELS.get(tool_name, tool_name), theme.primary),
            (count, theme.muted),
        )
        tree = Tree(title, guide_style=theme.muted)
        for call in calls:
            tree.add(Text(format_tool_target(call.name, call.arguments)))
        trees.append(tree)
    return trees


def build_message_renderable(
    state: MessageRenderState,
    tool_calls: list[ToolCall],
    theme: Theme,
) -> RenderableType | None:
    """创建轻量消息正文，避免每一轮内容都被边框切割。"""
    content: list[RenderableType] = []
    if state.current_thinking:
        content.append(Text(state.current_thinking, style=theme.thinking))
    if state.current_text:
        content.append(Markdown(state.current_text))
    content.extend(build_tool_trees(tool_calls, theme))
    return Group(*content) if content else None


def build_welcome_panel(
    version: str,
    sessions: list[SessionInfo],
    *,
    model: Model | None,
    thinking_level: ModelThinkingLevel,
    session_id: str | None,
    cwd: Path,
    theme: Theme,
    console_width: int,
) -> Panel:
    """构建响应式欢迎面板。"""
    brand = Text(PIY_LOGO, style=theme.primary)
    brand.append("\n\n")
    brand.append(model.name if model else "No model", style="bold")
    if model is not None:
        brand.append(f"\n{model.provider}", style=theme.muted)
    if model is not None and model.reasoning:
        brand.append(f"  thinking {thinking_level.value}", style=theme.thinking)

    actions = Text()
    actions.append("Quick actions\n", style=theme.primary)
    for command, description in (
        ("/help", "Commands"),
        ("/model", "Switch model"),
        ("/thinking", "Reasoning level"),
        ("/sessions", "Session history"),
    ):
        actions.append(f"{command:<12}", style=theme.primary)
        actions.append(f"{description}\n", style=theme.muted)

    recent = Text()
    recent.append("\nRecent sessions\n", style=theme.primary)
    visible_sessions = [item for item in sessions if item.session_id != session_id][:3]
    if not visible_sessions:
        recent.append("No recent sessions", style=theme.muted)
    for item in visible_sessions:
        preview = item.preview or "Empty session"
        recent.append(f"{item.session_id[:8]}  ", style=theme.primary)
        recent.append(f"{preview}\n", style=theme.muted)

    details = Group(actions, recent)
    grid = Table.grid(expand=True, padding=(0, 2))
    if console_width >= 90:
        grid.add_column(width=24)
        grid.add_column(ratio=1)
        grid.add_row(brand, details)
    else:
        grid.add_column(ratio=1)
        grid.add_row(brand)
        grid.add_row(details)

    return Panel(
        grid,
        title=f" piY v{version} ",
        subtitle=f" workspace {cwd.name} ",
        subtitle_align="left",
        border_style=theme.primary,
        box=box.SQUARE,
        width=min(96, max(20, console_width - 2)),
    )


class AgentEventRenderer:
    """把 Agent 生命周期事件转换为 Rich 输出。"""

    def __init__(self) -> None:
        self.state = MessageRenderState()

    def render(self, event: AgentEvent, console: Console, theme: Theme) -> RenderOutcome:
        if event.type == "message_start":
            self.state.reset()
        elif event.type == "text_delta":
            if self.state.current_thinking and not self.state.thinking_printed:
                console.print(Text(self.state.current_thinking, style=theme.thinking))
                self.state.thinking_printed = True
            self.state.current_text += event.delta
            self.state.stream_buffer += event.delta
            complete, self.state.stream_buffer = split_complete_markdown(self.state.stream_buffer)
            if complete:
                console.print(Markdown(complete))
                self.state.rendered_text += complete
        elif event.type == "thinking_delta":
            self.state.current_thinking += event.delta
        elif event.type == "message_end":
            self._render_message_end(event, console, theme)
            return RenderOutcome(refresh_prompt=True)
        elif event.type == "tool_execution_start":
            self.state.current_text = ""
            self.state.current_thinking = ""
            return RenderOutcome(refresh_prompt=True)
        elif event.type == "tool_execution_end":
            if event.result and event.result.is_error:
                text = "\n".join(
                    block.text for block in event.result.content if isinstance(block, TextContent)
                )
                console.print(build_tool_error_renderable(event.tool_name, text, theme))
        elif event.type == "turn_end":
            if event.message.stop_reason == "aborted":
                console.print("Aborted.", style=theme.muted)
            elif event.message.error_message:
                console.print(f"Error: {event.message.error_message}", style=theme.error)
            return RenderOutcome(usage=event.message.usage)
        elif event.type == "provider_retry":
            console.print(
                f"Retry {event.attempt}/{event.max_retries} in {event.delay_ms} ms",
                style=theme.warning,
            )
        elif event.type == "context_compacted":
            console.print(
                f"Context compacted: {event.original_tokens} -> {event.compacted_tokens} tokens",
                style=theme.muted,
            )
        return RenderOutcome()

    def _render_message_end(self, event: AgentEvent, console: Console, theme: Theme) -> None:
        final_text = "\n".join(
            block.text
            for block in event.message.content
            if isinstance(block, TextContent) and block.text
        )
        final_thinking = "\n".join(
            block.thinking
            for block in event.message.content
            if isinstance(block, ThinkingContent) and block.thinking
        )
        complete_text = final_text or self.state.current_text
        self.state.current_thinking = final_thinking or self.state.current_thinking
        tool_calls = [block for block in event.message.content if isinstance(block, ToolCall)]
        if complete_text.startswith(self.state.rendered_text):
            pending_text = complete_text[len(self.state.rendered_text) :]
        elif self.state.rendered_text:
            pending_text = self.state.stream_buffer
        else:
            pending_text = complete_text
        self.state.current_text = pending_text if pending_text.strip() else ""
        self.state.stream_buffer = ""
        if self.state.thinking_printed:
            self.state.current_thinking = ""
        renderable = build_message_renderable(self.state, tool_calls, theme)
        if renderable is not None:
            console.print(renderable)
        self.state.reset()
