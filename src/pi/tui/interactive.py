"""使用 rich 渲染的交互式 REPL 会话。

提供对话式界面，支持流式输出、工具调用指示器，
以及助手响应的 Markdown 渲染。
"""

from __future__ import annotations

import asyncio
import inspect
import signal
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.history import FileHistory, InMemoryHistory
from prompt_toolkit.input import DummyInput
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from rich import box
from rich.console import Console, Group, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from pi.agent.agent import Agent, AgentOptions
from pi.agent.session.base import SessionStorage
from pi.agent.session.jsonl import JsonlStorage
from pi.agent.tools import ToolContext
from pi.agent.types import AgentEvent, AgentTool, create_user_message
from pi.ai.models import list_models
from pi.ai.oauth import CredentialStore, get_default_credential_store
from pi.ai.types import Model, ModelThinkingLevel, TextContent, ThinkingContent, ToolCall, Usage
from pi.coding_agent.extensions import ExtensionCommand, ExtensionContext
from pi.coding_agent.file_references import expand_file_references
from pi.coding_agent.model_auth import contains_likely_api_key, ensure_model_auth
from pi.coding_agent.prompt_templates import PromptTemplate
from pi.coding_agent.sessions import (
    SessionInfo,
    format_session_tree,
    list_sessions,
    new_session_id,
    resolve_entry_id,
    session_path,
)
from pi.coding_agent.skills import Skill
from pi.coding_agent.themes import Theme
from pi.tui.selector import select_option

_PROMPT_STYLE = Style.from_dict(
    {
        "frame.border": "#3b82f6",
        "input.prompt": "bold #60a5fa",
        "input.placeholder": "#6b7280",
        "input.ready": "bold #a3e635",
        "input.working": "bold #fbbf24",
        "bottom-toolbar": "bg:#171717 #d1d5db",
        "status.brand": "bold bg:#2563eb #ffffff",
        "status.model": "bold #22d3ee",
        "status.thinking": "#c084fc",
        "status.context": "#a3e635",
        "status.meta": "#9ca3af",
        "status.separator": "#525252",
    }
)

_PIY_LOGO = """       _ __   __
 _ __ (_)\\ \\ / /
| '_ \\| | \\ V /
| |_) | |  | |
| .__/|_|  |_|
|_|"""


def _format_tokens(tokens: int) -> str:
    """将 token 数格式化为适合状态栏的紧凑文本。"""
    if tokens < 1_000:
        return str(tokens)
    if tokens < 10_000:
        return f"{tokens / 1_000:.1f}k"
    if tokens < 1_000_000:
        return f"{round(tokens / 1_000)}k"
    if tokens < 10_000_000:
        return f"{tokens / 1_000_000:.1f}m"
    return f"{round(tokens / 1_000_000)}m"


def _format_tool_display(tool_name: str, arguments: dict) -> str:
    """根据工具名称和参数提取有意义的简短描述。"""
    path = arguments.get("path", "")
    pattern = arguments.get("pattern", "")
    command = arguments.get("command", "")

    if tool_name == "bash":
        return f"bash: {command}" if command else "bash"
    elif tool_name == "read":
        return f"read: {path}" if path else "read"
    elif tool_name == "write":
        return f"write: {path}" if path else "write"
    elif tool_name == "edit":
        return f"edit: {path}" if path else "edit"
    elif tool_name == "ls":
        return f"ls: {path}" if path else "ls"
    elif tool_name == "find":
        suffix = f": {pattern}" if pattern else ""
        dir_info = f" in {path}" if path and path != "." else ""
        return f"find{suffix}{dir_info}"
    elif tool_name == "grep":
        suffix = f": /{pattern}/" if pattern else ""
        return f"grep{suffix}"
    else:
        return tool_name


def _format_tool_target(tool_name: str, arguments: dict) -> str:
    """生成工具树子项文本，去除已经由分组标题表达的工具名称。"""
    display = _format_tool_display(tool_name, arguments)
    prefix = f"{tool_name}: "
    return display.removeprefix(prefix)


class _SafeFileHistory(FileHistory):
    """过滤疑似 API Key，避免凭据落入磁盘输入历史。"""

    def store_string(self, string: str) -> None:
        if not contains_likely_api_key(string):
            super().store_string(string)


class _SafeInMemoryHistory(InMemoryHistory):
    """过滤疑似 API Key，避免凭据保留在当前进程输入历史。"""

    def store_string(self, string: str) -> None:
        if not contains_likely_api_key(string):
            super().store_string(string)


class InteractiveSession:
    """带 rich 终端输出的交互式编码 agent 会话。"""

    def __init__(
        self,
        model: Model,
        system_prompt: str,
        tools: list[AgentTool],
        stream_fn: Any,
        session_id: str | None = None,
        session_storage: SessionStorage | None = None,
        sessions_dir: Path | None = None,
        thinking_level: ModelThinkingLevel = ModelThinkingLevel.OFF,
        temperature: float | None = None,
        max_tokens: int | None = None,
        commands: dict[str, ExtensionCommand] | None = None,
        extension_context: ExtensionContext | None = None,
        skills: list[Skill] | None = None,
        prompt_templates: list[PromptTemplate] | None = None,
        theme: Theme | None = None,
        cwd: Path | None = None,
        history_file: Path | None = None,
        credential_store: CredentialStore | None = None,
        on_model_selected: Callable[[Model], None] | None = None,
        on_thinking_selected: Callable[[ModelThinkingLevel], None] | None = None,
    ) -> None:
        self._cwd = (cwd or Path.cwd()).resolve()
        self._theme = theme or Theme()
        self._console = Console()
        self._agent = Agent(
            AgentOptions(
                model=model,
                system_prompt=system_prompt,
                tools=tools,
                stream_fn=stream_fn,
                tool_context=ToolContext(cwd=self._cwd),
                session_id=session_id,
                session_storage=session_storage,
                temperature=temperature,
                max_tokens=max_tokens,
                thinking_level=thinking_level,
            )
        )
        self._agent.subscribe(self._on_event)
        self._current_text = ""
        self._current_thinking = ""
        self._streamed_text = False
        self._thinking_printed = False
        self._agent_task: asyncio.Task[None] | None = None
        self._request_active = False
        self._request_usage = Usage()
        self._sessions_dir = sessions_dir
        self._session_id = session_id
        self._session_storage = session_storage
        self._commands = commands or {}
        self._extension_context = extension_context or ExtensionContext()
        self._skills = {skill.name: skill for skill in skills or []}
        self._prompt_templates = {template.name: template for template in prompt_templates or []}
        reserved_commands = {
            "branch",
            "clear",
            "compact",
            "exit",
            "follow-up",
            "help",
            "model",
            "new",
            "resume",
            "sessions",
            "skill",
            "steer",
            "templates",
            "thinking",
            "tree",
        }
        conflicts = set(self._prompt_templates) & (reserved_commands | set(self._commands))
        if conflicts:
            names = ", ".join(sorted(conflicts))
            raise ValueError(f"Prompt template command conflicts: {names}")
        extension_conflicts = set(self._commands) & reserved_commands
        if extension_conflicts:
            names = ", ".join(sorted(extension_conflicts))
            raise ValueError(f"Extension command conflicts: {names}")
        self._credential_store = credential_store or get_default_credential_store()
        self._on_model_selected = on_model_selected
        self._on_thinking_selected = on_thinking_selected
        command_names = [
            "/clear",
            "/branch",
            "/compact",
            "/exit",
            "/help",
            "/model",
            "/new",
            "/resume",
            "/sessions",
            "/templates",
            "/thinking",
            "/tree",
            "/skill",
            "/steer",
            "/follow-up",
            *[f"/{name}" for name in self._commands],
            *[f"/skill:{name}" for name in self._skills],
            *[f"/{name}" for name in self._prompt_templates],
        ]
        history = _SafeFileHistory(str(history_file)) if history_file else _SafeInMemoryHistory()
        interactive_terminal = sys.stdin.isatty() and sys.stdout.isatty()
        self._prompt_session: PromptSession[str] = PromptSession(
            message=[("class:input.prompt", " > ")],
            history=history,
            auto_suggest=AutoSuggestFromHistory(),
            completer=WordCompleter(sorted(command_names), sentence=True),
            complete_while_typing=False,
            placeholder=[("class:input.placeholder", "Type a message or /command")],
            rprompt=self._input_state,
            show_frame=True,
            style=_PROMPT_STYLE,
            input=None if interactive_terminal else DummyInput(),
            output=None if interactive_terminal else DummyOutput(),
        )

    def _tool_tree(self, tool_calls: list[ToolCall]) -> list[Tree]:
        """把同一轮工具调用按工具名称分组为紧凑树形结构。"""
        groups: dict[str, list[ToolCall]] = {}
        for call in tool_calls:
            groups.setdefault(call.name, []).append(call)

        trees = []
        labels = {
            "bash": "Bash",
            "edit": "Edit",
            "find": "Find",
            "grep": "Grep",
            "ls": "List",
            "read": "Read",
            "write": "Write",
        }
        for tool_name, calls in groups.items():
            count = f" ({len(calls)})" if len(calls) > 1 else ""
            title = Text.assemble(
                ("> ", self._theme.success),
                (labels.get(tool_name, tool_name), self._theme.primary),
                (count, self._theme.muted),
            )
            tree = Tree(title, guide_style=self._theme.muted)
            for call in calls:
                tree.add(Text(_format_tool_target(call.name, call.arguments)))
            trees.append(tree)
        return trees

    def _message_renderable(self, tool_calls: list[ToolCall]) -> RenderableType | None:
        """创建轻量消息正文，避免每一轮内容都被边框切割。"""
        content: list[RenderableType] = []
        if self._current_thinking:
            content.append(Text(self._current_thinking, style=self._theme.thinking))
        if self._current_text:
            content.append(Markdown(self._current_text))
        content.extend(self._tool_tree(tool_calls))
        return Group(*content) if content else None

    def _welcome_panel(self, version: str, sessions: list[SessionInfo]) -> Panel:
        """构建响应式欢迎面板。"""
        model = self._agent.state.model
        brand = Text(_PIY_LOGO, style=self._theme.primary)
        brand.append("\n\n")
        brand.append(model.name if model else "No model", style="bold")
        if model is not None:
            brand.append(f"\n{model.provider}", style=self._theme.muted)
        if model is not None and model.reasoning:
            brand.append(
                f"  thinking {self._agent.thinking_level.value}",
                style=self._theme.thinking,
            )

        actions = Text()
        actions.append("Quick actions\n", style=self._theme.primary)
        for command, description in (
            ("/help", "Commands"),
            ("/model", "Switch model"),
            ("/thinking", "Reasoning level"),
            ("/sessions", "Session history"),
        ):
            actions.append(f"{command:<12}", style=self._theme.primary)
            actions.append(f"{description}\n", style=self._theme.muted)

        recent = Text()
        recent.append("\nRecent sessions\n", style=self._theme.primary)
        visible_sessions = [item for item in sessions if item.session_id != self._session_id][:3]
        if not visible_sessions:
            recent.append("No recent sessions", style=self._theme.muted)
        for item in visible_sessions:
            preview = item.preview or "Empty session"
            recent.append(f"{item.session_id[:8]}  ", style=self._theme.primary)
            recent.append(f"{preview}\n", style=self._theme.muted)

        details = Group(actions, recent)
        grid = Table.grid(expand=True, padding=(0, 2))
        if self._console.width >= 90:
            grid.add_column(width=24)
            grid.add_column(ratio=1)
            grid.add_row(brand, details)
        else:
            grid.add_column(ratio=1)
            grid.add_row(brand)
            grid.add_row(details)

        width = min(96, max(20, self._console.width - 2))
        return Panel(
            grid,
            title=f" piY v{version} ",
            subtitle=f" workspace {self._cwd.name} ",
            subtitle_align="left",
            border_style=self._theme.primary,
            box=box.SQUARE,
            width=width,
        )

    def _input_state(self) -> StyleAndTextTuples:
        """显示当前输入区状态。"""
        if self._request_active:
            return [("class:input.working", " Working ")]
        return [("class:input.ready", " Ready ")]

    def _refresh_prompt(self) -> None:
        """通知输入应用刷新状态，避免直接操作终端光标。"""
        if self._prompt_session.app.is_running:
            self._prompt_session.app.invalidate()

    def _record_usage(self, usage: Usage) -> None:
        """累计一次用户请求内所有模型轮次的 token 用量。"""
        self._request_usage.input += usage.input
        self._request_usage.output += usage.output
        self._request_usage.cache_read += usage.cache_read
        self._request_usage.cache_write += usage.cache_write
        self._request_usage.total_tokens += usage.total_tokens

    def _print_request_usage(self) -> None:
        """在一次用户请求完全结束后输出一条汇总用量。"""
        usage = self._request_usage
        if not usage.total_tokens:
            return
        cache = f" / {usage.cache_read} cached" if usage.cache_read else ""
        self._console.print(
            f"{usage.input} in / {usage.output} out / "
            f"{usage.total_tokens} total tokens{cache}",
            style=self._theme.muted,
        )

    async def _run_agent_prompt(self, prompt: str) -> None:
        """运行完整请求，并在所有工具轮次结束后输出汇总用量。"""
        try:
            await self._agent.prompt(prompt)
        finally:
            self._request_active = False
            self._refresh_prompt()
            self._print_request_usage()

    async def _handle_command(self, prompt: str) -> tuple[bool, bool]:
        """处理斜杠命令，返回（是否已处理，是否退出）。"""
        parts = prompt.strip().split(maxsplit=1)
        command = parts[0].lower()
        argument = parts[1].strip() if len(parts) > 1 else ""
        if command in ("exit", "quit", "/q", "/quit", "/exit"):
            return True, True
        if command == "/clear":
            self._console.clear()
            return True, False
        if command == "/help":
            self._console.print(
                "/new  /resume <id>  /sessions  /model [provider/id]  "
                "/tree  /branch <entry-id>  "
                "/compact [target_tokens]  "
                "/thinking [level]  "
                "/templates  /<template> [arguments]  "
                "/skill <name> [task]  /steer <message>  /follow-up <message>  "
                "/clear  /help  /exit",
                style=self._theme.muted,
            )
            return True, False
        if command == "/thinking":
            model = self._agent.state.model
            if model is None or not model.reasoning:
                self._console.print(
                    "Current model does not support thinking.",
                    style=self._theme.warning,
                )
                return True, False
            if not argument:
                level = await select_option(
                    "Thinking level",
                    [(candidate, candidate.value) for candidate in ModelThinkingLevel],
                    default=self._agent.thinking_level,
                )
                if level is None:
                    self._console.print("Thinking selection cancelled.", style=self._theme.muted)
                    return True, False
            else:
                try:
                    level = ModelThinkingLevel(argument.lower())
                except ValueError:
                    self._console.print(
                        f"Invalid thinking level: {argument}",
                        style=self._theme.error,
                    )
                    return True, False
            try:
                self._agent.set_thinking_level(level)
            except RuntimeError as exc:
                self._console.print(str(exc), style=self._theme.warning)
                return True, False
            if self._on_thinking_selected is not None:
                self._on_thinking_selected(level)
            self._console.print(f"Thinking: {level.value}", style=self._theme.muted)
            return True, False
        if command == "/compact":
            if self._agent.is_busy:
                self._console.print("Agent is busy. Wait for idle.", style=self._theme.warning)
                return True, False
            target: int | None = None
            if argument:
                try:
                    target = int(argument)
                except ValueError:
                    self._console.print(
                        f"Invalid token count: {argument}", style=self._theme.error
                    )
                    return True, False
            try:
                result = await self._agent.compact(target)
            except RuntimeError as exc:
                self._console.print(str(exc), style=self._theme.error)
                return True, False
            if result.dropped_messages > 0:
                self._console.print(
                    f"Compacted: {result.original_tokens} -> {result.compacted_tokens} tokens "
                    f"({result.dropped_messages} messages dropped)",
                    style=self._theme.muted,
                )
            else:
                self._console.print(
                    f"No compaction needed ({result.original_tokens} tokens)",
                    style=self._theme.muted,
                )
            return True, False
        if command == "/templates":
            if not self._prompt_templates:
                self._console.print("No prompt templates.", style=self._theme.muted)
                return True, False
            for template in self._prompt_templates.values():
                description = f" - {template.description}" if template.description else ""
                self._console.print(f"/{template.name}{description}", style=self._theme.muted)
            return True, False
        if command == "/model":
            models = list_models()
            selected: Model | None = None
            if argument:
                matches = [
                    model
                    for model in models
                    if argument in (model.id, f"{model.provider}/{model.id}")
                ]
                if len(matches) != 1:
                    self._console.print(
                        f"Unknown or ambiguous model: {argument}",
                        style=self._theme.error,
                    )
                    return True, False
                selected = matches[0]
            else:
                current = self._agent.state.model
                model_options = []
                for model in models:
                    is_current = current and (model.provider, model.id) == (
                        current.provider,
                        current.id,
                    )
                    suffix = "  current" if is_current else ""
                    model_options.append((model, f"{model.provider}/{model.id}{suffix}"))
                selected = await select_option(
                    "Select model",
                    model_options,
                    default=current,
                )
                if selected is None:
                    self._console.print("Model selection cancelled.", style=self._theme.muted)
                    return True, False

            if not await ensure_model_auth(selected, self._credential_store):
                return True, False
            self._agent.set_model(selected)
            if self._session_storage is not None:
                await self._session_storage.append_model_change(
                    selected.provider,
                    selected.id,
                )
            if self._on_model_selected is not None:
                self._on_model_selected(selected)
            self._console.print(
                f"Model: {selected.provider}/{selected.id}",
                style=self._theme.muted,
            )
            return True, False
        if command == "/tree":
            if self._session_storage is None:
                self._console.print("Session storage is disabled.", style=self._theme.warning)
                return True, False
            entries = await self._session_storage.get_entries()
            leaf_id = await self._session_storage.get_leaf_id()
            self._console.print(format_session_tree(entries, leaf_id), style=self._theme.muted)
            return True, False
        if command == "/branch":
            if self._session_storage is None:
                self._console.print("Session storage is disabled.", style=self._theme.warning)
                return True, False
            if not argument:
                self._console.print("Usage: /branch <entry-id>", style=self._theme.warning)
                return True, False
            try:
                entry_id = resolve_entry_id(
                    await self._session_storage.get_entries(),
                    argument,
                )
                await self._session_storage.branch_from(entry_id)
            except (KeyError, ValueError) as exc:
                self._console.print(str(exc), style=self._theme.error)
                return True, False
            await self._agent.switch_session(self._session_storage, self._session_id)
            await self._restore_session_model()
            await self._emit_extension_event(
                "session_switch",
                {"reason": "branch", "session_id": self._session_id},
            )
            self._console.print(f"Branched from: {entry_id}", style=self._theme.muted)
            return True, False
        if command == "/skill" or command.startswith("/skill:"):
            if command.startswith("/skill:"):
                skill_name = command.split(":", 1)[1]
                skill_argument = argument
            else:
                skill_parts = argument.split(maxsplit=1)
                skill_name = skill_parts[0] if skill_parts else ""
                skill_argument = skill_parts[1] if len(skill_parts) > 1 else ""
            if not skill_name:
                self._console.print(
                    "  ".join(sorted(self._skills)) or "No skills.",
                    style=self._theme.muted,
                )
                return True, False
            skill = self._skills.get(skill_name)
            if skill is None:
                self._console.print(f"Skill not found: {skill_name}", style=self._theme.error)
                return True, False
            prompt = (
                f"Apply the following skill instructions.\n\n{skill.read()}"
                f"\n\nTask:\n{skill_argument or 'Follow the skill instructions.'}"
            )
            self._submit_agent_prompt(prompt)
            return True, False
        if command in ("/steer", "/follow-up"):
            if not argument:
                self._console.print(f"Usage: {command} <message>", style=self._theme.warning)
                return True, False
            self._submit_agent_prompt(
                expand_file_references(argument, self._cwd),
                follow_up=command == "/follow-up",
            )
            return True, False
        if command == "/sessions":
            if self._sessions_dir is None:
                self._console.print("Session storage is disabled.", style=self._theme.warning)
                return True, False
            sessions = await list_sessions(self._sessions_dir)
            if not sessions:
                self._console.print("No sessions.", style=self._theme.muted)
            for item in sessions:
                self._console.print(
                    f"{item.session_id}  {item.message_count:>4}  {item.preview}",
                    style=self._theme.muted,
                )
            return True, False
        if command in ("/new", "/resume"):
            if self._sessions_dir is None:
                self._console.print("Session storage is disabled.", style=self._theme.warning)
                return True, False
            session_id = argument if command == "/resume" else new_session_id()
            if not session_id:
                self._console.print("Usage: /resume <session-id>", style=self._theme.warning)
                return True, False
            try:
                path = session_path(self._sessions_dir, session_id)
            except ValueError as exc:
                self._console.print(str(exc), style=self._theme.error)
                return True, False
            if command == "/resume" and not path.exists():
                self._console.print(f"Session not found: {session_id}", style=self._theme.error)
                return True, False
            storage = JsonlStorage(path)
            if command == "/new":
                current_model = self._agent.state.model
                if current_model is not None:
                    await storage.append_model_change(
                        current_model.provider,
                        current_model.id,
                    )
            await self._agent.switch_session(storage, session_id)
            self._session_storage = storage
            self._session_id = session_id
            await self._restore_session_model()
            await self._emit_extension_event(
                "session_switch",
                {"reason": command.lstrip("/"), "session_id": session_id},
            )
            self._console.print(f"Session: {session_id}", style=self._theme.muted)
            return True, False
        extension_command = self._commands.get(command.lstrip("/"))
        if extension_command is not None:
            result = extension_command(argument, self._agent)
            if inspect.isawaitable(result):
                result = await result
            if result:
                self._console.print(result)
            return True, False
        prompt_template = self._prompt_templates.get(command.lstrip("/"))
        if prompt_template is not None:
            self._submit_agent_prompt(
                expand_file_references(prompt_template.render(argument), self._cwd)
            )
            return True, False
        if prompt.startswith("/"):
            self._console.print(f"Unknown command: {command}", style=self._theme.warning)
            return True, False
        return False, False

    def _submit_agent_prompt(self, prompt: str, follow_up: bool = False) -> None:
        """启动新请求，或把消息加入正在运行的 agent 队列。"""
        if self._agent.is_busy:
            message = create_user_message(prompt)
            if follow_up:
                self._agent.follow_up(message)
                self._console.print("Queued follow-up.", style=self._theme.muted)
            else:
                self._agent.steer(message)
                self._console.print("Queued steering message.", style=self._theme.muted)
            return
        self._request_usage = Usage()
        self._request_active = True
        self._agent_task = asyncio.create_task(self._run_agent_prompt(prompt))
        self._refresh_prompt()

    async def _on_event(self, event: AgentEvent) -> None:
        """处理 agent 事件用于显示。"""
        await self._emit_extension_event("agent_event", event)
        if event.type == "message_start":
            self._current_text = ""
            self._current_thinking = ""
            self._streamed_text = False
            self._thinking_printed = False
        elif event.type == "text_delta":
            if self._current_thinking and not self._thinking_printed:
                self._console.print(Text(self._current_thinking, style=self._theme.thinking))
                self._thinking_printed = True
            self._current_text += event.delta
            sys.stdout.write(event.delta)
            self._streamed_text = True
        elif event.type == "thinking_delta":
            self._current_thinking += event.delta
        elif event.type == "message_end":
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
            self._current_text = final_text or self._current_text
            self._current_thinking = final_thinking or self._current_thinking
            tool_calls = [block for block in event.message.content if isinstance(block, ToolCall)]
            if self._streamed_text:
                sys.stdout.write("\n")
                sys.stdout.flush()
                self._current_text = ""
            if self._thinking_printed:
                self._current_thinking = ""
            renderable = self._message_renderable(tool_calls)
            if renderable is not None:
                self._console.print(renderable)
            self._current_text = ""
            self._current_thinking = ""
            self._refresh_prompt()
        elif event.type == "tool_execution_start":
            self._current_text = ""
            self._current_thinking = ""
            self._refresh_prompt()
        elif event.type == "tool_execution_end":
            if event.result and event.result.is_error:
                for block in event.result.content:
                    if hasattr(block, "text"):
                        self._console.print(f"  error: {block.text}", style=self._theme.error)
        elif event.type == "turn_end":
            if event.message.stop_reason == "aborted":
                self._console.print("Aborted.", style=self._theme.muted)
            elif event.message.error_message:
                self._console.print(
                    f"Error: {event.message.error_message}", style=self._theme.error
                )
            self._record_usage(event.message.usage)
        elif event.type == "provider_retry":
            self._console.print(
                f"Retry {event.attempt}/{event.max_retries} in {event.delay_ms} ms",
                style=self._theme.warning,
            )
        elif event.type == "context_compacted":
            self._console.print(
                f"Context compacted: {event.original_tokens} -> {event.compacted_tokens} tokens",
                style=self._theme.muted,
            )

    def _bottom_toolbar(self) -> StyleAndTextTuples:
        model = self._agent.state.model
        model_name = model.name if model else "No model"
        session = self._session_id[:8] if self._session_id else "memory"
        width = self._console.width
        context_usage = self._agent.get_context_usage()
        fragments: StyleAndTextTuples = [
            ("class:status.brand", " piY "),
            ("class:status.separator", " | "),
            ("class:status.model", f"{model_name}"),
        ]
        if width >= 72 and model is not None and model.reasoning:
            fragments.extend(
                [
                    ("class:status.separator", " | "),
                    ("class:status.thinking", f"thinking {self._agent.thinking_level.value}"),
                ]
            )
        if width >= 52 and context_usage is not None:
            fragments.extend(
                [
                    ("class:status.separator", " | "),
                    (
                        "class:status.context",
                        f"ctx {_format_tokens(context_usage.tokens)}/"
                        f"{_format_tokens(context_usage.context_window)} "
                        f"{context_usage.percent:.1f}%",
                    ),
                ]
            )
        if width >= 100:
            fragments.extend(
                [
                    ("class:status.separator", " | "),
                    ("class:status.meta", f"session {session}"),
                ]
            )
        if width >= 130:
            fragments.extend(
                [
                    ("class:status.separator", " | "),
                    ("class:status.meta", str(self._cwd)),
                ]
            )
        fragments.append(("class:status.meta", " "))
        return fragments

    async def _restore_session_model(self) -> None:
        """恢复当前分支最后记录的模型选择。"""
        if self._session_storage is None:
            return
        selection = await self._session_storage.get_model_selection()
        if selection is None:
            return
        provider, model_id = selection
        model = next(
            (
                candidate
                for candidate in list_models()
                if candidate.provider == provider and candidate.id == model_id
            ),
            None,
        )
        if model is not None:
            self._agent.set_model(model)

    async def _emit_extension_event(self, event_type: str, data: Any = None) -> None:
        """触发扩展事件并把隔离的错误输出到终端。"""
        failures = await self._extension_context.emit(event_type, data, self._agent)
        for failure in failures:
            self._console.print(
                f"Extension {failure.source} failed during {failure.event_type}: {failure.error}",
                style=self._theme.error,
            )

    async def run(self) -> None:
        """运行交互式 REPL 循环。"""
        from pi import __version__

        await self._agent.restore()
        await self._restore_session_model()
        await self._emit_extension_event(
            "session_start",
            {"session_id": self._session_id},
        )

        sessions = await list_sessions(self._sessions_dir) if self._sessions_dir else []
        self._console.print(self._welcome_panel(__version__, sessions))
        self._console.print()

        previous_sigint = signal.getsignal(signal.SIGINT)

        def handle_sigint(signum, frame) -> None:
            if self._agent.is_busy:
                self._agent.abort()
                return
            raise KeyboardInterrupt

        signal.signal(signal.SIGINT, handle_sigint)
        try:
            while True:
                try:
                    with patch_stdout(raw=True):
                        prompt = await self._prompt_session.prompt_async(
                            bottom_toolbar=self._bottom_toolbar,
                        )
                    if not prompt.strip():
                        continue
                    if contains_likely_api_key(prompt):
                        self._console.print(
                            "Input looks like an API key and was not sent. "
                            "Use /model to configure credentials.",
                            style=self._theme.error,
                        )
                        continue
                    handled, should_exit = await self._handle_command(prompt)
                    if should_exit:
                        break
                    if handled:
                        continue

                    self._submit_agent_prompt(expand_file_references(prompt, self._cwd))
                except (KeyboardInterrupt, EOFError):
                    self._console.print("\nGoodbye.", style=self._theme.muted)
                    break
                except Exception as exc:
                    self._console.print(f"Error: {exc}", style=self._theme.error)
        finally:
            if self._agent.is_busy:
                self._agent.abort()
                await self._agent.wait_for_idle()
            await self._emit_extension_event(
                "session_shutdown",
                {"session_id": self._session_id},
            )
            signal.signal(signal.SIGINT, previous_sigint)
