"""交互式 TUI 会话的依赖装配与生命周期协调。"""

from __future__ import annotations

import asyncio
import signal
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.tree import Tree

from pi.agent.agent import Agent, AgentOptions
from pi.agent.session.base import SessionStorage
from pi.agent.tools import ToolContext
from pi.agent.types import AgentEvent, AgentTool, create_user_message
from pi.ai.models import list_models
from pi.ai.oauth import CredentialStore, get_default_credential_store
from pi.ai.types import Model, ModelThinkingLevel, ToolCall, Usage
from pi.coding_agent.extensions import ExtensionCommand, ExtensionContext
from pi.coding_agent.file_references import expand_file_references
from pi.coding_agent.model_auth import contains_likely_api_key
from pi.coding_agent.prompt_templates import PromptTemplate
from pi.coding_agent.sessions import SessionInfo, list_sessions
from pi.coding_agent.skills import Skill
from pi.coding_agent.themes import Theme
from pi.tui.commands import CommandDispatcher, build_command_names, validate_command_names
from pi.tui.formatting import read_git_status as _read_git_status
from pi.tui.prompt import (
    ANIMATION_FPS,
    PROMPT_STYLE_READY,
    PROMPT_STYLE_WORKING,
    build_input_prompt,
    build_input_status,
    create_prompt_session,
)
from pi.tui.rendering import (
    AgentEventRenderer,
    build_message_renderable,
    build_tool_trees,
    build_welcome_panel,
)
from pi.tui.selector import select_option


class InteractiveSession:
    """带 Rich 输出和 prompt-toolkit 输入的编码 agent 会话。"""

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
        self._git_status = _read_git_status(self._cwd)
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
        self._event_renderer = AgentEventRenderer()
        self._agent_task: asyncio.Task[None] | None = None
        self._request_active = False
        self._request_usage = Usage()
        self._pending_follow_ups: list[str] = []
        self._sessions_dir = sessions_dir
        self._session_id = session_id
        self._session_storage = session_storage
        self._commands = commands or {}
        self._extension_context = extension_context or ExtensionContext()
        self._skills = {skill.name: skill for skill in skills or []}
        self._prompt_templates = {template.name: template for template in prompt_templates or []}
        validate_command_names(set(self._commands), set(self._prompt_templates))
        self._credential_store = credential_store or get_default_credential_store()
        self._on_model_selected = on_model_selected
        self._on_thinking_selected = on_thinking_selected
        command_names = build_command_names(
            set(self._commands),
            set(self._skills),
            set(self._prompt_templates),
        )
        self._prompt_session: PromptSession[str] = create_prompt_session(
            self._input_prompt,
            command_names,
            history_file,
            self._should_erase_prompt_input,
        )

    def _tool_tree(self, tool_calls: list[ToolCall]) -> list[Tree]:
        """兼容入口：构建按名称分组的工具调用树。"""
        return build_tool_trees(tool_calls, self._theme)

    def _message_renderable(self, tool_calls: list[ToolCall]) -> RenderableType | None:
        """兼容入口：构建当前消息的 Rich 可渲染对象。"""
        return build_message_renderable(
            self._event_renderer.state,
            tool_calls,
            self._theme,
        )

    def _welcome_panel(self, version: str, sessions: list[SessionInfo]) -> Panel:
        """构建响应式欢迎面板。"""
        return build_welcome_panel(
            version,
            sessions,
            model=self._agent.state.model,
            thinking_level=self._agent.thinking_level,
            session_id=self._session_id,
            cwd=self._cwd,
            theme=self._theme,
            console_width=self._console.width,
        )

    def _animation_frame(self, frames: tuple[str, ...]) -> str:
        """返回固定宽度的动画帧，避免刷新时移动输入光标。"""
        return frames[int(time.monotonic() * ANIMATION_FPS) % len(frames)]

    def _input_prompt(self) -> StyleAndTextTuples:
        """构建状态轨道和开放式输入行。"""
        return build_input_prompt(
            self._input_status(),
            console_width=self._console.width,
            request_active=self._request_active,
            timestamp=time.monotonic(),
            pending_follow_ups=self._pending_follow_ups,
        )

    def _input_status(self) -> StyleAndTextTuples:
        """按终端宽度生成输入框顶部状态轨道。"""
        return build_input_status(
            self._agent,
            request_active=self._request_active,
            session_id=self._session_id,
            cwd=self._cwd,
            git_status=self._git_status,
            console_width=self._console.width,
        )

    def _should_erase_prompt_input(self, prompt: str) -> bool:
        """只擦除进入可视队列的 follow-up 命令，避免滚动区重复显示。"""
        command, _, argument = prompt.lstrip().partition(" ")
        return self._agent.is_busy and command == "/follow-up" and bool(argument.strip())

    def _set_request_active(self, active: bool) -> None:
        """同步更新请求状态、输入框边框与动态提示。"""
        self._request_active = active
        self._prompt_session.style = PROMPT_STYLE_WORKING if active else PROMPT_STYLE_READY
        self._refresh_prompt()

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
        """在一次用户请求完全结束后输出汇总用量。"""
        usage = self._request_usage
        if not usage.total_tokens:
            return
        cache = f" / {usage.cache_read} cached" if usage.cache_read else ""
        self._console.print(
            f"{usage.input} in / {usage.output} out / {usage.total_tokens} total tokens{cache}",
            style=self._theme.muted,
        )

    async def _run_agent_prompt(self, prompt: str) -> None:
        """运行完整请求，并在所有工具轮次结束后输出汇总用量。"""
        try:
            await self._agent.prompt(prompt)
        finally:
            self._pending_follow_ups.clear()
            self._git_status = await asyncio.to_thread(_read_git_status, self._cwd)
            self._set_request_active(False)
            self._print_request_usage()

    async def _handle_command(self, prompt: str) -> tuple[bool, bool]:
        """把输入委托给命令分发器。"""
        return await CommandDispatcher(self, select_option).handle(prompt)

    def _submit_agent_prompt(
        self,
        prompt: str,
        follow_up: bool = False,
        display_prompt: str | None = None,
    ) -> None:
        """启动新请求，或把消息加入正在运行的 agent 队列。"""
        if self._agent.is_busy:
            message = create_user_message(prompt)
            if follow_up:
                self._agent.follow_up(message)
                self._pending_follow_ups.append(display_prompt or prompt)
                self._refresh_prompt()
            else:
                self._agent.steer(message)
                self._console.print("Queued steering message.", style=self._theme.muted)
            return
        self._request_usage = Usage()
        self._set_request_active(True)
        self._agent_task = asyncio.create_task(self._run_agent_prompt(prompt))

    async def _on_event(self, event: AgentEvent) -> None:
        """转发扩展事件并把 Agent 事件交给渲染器。"""
        await self._emit_extension_event("agent_event", event)
        outcome = self._event_renderer.render(event, self._console, self._theme)
        if event.type == "agent_end" and self._pending_follow_ups:
            self._pending_follow_ups.pop(0)
            self._refresh_prompt()
        if outcome.usage is not None:
            self._record_usage(outcome.usage)
        if outcome.refresh_prompt:
            self._refresh_prompt()

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
                        prompt = await self._prompt_session.prompt_async()
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
