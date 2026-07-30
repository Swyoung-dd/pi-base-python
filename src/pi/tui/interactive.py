"""使用 rich 渲染的交互式 REPL 会话。

提供对话式界面，支持流式输出、工具调用指示器，
以及助手响应的 Markdown 渲染。
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from pi.agent.agent import Agent, AgentOptions
from pi.agent.session.base import SessionStorage
from pi.agent.session.jsonl import JsonlStorage
from pi.agent.types import AgentEvent, AgentTool
from pi.ai.types import Model, ModelThinkingLevel
from pi.coding_agent.extensions import ExtensionCommand
from pi.coding_agent.sessions import list_sessions, new_session_id, session_path
from pi.coding_agent.skills import Skill


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
        commands: dict[str, ExtensionCommand] | None = None,
        skills: list[Skill] | None = None,
    ) -> None:
        self._console = Console()
        self._agent = Agent(
            AgentOptions(
                model=model,
                system_prompt=system_prompt,
                tools=tools,
                stream_fn=stream_fn,
                session_id=session_id,
                session_storage=session_storage,
                thinking_level=thinking_level,
            )
        )
        self._agent.subscribe(self._on_event)
        self._current_text = ""
        self._current_thinking = ""
        self._live: Live | None = None
        self._sessions_dir = sessions_dir
        self._commands = commands or {}
        self._skills = {skill.name: skill for skill in skills or []}

    def _update_live(self) -> None:
        if self._live is None:
            return
        content = []
        if self._current_thinking:
            content.append(Text(self._current_thinking, style="dim"))
        if self._current_text:
            content.append(Markdown(self._current_text))
        self._live.update(Panel(Group(*content), title="pi", border_style="blue"))

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
                "/new  /resume <id>  /sessions  /skill <name> [task]  /clear  /help  /exit",
                style="dim",
            )
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
                self._console.print("  ".join(sorted(self._skills)) or "No skills.", style="dim")
                return True, False
            skill = self._skills.get(skill_name)
            if skill is None:
                self._console.print(f"Skill not found: {skill_name}", style="red")
                return True, False
            prompt = (
                f"Apply the following skill instructions.\n\n{skill.read()}"
                f"\n\nTask:\n{skill_argument or 'Follow the skill instructions.'}"
            )
            await self._agent.prompt(prompt)
            return True, False
        if command == "/sessions":
            if self._sessions_dir is None:
                self._console.print("Session storage is disabled.", style="yellow")
                return True, False
            sessions = await list_sessions(self._sessions_dir)
            if not sessions:
                self._console.print("No sessions.", style="dim")
            for item in sessions:
                self._console.print(
                    f"{item.session_id}  {item.message_count:>4}  {item.preview}",
                    style="dim",
                )
            return True, False
        if command in ("/new", "/resume"):
            if self._sessions_dir is None:
                self._console.print("Session storage is disabled.", style="yellow")
                return True, False
            session_id = argument if command == "/resume" else new_session_id()
            if not session_id:
                self._console.print("Usage: /resume <session-id>", style="yellow")
                return True, False
            try:
                path = session_path(self._sessions_dir, session_id)
            except ValueError as exc:
                self._console.print(str(exc), style="red")
                return True, False
            if command == "/resume" and not path.exists():
                self._console.print(f"Session not found: {session_id}", style="red")
                return True, False
            await self._agent.switch_session(JsonlStorage(path), session_id)
            self._console.print(f"Session: {session_id}", style="dim")
            return True, False
        extension_command = self._commands.get(command.lstrip("/"))
        if extension_command is not None:
            result = extension_command(argument, self._agent)
            if inspect.isawaitable(result):
                result = await result
            if result:
                self._console.print(result)
            return True, False
        if prompt.startswith("/"):
            self._console.print(f"Unknown command: {command}", style="yellow")
            return True, False
        return False, False

    async def _on_event(self, event: AgentEvent) -> None:
        """处理 agent 事件用于显示。"""
        if event.type == "message_start":
            self._current_text = ""
            self._current_thinking = ""
            self._live = Live(
                Panel("", title="pi", border_style="blue"),
                console=self._console,
                refresh_per_second=15,
            )
            self._live.start()
        elif event.type == "text_delta":
            self._current_text += event.delta
            self._update_live()
        elif event.type == "thinking_delta":
            self._current_thinking += event.delta
            self._update_live()
        elif event.type == "message_end":
            if self._live:
                self._live.stop()
                self._live = None
            self._current_text = ""
            self._current_thinking = ""
        elif event.type == "tool_execution_start":
            if self._live:
                self._live.stop()
                self._live = None
            self._current_text = ""
            self._current_thinking = ""
            self._console.print(f"  [dim]-> {event.tool_name}[/dim]")
        elif event.type == "tool_execution_end":
            if event.result and event.result.is_error:
                for block in event.result.content:
                    if hasattr(block, "text"):
                        self._console.print(f"  [red]error: {block.text}[/red]")
        elif event.type == "turn_end" and event.message.error_message:
            self._console.print(f"[red]Error: {event.message.error_message}[/red]")
        elif event.type == "provider_retry":
            self._console.print(
                f"Retry {event.attempt}/{event.max_retries} in {event.delay_ms} ms",
                style="yellow",
            )

    async def run(self) -> None:
        """运行交互式 REPL 循环。"""
        from pi import __version__

        await self._agent.restore()

        self._console.print(
            Panel(Text(f"Pi v{__version__} - coding agent", justify="center"), style="blue")
        )
        self._console.print(f"Model: [bold]{self._agent.state.model.id}[/bold]")
        self._console.print("Type your message and press Enter. Ctrl+C to exit.\n")

        while True:
            try:
                prompt = self._console.input("[bold green]>>>[/bold green] ")
                if not prompt.strip():
                    continue
                handled, should_exit = await self._handle_command(prompt)
                if should_exit:
                    break
                if handled:
                    continue

                await self._agent.prompt(prompt)
                self._console.print()
            except (KeyboardInterrupt, EOFError):
                self._console.print("\n[dim]Goodbye.[/dim]")
                break
            except Exception as exc:
                self._console.print(f"[red]Error: {exc}[/red]")
