"""使用 rich 渲染的交互式 REPL 会话。

提供对话式界面，支持流式输出、工具调用指示器，
以及助手响应的 Markdown 渲染。
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from pi.agent.agent import Agent, AgentOptions
from pi.agent.types import AgentEvent, AgentTool
from pi.ai.types import Model


class InteractiveSession:
    """带 rich 终端输出的交互式编码 agent 会话。"""

    def __init__(
        self,
        model: Model,
        system_prompt: str,
        tools: list[AgentTool],
        stream_fn: Any,
    ) -> None:
        self._console = Console()
        self._agent = Agent(AgentOptions(
            model=model,
            system_prompt=system_prompt,
            tools=tools,
            stream_fn=stream_fn,
        ))
        self._agent.subscribe(self._on_event)
        self._current_text = ""
        self._live: Live | None = None

    async def _on_event(self, event: AgentEvent) -> None:
        """处理 agent 事件用于显示。"""
        if event.type == "message_start":
            self._current_text = ""
            self._live = Live(
                Panel("", title="pi", border_style="blue"),
                console=self._console,
                refresh_per_second=15,
            )
            self._live.start()
        elif event.type == "text_delta":
            self._current_text += event.delta
            if self._live:
                self._live.update(
                    Panel(Markdown(self._current_text), title="pi", border_style="blue")
                )
        elif event.type == "message_end":
            if self._live:
                self._live.stop()
                self._live = None
            self._current_text = ""
        elif event.type == "tool_execution_start":
            if self._live:
                self._live.stop()
                self._live = None
            self._current_text = ""
            self._console.print(f"  [dim]-> {event.tool_name}[/dim]")
        elif event.type == "tool_execution_end":
            if event.result and event.result.is_error:
                for block in event.result.content:
                    if hasattr(block, "text"):
                        self._console.print(f"  [red]error: {block.text}[/red]")
        elif event.type == "turn_end" and event.message.error_message:
            self._console.print(f"[red]Error: {event.message.error_message}[/red]")

    async def run(self) -> None:
        """运行交互式 REPL 循环。"""
        from pi import __version__

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
                if prompt.strip().lower() in ("exit", "quit", "/q"):
                    break

                await self._agent.prompt(prompt)
                self._console.print()
            except (KeyboardInterrupt, EOFError):
                self._console.print("\n[dim]Goodbye.[/dim]")
                break
            except Exception as exc:
                self._console.print(f"[red]Error: {exc}[/red]")
