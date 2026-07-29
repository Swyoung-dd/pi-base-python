"""Pi 编码 agent 的 CLI 入口。

支持：
- 交互模式（默认）：带流式输出的 REPL
- 打印模式（-p）：一次性提示，打印结果后退出
- 模型列表（--list-models）：显示可用模型
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click

from pi.ai.models import list_models
from pi.ai.providers.registry import get_provider
from pi.ai.streaming import DoneEvent, ErrorEvent, TextDeltaEvent
from pi.ai.types import StreamOptions, TextContent
from pi.agent.agent import Agent, AgentOptions
from pi.agent.tools.base import ToolContext
from pi.agent.types import AgentEvent, AgentTool
from pi.coding_agent.config import load_config
from pi.coding_agent.system_prompt import build_system_prompt
from pi.coding_agent.tools import (
    create_bash_tool,
    create_edit_tool,
    create_find_tool,
    create_grep_tool,
    create_ls_tool,
    create_read_tool,
    create_write_tool,
)


def _build_tools(cwd: Path) -> list[AgentTool]:
    return [
        create_read_tool(),
        create_write_tool(),
        create_edit_tool(),
        create_bash_tool(),
        create_ls_tool(),
        create_find_tool(),
        create_grep_tool(),
    ]


def _make_stream_fn(model):
    """创建绑定到模型提供商的 stream 函数。"""
    provider = get_provider(model.provider)
    if provider is None:
        raise RuntimeError(f"No provider registered for: {model.provider}")

    async def stream_fn(m, ctx, options):
        return await provider.stream(m, ctx, options)

    return stream_fn


def _print_event(event: AgentEvent) -> None:
    """打印模式下将 agent 事件输出到 stdout。"""
    if event.type == "message_end":
        for block in event.message.content:
            if hasattr(block, "text"):
                click.echo(block.text)
    elif event.type == "tool_execution_start":
        newline = chr(10)
        click.echo(f"{newline}[tool: {event.tool_name}]", err=True)
    elif event.type == "tool_execution_end":
        if event.result and event.result.is_error:
            for block in event.result.content:
                if hasattr(block, "text"):
                    click.echo(f"  [error] {block.text}", err=True)
    elif event.type == "turn_end":
        if event.message.error_message:
            click.echo(f"Error: {event.message.error_message}", err=True)


async def run_print_mode(prompt: str, config) -> None:
    """执行一次性提示并打印结果。"""
    model = None
    for m in list_models():
        if m.id == config.model:
            model = m
            break
    if model is None:
        click.echo(f"Unknown model: {config.model}", err=True)
        click.echo("Available models:", err=True)
        for m in list_models():
            click.echo(f"  {m.id}", err=True)
        sys.exit(1)

    cwd = Path.cwd()
    tools = _build_tools(cwd)
    tool_names = [t.name for t in tools]
    system_prompt = build_system_prompt(cwd, tool_names)
    if config.system_prompt:
        system_prompt = config.system_prompt + chr(10) + chr(10) + system_prompt

    stream_fn = _make_stream_fn(model)

    agent = Agent(AgentOptions(
        model=model,
        system_prompt=system_prompt,
        tools=tools,
        stream_fn=stream_fn,
    ))
    agent.subscribe(_print_event)

    await agent.prompt(prompt)


async def run_interactive_mode(config) -> None:
    """运行交互式 REPL 模式。"""
    from pi.tui.interactive import InteractiveSession

    model = None
    for m in list_models():
        if m.id == config.model:
            model = m
            break
    if model is None:
        click.echo(f"Unknown model: {config.model}", err=True)
        sys.exit(1)

    cwd = Path.cwd()
    tools = _build_tools(cwd)
    tool_names = [t.name for t in tools]
    system_prompt = build_system_prompt(cwd, tool_names)
    if config.system_prompt:
        system_prompt = config.system_prompt + chr(10) + chr(10) + system_prompt

    stream_fn = _make_stream_fn(model)

    session = InteractiveSession(
        model=model,
        system_prompt=system_prompt,
        tools=tools,
        stream_fn=stream_fn,
    )
    await session.run()


@click.command()
@click.option("-p", "--prompt", "prompt_text", default=None, help="One-shot prompt (print mode)")
@click.option("-m", "--model", "model_id", default=None, help="Model ID to use")
@click.option("--list-models", "list_models_flag", is_flag=True, help="List available models and exit")
@click.option("--version", is_flag=True, help="Show version and exit")
def main(prompt_text, model_id, list_models_flag, version):
    """Pi - 终端中的 AI 编码 agent。"""
    from pi import __version__

    if version:
        click.echo(f"pi {__version__}")
        return

    if list_models_flag:
        for m in list_models():
            click.echo(f"{m.id:40s}  {m.name}  ({m.provider})")
        return

    config = load_config()
    if model_id:
        config.model = model_id

    if prompt_text is not None:
        asyncio.run(run_print_mode(prompt_text, config))
    else:
        asyncio.run(run_interactive_mode(config))


if __name__ == "__main__":
    main()
