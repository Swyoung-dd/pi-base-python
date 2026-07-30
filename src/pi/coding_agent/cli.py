"""Pi 编码 agent 的 CLI 入口。

支持：
- 交互模式（默认）：带流式输出的 REPL
- 打印模式（-p）：一次性提示，打印结果后退出
- 模型列表（--list-models）：显示可用模型
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import click

from pi.agent.agent import Agent, AgentOptions
from pi.agent.session import JsonlStorage
from pi.agent.types import AgentTool
from pi.ai.models import get_model, list_models
from pi.ai.oauth import OAuthEvent, get_default_oauth_store, login_oauth
from pi.ai.oauth_xai import register_xai_oauth
from pi.ai.providers.registry import get_provider
from pi.ai.types import ModelThinkingLevel
from pi.coding_agent.config import load_config
from pi.coding_agent.extensions import load_extensions
from pi.coding_agent.file_references import expand_file_references
from pi.coding_agent.output import PrintRenderer
from pi.coding_agent.sessions import list_sessions, resolve_session_id, session_path
from pi.coding_agent.setup import run_setup
from pi.coding_agent.skills import format_skills_for_prompt, load_skills
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


async def _build_runtime(config, cwd: Path):
    extensions = await load_extensions(
        config.extension_paths,
        config.enable_entrypoint_extensions,
    )
    tools = [*_build_tools(cwd), *extensions.tools]
    tool_names = [tool.name for tool in tools]
    if len(tool_names) != len(set(tool_names)):
        raise click.ClickException("Extension tool name conflicts with a built-in tool")
    skills = []
    if config.enable_skills:
        skill_paths = [
            Path.home() / ".pi" / "skills",
            config.config_dir / "skills",
            *config.skill_paths,
        ]
        skills = load_skills([path for path in skill_paths if path.exists()])
    system_prompt = build_system_prompt(cwd, tool_names)
    sections = [
        config.system_prompt,
        *extensions.system_prompt_sections,
        system_prompt,
        format_skills_for_prompt(skills),
    ]
    return (
        tools,
        "\n\n".join(section for section in sections if section),
        extensions.commands,
        skills,
    )


def _make_stream_fn(model):
    """创建绑定到模型提供商的 stream 函数。"""
    provider = get_provider(model.provider)
    if provider is None:
        raise RuntimeError(f"No provider registered for: {model.provider}")

    async def stream_fn(m, ctx, options):
        return await provider.stream(m, ctx, options)

    return stream_fn


def _resolve_model(config):
    """按配置中的 ID/provider 解析唯一模型。"""
    model = get_model(config.model, config.provider)
    if model is not None:
        return model
    click.echo(f"Unknown or ambiguous model: {config.model}", err=True)
    click.echo("Available models:", err=True)
    for candidate in list_models():
        click.echo(f"  {candidate.provider}/{candidate.id}", err=True)
    raise click.ClickException("Model selection failed")


async def run_print_mode(prompt: str, config, output_format: str = "text") -> None:
    """执行一次性提示并打印结果。"""
    model = _resolve_model(config)

    cwd = Path.cwd()
    tools, system_prompt, _, _ = await _build_runtime(config, cwd)

    stream_fn = _make_stream_fn(model)

    agent = Agent(
        AgentOptions(
            model=model,
            system_prompt=system_prompt,
            tools=tools,
            stream_fn=stream_fn,
            thinking_level=ModelThinkingLevel(config.thinking_level),
        )
    )
    agent.subscribe(PrintRenderer(output_format))

    await agent.prompt(expand_file_references(prompt, cwd))


async def run_interactive_mode(
    config,
    requested_session_id: str | None = None,
    continue_latest: bool = False,
) -> None:
    """运行交互式 REPL 模式。"""
    from pi.tui.interactive import InteractiveSession

    model = _resolve_model(config)

    cwd = Path.cwd()
    tools, system_prompt, commands, skills = await _build_runtime(config, cwd)

    stream_fn = _make_stream_fn(model)
    session_id = await resolve_session_id(
        config.sessions_dir,
        requested_id=requested_session_id,
        continue_latest=continue_latest,
    )
    session_storage = JsonlStorage(session_path(config.sessions_dir, session_id))

    session = InteractiveSession(
        model=model,
        system_prompt=system_prompt,
        tools=tools,
        stream_fn=stream_fn,
        session_id=session_id,
        session_storage=session_storage,
        sessions_dir=config.sessions_dir,
        thinking_level=ModelThinkingLevel(config.thinking_level),
        commands=commands,
        skills=skills,
        cwd=cwd,
        history_file=config.config_dir / "history",
    )
    await session.run()


async def print_sessions(config) -> None:
    """输出可恢复会话列表。"""
    for item in await list_sessions(config.sessions_dir):
        updated = item.updated_at.astimezone().strftime("%Y-%m-%d %H:%M")
        click.echo(f"{item.session_id:32s}  {item.message_count:4d}  {updated}  {item.preview}")


def _print_oauth_event(event: OAuthEvent) -> None:
    if event.type == "device_code":
        click.echo(f"Open: {event.verification_uri}")
        click.echo(f"Code: {event.user_code}")
    elif event.message:
        click.echo(event.message)


async def oauth_login(provider_id: str) -> None:
    register_xai_oauth()
    await login_oauth(provider_id, _print_oauth_event)
    click.echo(f"Logged in: {provider_id}")


async def oauth_logout(provider_id: str) -> None:
    await get_default_oauth_store().delete(provider_id)
    click.echo(f"Logged out: {provider_id}")


async def print_auth_providers() -> None:
    for provider_id in await get_default_oauth_store().list():
        click.echo(f"{provider_id:24s} OAuth")


@click.command()
@click.option("-p", "--prompt", "prompt_text", default=None, help="One-shot prompt (print mode)")
@click.option("-m", "--model", "model_id", default=None, help="Model ID to use")
@click.option("--session", "session_id", default=None, help="Resume or create a session ID")
@click.option("-c", "--continue", "continue_session", is_flag=True, help="Resume latest session")
@click.option("--list-sessions", is_flag=True, help="List saved sessions and exit")
@click.option("--login", "login_provider", default=None, help="Login to an OAuth provider")
@click.option("--logout", "logout_provider", default=None, help="Remove stored OAuth credentials")
@click.option("--auth-list", is_flag=True, help="List stored OAuth providers and exit")
@click.option("--no-skills", is_flag=True, help="Disable skills discovery")
@click.option("--setup", "setup_config", is_flag=True, help="Run model setup and exit")
@click.option(
    "--output",
    "output_format",
    type=click.Choice(["text", "json", "jsonl"]),
    default="text",
    show_default=True,
    help="Print-mode output format",
)
@click.option(
    "--thinking",
    type=click.Choice([level.value for level in ModelThinkingLevel]),
    default=None,
    help="Reasoning effort",
)
@click.option(
    "--list-models", "list_models_flag", is_flag=True, help="List available models and exit"
)
@click.option("--version", is_flag=True, help="Show version and exit")
def main(
    prompt_text,
    model_id,
    session_id,
    continue_session,
    list_sessions,
    login_provider,
    logout_provider,
    auth_list,
    no_skills,
    setup_config,
    output_format,
    thinking,
    list_models_flag,
    version,
):
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
    if login_provider:
        asyncio.run(oauth_login(login_provider))
        return
    if logout_provider:
        asyncio.run(oauth_logout(logout_provider))
        return
    if auth_list:
        asyncio.run(print_auth_providers())
        return
    if setup_config:
        run_setup(config)
        return
    if not config.is_configured and prompt_text is None and click.get_text_stream("stdin").isatty():
        run_setup(config)
    if list_sessions:
        asyncio.run(print_sessions(config))
        return
    if model_id:
        config.model = model_id
    if thinking:
        config.thinking_level = thinking
    if no_skills:
        config.enable_skills = False

    if prompt_text is not None:
        asyncio.run(run_print_mode(prompt_text, config, output_format))
    else:
        if output_format != "text":
            raise click.UsageError("--output json/jsonl requires --prompt")
        asyncio.run(run_interactive_mode(config, session_id, continue_session))


if __name__ == "__main__":
    main()
