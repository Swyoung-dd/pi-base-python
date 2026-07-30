"""CLI、SDK 和 RPC 共用的编码 agent 运行时资源构建器。"""

from __future__ import annotations

from pathlib import Path

from pi.agent.types import AgentTool
from pi.ai.providers.registry import get_provider
from pi.coding_agent.config import Config
from pi.coding_agent.context_files import format_context_files, load_context_files
from pi.coding_agent.extensions import load_extensions
from pi.coding_agent.prompt_templates import load_prompt_templates
from pi.coding_agent.skills import format_skills_for_prompt, load_skills
from pi.coding_agent.system_prompt import build_system_prompt
from pi.coding_agent.themes import load_theme
from pi.coding_agent.tools import (
    create_bash_tool,
    create_edit_tool,
    create_find_tool,
    create_grep_tool,
    create_ls_tool,
    create_read_tool,
    create_write_tool,
)


def build_tools() -> list[AgentTool]:
    """创建默认编码工具集。"""
    return [
        create_read_tool(),
        create_write_tool(),
        create_edit_tool(),
        create_bash_tool(),
        create_ls_tool(),
        create_find_tool(),
        create_grep_tool(),
    ]


async def build_runtime_resources(config: Config, cwd: Path):
    """加载工具、扩展、skills、模板、系统提示和主题。"""
    extensions = await load_extensions(
        config.extension_paths,
        config.enable_entrypoint_extensions,
    )
    tools = [*build_tools(), *extensions.tools]
    tool_names = [tool.name for tool in tools]
    if len(tool_names) != len(set(tool_names)):
        raise ValueError("Extension tool name conflicts with a built-in tool")
    skills = []
    if config.enable_skills:
        skill_paths = [
            *config.skill_paths,
            config.config_dir / "skills",
            Path.home() / ".piy" / "skills",
        ]
        skills = load_skills([path for path in skill_paths if path.exists()])
    prompt_templates = []
    if config.enable_prompt_templates:
        prompt_paths = [
            *config.prompt_paths,
            config.config_dir / "prompts",
            Path.home() / ".piy" / "prompts",
        ]
        prompt_templates = load_prompt_templates([path for path in prompt_paths if path.exists()])
    context_files = (
        format_context_files(load_context_files(cwd)) if config.enable_context_files else ""
    )
    sections = [
        config.system_prompt,
        *extensions.system_prompt_sections,
        build_system_prompt(cwd, tool_names),
        context_files,
        format_skills_for_prompt(skills),
    ]
    theme = load_theme(
        config.theme,
        [config.config_dir / "themes", Path.home() / ".piy" / "themes"],
    )
    return (
        tools,
        "\n\n".join(section for section in sections if section),
        extensions,
        skills,
        prompt_templates,
        theme,
    )


def make_stream_fn():
    """创建按当前模型动态选择 provider 的 stream 函数。"""

    async def stream_fn(model, context, options):
        provider = get_provider(model.provider)
        if provider is None:
            raise RuntimeError(f"No provider registered for: {model.provider}")
        return await provider.stream(model, context, options)

    return stream_fn
