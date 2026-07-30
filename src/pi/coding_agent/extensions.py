"""编码 agent 的显式扩展加载与注册接口。"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import inspect
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pi.agent.types import AgentTool
from pi.ai.models import register_model
from pi.ai.providers.base import BaseProvider
from pi.ai.providers.registry import register_provider
from pi.ai.types import Model

ExtensionCommand = Callable[[str, Any], str | None | Awaitable[str | None]]


@dataclass
class ExtensionContext:
    """扩展可修改的受控运行时注册表。"""

    tools: list[AgentTool] = field(default_factory=list)
    system_prompt_sections: list[str] = field(default_factory=list)
    commands: dict[str, ExtensionCommand] = field(default_factory=dict)

    def add_tool(self, tool: AgentTool) -> None:
        if any(existing.name == tool.name for existing in self.tools):
            raise ValueError(f"Duplicate extension tool: {tool.name}")
        self.tools.append(tool)

    def add_model(self, model: Model) -> None:
        register_model(model)

    def add_provider(self, provider: BaseProvider) -> None:
        register_provider(provider)

    def add_system_prompt(self, text: str) -> None:
        if text.strip():
            self.system_prompt_sections.append(text.strip())

    def add_command(self, name: str, command: ExtensionCommand) -> None:
        normalized = name.strip().lower().lstrip("/")
        if not normalized or any(character.isspace() for character in normalized):
            raise ValueError(f"Invalid extension command: {name}")
        if normalized in self.commands:
            raise ValueError(f"Duplicate extension command: {normalized}")
        self.commands[normalized] = command


async def _run_setup(setup: Any, context: ExtensionContext, source: str) -> None:
    if not callable(setup):
        raise TypeError(f"Extension has no callable setup(context): {source}")
    result = setup(context)
    if inspect.isawaitable(result):
        await result


async def _load_file(path: Path, context: ExtensionContext) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Extension file not found: {path}")
    digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
    module_name = f"pi_local_extension_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load extension: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        await _run_setup(getattr(module, "setup", None), context, str(path))
    except Exception:
        sys.modules.pop(module_name, None)
        raise


async def load_extensions(
    paths: list[Path],
    enable_entrypoints: bool = False,
) -> ExtensionContext:
    """按配置顺序加载扩展，并返回聚合注册内容。"""
    context = ExtensionContext()
    for path in paths:
        await _load_file(path.resolve(), context)
    if enable_entrypoints:
        entrypoints = importlib.metadata.entry_points(group="pi.extensions")
        for entrypoint in sorted(entrypoints, key=lambda item: item.name):
            loaded = entrypoint.load()
            setup = getattr(loaded, "setup", loaded)
            await _run_setup(setup, context, f"entrypoint:{entrypoint.name}")
    return context
