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
ExtensionEventHandler = Callable[["ExtensionEvent", Any], None | Awaitable[None]]

_EXTENSION_EVENT_TYPES = {
    "agent_event",
    "session_shutdown",
    "session_start",
    "session_switch",
}


@dataclass(frozen=True)
class ExtensionEvent:
    """发送给扩展生命周期处理器的事件。"""

    type: str
    data: Any = None


@dataclass(frozen=True)
class ExtensionFailure:
    """被隔离的扩展处理器错误。"""

    source: str
    event_type: str
    error: Exception


@dataclass
class ExtensionContext:
    """扩展可修改的受控运行时注册表。"""

    tools: list[AgentTool] = field(default_factory=list)
    system_prompt_sections: list[str] = field(default_factory=list)
    commands: dict[str, ExtensionCommand] = field(default_factory=dict)
    event_handlers: dict[str, list[tuple[str, ExtensionEventHandler]]] = field(default_factory=dict)
    _loading_source: str = field(default="extension", init=False, repr=False)

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

    def on(self, event_type: str, handler: ExtensionEventHandler) -> None:
        """注册生命周期事件处理器。"""
        if event_type not in _EXTENSION_EVENT_TYPES:
            raise ValueError(f"Unknown extension event: {event_type}")
        if not callable(handler):
            raise TypeError(f"Extension event handler is not callable: {event_type}")
        self.event_handlers.setdefault(event_type, []).append((self._loading_source, handler))

    async def emit(
        self,
        event_type: str,
        data: Any = None,
        agent: Any = None,
    ) -> list[ExtensionFailure]:
        """按注册顺序触发生命周期事件，并隔离单个处理器错误。"""
        event = ExtensionEvent(type=event_type, data=data)
        failures = []
        for source, handler in self.event_handlers.get(event_type, []):
            try:
                result = handler(event, agent)
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                failures.append(
                    ExtensionFailure(
                        source=source,
                        event_type=event_type,
                        error=exc,
                    )
                )
        return failures


async def _run_setup(setup: Any, context: ExtensionContext, source: str) -> None:
    if not callable(setup):
        raise TypeError(f"Extension has no callable setup(context): {source}")
    previous_source = context._loading_source
    context._loading_source = source
    try:
        result = setup(context)
        if inspect.isawaitable(result):
            await result
    finally:
        context._loading_source = previous_source


async def _load_file(path: Path, context: ExtensionContext) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Extension file not found: {path}")
    digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
    module_name = f"piy_local_extension_{digest}"
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
        entrypoints = importlib.metadata.entry_points(group="piy.extensions")
        for entrypoint in sorted(entrypoints, key=lambda item: item.name):
            loaded = entrypoint.load()
            setup = getattr(loaded, "setup", loaded)
            await _run_setup(setup, context, f"entrypoint:{entrypoint.name}")
    return context
