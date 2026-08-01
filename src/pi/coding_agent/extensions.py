"""编码 agent 的显式扩展加载与注册接口。"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import inspect
import logging
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pi.agent.types import AgentTool
from pi.ai.models import register_model
from pi.ai.providers.base import BaseProvider
from pi.ai.providers.registry import register_provider
from pi.ai.types import Model

ExtensionCommand = Callable[[str, Any], str | None | Awaitable[str | None]]
ExtensionEventHandler = Callable[["ExtensionEvent", Any], None | Awaitable[None]]
_logger = logging.getLogger("piy.extensions")


def _utc_now() -> str:
    """返回当前 UTC 时间的 ISO 格式字符串。"""
    return datetime.now(UTC).isoformat()


_EXTENSION_EVENT_TYPES = {
    "agent_event",
    "session_shutdown",
    "session_start",
    "session_switch",
    "transform_context",
    "before_request",
    "after_response",
    "before_tool",
    "after_tool",
    "before_compaction",
    "before_navigation",
    "resource_reload",
}


@dataclass(frozen=True)
class ExtensionEvent:
    """发送给扩展生命周期处理器的事件。"""

    type: str
    data: Any = None
    source: str = ""


@dataclass(frozen=True)
class ExtensionFailure:
    """被隔离的扩展处理器错误。"""

    source: str
    event_type: str
    error: Exception
    handler_index: int = 0


@dataclass
class ExtensionSource:
    """扩展来源信息，用于诊断和冲突检测。"""

    name: str
    path: str
    is_entrypoint: bool = False
    loaded_at: str = ""
    active: bool = True


@dataclass
class ExtensionConflict:
    """扩展冲突诊断信息。"""

    kind: str  # "tool_name" / "command_name" / "event_handler"
    sources: list[str] = field(default_factory=list)
    detail: str = ""


@dataclass
class ExtensionContext:
    """扩展可修改的受控运行时注册表。"""

    tools: list[AgentTool] = field(default_factory=list)
    system_prompt_sections: list[str] = field(default_factory=list)
    commands: dict[str, ExtensionCommand] = field(default_factory=dict)
    event_handlers: dict[str, list[tuple[str, ExtensionEventHandler]]] = field(default_factory=dict)
    _loading_source: str = field(default="extension", init=False, repr=False)
    _sources: list[ExtensionSource] = field(default_factory=list, init=False, repr=False)
    _tool_sources: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _prompt_sources: list[str] = field(default_factory=list, init=False, repr=False)
    _command_sources: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _context_transformers: list[tuple[str, Any]] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    def add_tool(self, tool: AgentTool) -> None:
        if any(existing.name == tool.name for existing in self.tools):
            raise ValueError(f"Duplicate extension tool: {tool.name}")
        self.tools.append(tool)
        self._tool_sources[tool.name] = self._loading_source
        self._check_conflicts()

    def add_model(self, model: Model) -> None:
        register_model(model)

    def add_provider(self, provider: BaseProvider) -> None:
        register_provider(provider)

    def add_system_prompt(self, text: str) -> None:
        if text.strip():
            self.system_prompt_sections.append(text.strip())
            self._prompt_sources.append(self._loading_source)

    def add_command(self, name: str, command: ExtensionCommand) -> None:
        normalized = name.strip().lower().lstrip("/")
        if not normalized or any(character.isspace() for character in normalized):
            raise ValueError(f"Invalid extension command: {name}")
        if normalized in self.commands:
            raise ValueError(f"Duplicate extension command: {normalized}")
        self.commands[normalized] = command
        self._command_sources[normalized] = self._loading_source

    def add_context_transformer(self, transformer: Any) -> None:
        """注册上下文变换器，在发送给模型前修改 system_prompt 或 messages。"""
        self._context_transformers.append((self._loading_source, transformer))

    @property
    def context_transformers(self) -> list[Any]:
        """返回已注册的上下文变换器列表（不含来源信息）。"""
        return [transformer for _source, transformer in self._context_transformers]

    async def transform_context(
        self,
        system_prompt: str,
        messages: list,
        agent: Any = None,
    ) -> tuple[str, list]:
        """应用所有已注册的上下文变换器。"""
        for _source, transformer in self._context_transformers:
            try:
                result = transformer(system_prompt, messages, agent)
                if inspect.isawaitable(result):
                    result = await result
                if isinstance(result, tuple) and len(result) == 2:
                    system_prompt, messages = result
            except Exception as exc:
                _logger.warning("Context transformer from %s failed: %s", _source, exc)
        return system_prompt, messages

    def on(self, event_type: str, handler: ExtensionEventHandler) -> None:
        """注册生命周期事件处理器。"""
        if event_type not in _EXTENSION_EVENT_TYPES:
            raise ValueError(f"Unknown extension event: {event_type}")
        if not callable(handler):
            raise TypeError(f"Extension event handler is not callable: {event_type}")
        self.event_handlers.setdefault(event_type, []).append((self._loading_source, handler))

    def register_source(self, source: ExtensionSource) -> None:
        """记录扩展来源信息。"""
        self._sources.append(source)

    def sources(self) -> list[ExtensionSource]:
        """返回所有已注册的扩展来源。"""
        return list(self._sources)

    def conflicts(self) -> list[ExtensionConflict]:
        """检测并返回扩展冲突。"""
        return self._detect_conflicts()

    def _detect_conflicts(self) -> list[ExtensionConflict]:
        """检测工具名、命令名冲突。"""
        result: list[ExtensionConflict] = []
        tool_names: dict[str, list[str]] = {}
        for tool in self.tools:
            tool_names.setdefault(tool.name, []).append(self._loading_source)
        for name, srcs in tool_names.items():
            if len(srcs) > 1:
                result.append(
                    ExtensionConflict(
                        kind="tool_name",
                        sources=srcs,
                        detail=f"Duplicate tool: {name}",
                    )
                )
        return result

    def _check_conflicts(self) -> None:
        """检查冲突并记录（不抛出异常，由调用方决定是否处理）。"""
        pass

    async def reload(self) -> list[ExtensionFailure]:
        """重新加载所有扩展来源。"""
        failures: list[ExtensionFailure] = []
        saved_sources = list(self._sources)
        self.tools.clear()
        self.system_prompt_sections.clear()
        self.commands.clear()
        self.event_handlers.clear()
        self._context_transformers.clear()
        self._sources.clear()
        self._tool_sources.clear()
        self._prompt_sources.clear()
        self._command_sources.clear()
        for source in saved_sources:
            if not source.active:
                self._sources.append(source)
                continue
            try:
                if source.is_entrypoint:
                    entrypoints = importlib.metadata.entry_points(group="piy.extensions")
                    for ep in sorted(entrypoints, key=lambda item: item.name):
                        if ep.name == source.name:
                            loaded = ep.load()
                            setup = getattr(loaded, "setup", loaded)
                            await _run_setup(setup, self, f"entrypoint:{ep.name}")
                else:
                    await _load_file(Path(source.path), self)
            except Exception as exc:
                failures.append(
                    ExtensionFailure(
                        source=source.name,
                        event_type="reload",
                        error=exc,
                    )
                )
        return failures

    def unload(self, source_name: str) -> bool:
        """卸载指定来源的扩展，移除其注册的工具、提示、命令和事件处理器。"""
        found = False
        for source in self._sources:
            if source.name == source_name:
                source.active = False
                found = True
        if not found:
            return False
        # Remove tools registered by this source
        self.tools = [t for t in self.tools if self._tool_sources.get(t.name) != source_name]
        self._tool_sources = {k: v for k, v in self._tool_sources.items() if v != source_name}
        # Remove system prompt sections from this source
        kept_prompts: list[str] = []
        kept_prompt_sources: list[str] = []
        for prompt, src in zip(self.system_prompt_sections, self._prompt_sources, strict=True):
            if src != source_name:
                kept_prompts.append(prompt)
                kept_prompt_sources.append(src)
        self.system_prompt_sections = kept_prompts
        self._prompt_sources = kept_prompt_sources
        # Remove commands from this source
        self.commands = {
            k: v for k, v in self.commands.items() if self._command_sources.get(k) != source_name
        }
        self._command_sources = {k: v for k, v in self._command_sources.items() if v != source_name}
        # Remove event handlers from this source
        for event_type in list(self.event_handlers):
            self.event_handlers[event_type] = [
                (src, h) for src, h in self.event_handlers[event_type] if src != source_name
            ]
            if not self.event_handlers[event_type]:
                del self.event_handlers[event_type]
        # Remove context transformers from this source
        self._context_transformers = [
            (src, t) for src, t in self._context_transformers if src != source_name
        ]
        return found

    async def emit(
        self,
        event_type: str,
        data: Any = None,
        agent: Any = None,
    ) -> list[ExtensionFailure]:
        """按注册顺序触发生命周期事件，并隔离单个处理器错误。"""
        event = ExtensionEvent(type=event_type, data=data, source=self._loading_source)
        failures = []
        for index, (source, handler) in enumerate(self.event_handlers.get(event_type, [])):
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
                        handler_index=index,
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
        context.register_source(
            ExtensionSource(
                name=path.stem,
                path=str(path.resolve()),
                is_entrypoint=False,
                loaded_at=_utc_now(),
            )
        )
        await _load_file(path.resolve(), context)
    if enable_entrypoints:
        entrypoints = importlib.metadata.entry_points(group="piy.extensions")
        for entrypoint in sorted(entrypoints, key=lambda item: item.name):
            context.register_source(
                ExtensionSource(
                    name=entrypoint.name,
                    path=str(entrypoint.value),
                    is_entrypoint=True,
                    loaded_at=_utc_now(),
                )
            )
            loaded = entrypoint.load()
            setup = getattr(loaded, "setup", loaded)
            await _run_setup(setup, context, f"entrypoint:{entrypoint.name}")
    return context
