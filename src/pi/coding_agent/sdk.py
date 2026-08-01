"""piY 编码 agent 的程序化嵌入接口。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pi.agent.agent import Agent, AgentOptions
from pi.agent.session import JsonlStorage, SessionStorage
from pi.agent.tools import LocalExecutionEnv, ToolContext
from pi.agent.types import AgentEvent, AgentMessage
from pi.ai.models import get_model
from pi.ai.types import Model, ModelThinkingLevel
from pi.coding_agent.config import Config, load_config
from pi.coding_agent.extensions import ExtensionContext, ExtensionFailure
from pi.coding_agent.prompt_templates import PromptTemplate
from pi.coding_agent.runtime import build_runtime_resources, make_stream_fn
from pi.coding_agent.sessions import new_session_id, session_path
from pi.coding_agent.skills import Skill
from pi.coding_agent.themes import Theme
from pi.coding_agent.tracing import Tracer, get_tracer


@dataclass
class CodingAgent:
    """包含 Agent 与已加载编码资源的可关闭运行时。"""

    agent: Agent
    config: Config
    cwd: Path
    extensions: ExtensionContext
    skills: list[Skill]
    prompt_templates: list[PromptTemplate]
    theme: Theme
    session_id: str | None = None
    extension_failures: list[ExtensionFailure] = field(default_factory=list)
    tracer: Tracer = field(default_factory=get_tracer)
    _started: bool = field(default=False, init=False, repr=False)

    async def start(self) -> None:
        """恢复会话并触发 session_start。重复调用无操作。"""
        if self._started:
            return
        await self.agent.restore()
        self.tracer.trace_session_event("session_start", self.session_id)
        self.extension_failures.extend(
            await self.extensions.emit(
                "session_start",
                {"session_id": self.session_id},
                self.agent,
            )
        )
        self._started = True

    async def prompt(self, text: str) -> list[AgentMessage]:
        """提交提示并返回本次新增的消息。"""
        await self.start()
        before = len(self.agent.state.messages)
        await self.agent.prompt(text)
        return self.agent.state.messages[before:]

    async def set_model(self, model: Model, persist: bool = True) -> None:
        """切换模型，并可选写入当前持久会话。"""
        self.agent.set_model(model)
        storage = self.agent.session_storage
        if persist and storage is not None:
            await storage.append_model_change(model.provider, model.id)

    async def close(self) -> None:
        """中止活动请求并触发 session_shutdown。"""
        if self.agent.is_busy:
            self.agent.abort()
            await self.agent.wait_for_idle()
        if self._started:
            self.tracer.trace_session_event("session_shutdown", self.session_id)
            self.extension_failures.extend(
                await self.extensions.emit(
                    "session_shutdown",
                    {"session_id": self.session_id},
                    self.agent,
                )
            )
            self._started = False
        self.tracer.close()

    async def __aenter__(self) -> CodingAgent:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def _on_agent_event(self, event: AgentEvent) -> None:
        if event.type == "tool_execution_end" and event.result:
            self.tracer.trace_tool_call(
                tool_name=event.tool_name,
                tool_call_id=event.tool_call_id,
                is_error=event.result.is_error,
            )
        elif event.type == "message_end":
            model = event.message.model
            provider = event.message.provider
            self.tracer.trace_llm_response(
                provider=provider,
                model=model,
                stop_reason=event.message.stop_reason,
                usage=event.message.usage.model_dump(mode="json") if event.message.usage else None,
            )
        elif event.type == "context_compacted":
            self.tracer.trace_compaction(
                original_tokens=event.original_tokens,
                compacted_tokens=event.compacted_tokens,
                dropped_messages=event.dropped_messages,
            )
        self.extension_failures.extend(await self.extensions.emit("agent_event", event, self.agent))


async def create_coding_agent(
    config: Config | None = None,
    cwd: Path | None = None,
    session_id: str | None = None,
    persist_session: bool = False,
    session_storage: SessionStorage | None = None,
    project_trusted: bool = False,
) -> CodingAgent:
    """创建带默认编码工具和资源的 piY 运行时。"""
    resolved_cwd = (cwd or Path.cwd()).resolve()
    resolved_config = config or load_config(
        resolved_cwd / ".piy",
        project_trusted=project_trusted,
    )
    configured_model = get_model(resolved_config.model, resolved_config.provider)
    if configured_model is None:
        raise ValueError(f"Unknown or ambiguous model: {resolved_config.model}")
    (
        tools,
        system_prompt,
        extensions,
        skills,
        prompt_templates,
        theme,
    ) = await build_runtime_resources(resolved_config, resolved_cwd)
    resolved_session_id = session_id
    storage = session_storage
    if storage is None and persist_session:
        resolved_session_id = resolved_session_id or new_session_id()
        storage = JsonlStorage(session_path(resolved_config.sessions_dir, resolved_session_id))
    model = configured_model
    if storage is not None:
        selection = await storage.get_model_selection()
        if selection is not None:
            restored_model = get_model(selection[1], selection[0])
            if restored_model is not None:
                model = restored_model
        elif not await storage.get_entries():
            await storage.append_model_change(model.provider, model.id)
    agent = Agent(
        AgentOptions(
            model=model,
            system_prompt=system_prompt,
            tools=tools,
            stream_fn=make_stream_fn(),
            session_id=resolved_session_id,
            session_storage=storage,
            tool_context=ToolContext(
                cwd=resolved_cwd,
                env=LocalExecutionEnv(resolved_cwd),
            ),
            before_tool_call=_make_before_tool_hook(extensions),
            after_tool_call=_make_after_tool_hook(extensions),
            temperature=resolved_config.temperature,
            max_tokens=resolved_config.max_tokens,
            thinking_level=ModelThinkingLevel(resolved_config.thinking_level),
        )
    )
    runtime = CodingAgent(
        agent=agent,
        config=resolved_config,
        cwd=resolved_cwd,
        extensions=extensions,
        skills=skills,
        prompt_templates=prompt_templates,
        theme=theme,
        session_id=resolved_session_id,
    )
    agent._tool_context.state["extensions"] = extensions
    agent.subscribe(runtime._on_agent_event)
    return runtime


def _make_before_tool_hook(extensions):
    """Create a before_tool_call hook that emits extension events."""

    async def hook(call, tool):
        await extensions.emit("before_tool", {"call": call, "tool": tool})

    return hook


def _make_after_tool_hook(extensions):
    """Create an after_tool_call hook that emits extension events."""

    async def hook(call, tool, result):
        await extensions.emit("after_tool", {"call": call, "tool": tool, "result": result})

    return hook
