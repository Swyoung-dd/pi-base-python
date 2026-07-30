"""piY 编码 agent 的程序化嵌入接口。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pi.agent.agent import Agent, AgentOptions
from pi.agent.session import JsonlStorage, SessionStorage
from pi.agent.tools import ToolContext
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
    _started: bool = field(default=False, init=False, repr=False)

    async def start(self) -> None:
        """恢复会话并触发 session_start。重复调用无操作。"""
        if self._started:
            return
        await self.agent.restore()
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
            self.extension_failures.extend(
                await self.extensions.emit(
                    "session_shutdown",
                    {"session_id": self.session_id},
                    self.agent,
                )
            )
            self._started = False

    async def _on_agent_event(self, event: AgentEvent) -> None:
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
            tool_context=ToolContext(cwd=resolved_cwd),
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
    agent.subscribe(runtime._on_agent_event)
    return runtime
