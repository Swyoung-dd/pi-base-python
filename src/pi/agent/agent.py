"""低层 agent 循环之上的有状态 Agent 包装器。

负责持有当前对话记录、发出生命周期事件、执行工具，
并提供 steer 与 follow-up 消息的排队 API。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

from pi.agent.agent_loop import AgentLoopResult, run_agent_loop
from pi.agent.compaction import (
    CompactionResult,
    apply_compaction_summary,
    compact_messages,
    estimate_context_tokens,
    estimate_context_tokens_with_overhead,
    format_messages_for_summary,
    prepare_compaction,
)
from pi.agent.session.base import SessionStorage
from pi.agent.types import (
    AfterToolCallFn,
    AgentAssistantMessage,
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentMessage,
    AgentPhase,
    AgentState,
    AgentTool,
    BeforeToolCallFn,
    MessageEndEvent,
    MessageStartEvent,
    QueueMode,
    ToolExecutionMode,
    TurnEndEvent,
    create_user_message,
)
from pi.ai.streaming import EventStream
from pi.ai.types import (
    Context,
    ImageContent,
    Model,
    ModelThinkingLevel,
    StreamOptions,
    TextContent,
    ThinkingContent,
    UserMessage,
)

StreamFn = Callable[
    [Model, Any, StreamOptions | None],
    Coroutine[Any, Any, EventStream],
]

AgentListener = Callable[[AgentEvent], Coroutine[Any, Any, None]]

_COMPACTION_SYSTEM_PROMPT = """你负责压缩编码 agent 的早期会话。
请生成可供后续模型继续工作的结构化摘要，必须保留：
- 用户目标、约束和偏好
- 已完成工作、修改文件和验证结果
- 关键技术决策及其原因
- 当前错误、阻塞项和下一步
不要添加原会话中不存在的事实。"""


@dataclass
class AgentOptions:
    """构造 Agent 的选项。"""

    model: Model | None = None
    system_prompt: str = ""
    tools: list[AgentTool] = field(default_factory=list)
    stream_fn: StreamFn | None = None
    session_id: str | None = None
    session_storage: SessionStorage | None = None
    steering_mode: QueueMode = QueueMode.ONE_AT_A_TIME
    follow_up_mode: QueueMode = QueueMode.ONE_AT_A_TIME
    tool_execution: ToolExecutionMode = ToolExecutionMode.PARALLEL
    tool_context: Any = None
    context_token_limit: int | None = None
    compact_to_tokens: int | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    max_turns: int | None = None
    thinking_level: ModelThinkingLevel = ModelThinkingLevel.OFF
    thinking_budget_tokens: int | None = None
    before_tool_call: BeforeToolCallFn | None = None
    after_tool_call: AfterToolCallFn | None = None


@dataclass(frozen=True)
class _TurnState:
    """单个 turn 的冻结配置快照，确保 turn 内配置不被中途修改影响。"""

    model: Model | None
    thinking_level: ModelThinkingLevel
    system_prompt: str
    tools: list[AgentTool]
    tool_context: Any
    context_token_limit: int | None
    compact_to_tokens: int | None
    temperature: float | None
    max_tokens: int | None
    thinking_budget_tokens: int | None


@dataclass(frozen=True)
class ContextUsage:
    """当前模型上下文窗口的使用情况。"""

    tokens: int
    context_window: int
    percent: float


class _PendingQueue:
    def __init__(self, mode: QueueMode) -> None:
        self.mode = mode
        self._items: list[AgentMessage] = []

    def enqueue(self, msg: AgentMessage) -> None:
        self._items.append(msg)

    def has_items(self) -> bool:
        return len(self._items) > 0

    def drain(self) -> list[AgentMessage]:
        if self.mode == QueueMode.ALL:
            items = self._items[:]
            self._items.clear()
            return items
        if not self._items:
            return []
        return [self._items.pop(0)]

    def clear(self) -> None:
        self._items.clear()


class Agent:
    """agent 循环之上的有状态包装器。

    用法：
        agent = Agent(AgentOptions(model=..., stream_fn=..., tools=...))
        agent.subscribe(my_listener)
        await agent.prompt("Hello")
    """

    def __init__(self, options: AgentOptions) -> None:
        tools = options.tools[:]
        # 自动注入 compact 工具，允许 AI 主动压缩上下文
        if not any(t.name == "compact" for t in tools):
            from pi.coding_agent.tools.compact import create_compact_tool

            tools.append(create_compact_tool())
        self._state = AgentState(
            system_prompt=options.system_prompt,
            model=options.model,
            tools=tools,
            thinking_level=options.thinking_level.value,
        )
        self._listeners: list[AgentListener] = []
        self._steering_queue = _PendingQueue(options.steering_mode)
        self._follow_up_queue = _PendingQueue(options.follow_up_mode)
        self._stream_fn = options.stream_fn
        self._session_id = options.session_id
        self._session_storage = options.session_storage
        self._session_loaded = False
        self._tool_execution = options.tool_execution
        self._tool_context = options.tool_context
        if self._tool_context is not None:
            # 让工具在执行时能拿到当前 agent 引用（如 subagent 工具）。
            self._tool_context.state.setdefault("agent", self)
        model_context_limit = None
        if options.model is not None and options.model.context_window:
            reserved_tokens = options.max_tokens or options.model.max_tokens or 4096
            model_context_limit = max(1, options.model.context_window - reserved_tokens)
        self._context_token_limit = options.context_token_limit or model_context_limit
        self._compact_to_tokens = options.compact_to_tokens
        self._temperature = options.temperature
        self._max_tokens = options.max_tokens
        self._max_turns = options.max_turns
        self._thinking_level = options.thinking_level
        self._thinking_budget_tokens = options.thinking_budget_tokens
        self._before_tool_call = options.before_tool_call
        self._after_tool_call = options.after_tool_call
        self._phase: AgentPhase = AgentPhase.IDLE
        self._pending_writes: list = []
        self._abort_event = asyncio.Event()
        self._idle_event = asyncio.Event()
        self._idle_event.set()

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def is_busy(self) -> bool:
        return not self._idle_event.is_set()

    @property
    def context_token_limit(self) -> int | None:
        return self._context_token_limit

    @property
    def thinking_level(self) -> ModelThinkingLevel:
        """返回后续模型请求使用的思考级别。"""
        return self._thinking_level

    @property
    def stream_fn(self) -> StreamFn | None:
        """返回当前流函数，供子 agent 复用。"""
        return self._stream_fn

    @property
    def temperature(self) -> float | None:
        """返回当前温度设置。"""
        return self._temperature

    @property
    def max_tokens(self) -> int | None:
        """返回当前 max_tokens 设置。"""
        return self._max_tokens

    @property
    def session_storage(self) -> SessionStorage | None:
        """返回当前会话存储，供嵌入层执行显式状态操作。"""
        return self._session_storage

    def get_context_usage(self) -> ContextUsage | None:
        """返回当前会话上下文使用量；无窗口信息时返回 None。"""
        model = self._state.model
        if model is None or model.context_window <= 0:
            return None
        tokens = estimate_context_tokens(self._state.messages).tokens
        return ContextUsage(
            tokens=tokens,
            context_window=model.context_window,
            percent=tokens / model.context_window * 100,
        )

    async def compact(self, target_tokens: int | None = None) -> CompactionResult:
        """主动触发上下文压缩，可在任意时刻调用。

        Args:
            target_tokens: 压缩目标 token 数。默认使用 compact_to_tokens
                或 context_token_limit * 3/4。

        Returns:
            CompactionResult 包含压缩后的消息列表与统计信息。

        Raises:
            RuntimeError: 当 Agent 正在处理时抛出。
        """
        if self.is_busy:
            raise RuntimeError("Agent is busy. Wait for idle before compacting.")
        await self.restore()
        messages = self._state.messages
        if not messages:
            return CompactionResult(
                messages=[],
                original_tokens=0,
                compacted_tokens=0,
                dropped_messages=0,
            )
        model = self._state.model
        limit = self._context_token_limit
        if target_tokens is None:
            target_tokens = self._compact_to_tokens or (
                (limit * 3 // 4)
                if limit
                else (model.context_window * 3 // 4)
                if model and model.context_window
                else len(messages) * 50
            )
        estimated = estimate_context_tokens_with_overhead(
            messages,
            self._state.system_prompt,
            self._state.tools,
        )
        result = await self._compact_with_model(messages, target_tokens, estimated)
        self._state.messages = result.messages
        if self._session_storage is not None and result.dropped_messages > 0:
            await self._session_storage.append_compaction(
                result.messages,
                result.original_tokens,
                result.compacted_tokens,
                result.dropped_messages,
                result.usage,
            )
        return result

    def subscribe(self, listener: AgentListener) -> Callable[[], None]:
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener) if listener in self._listeners else None

    def steer(self, message: AgentMessage) -> None:
        self._steering_queue.enqueue(message)

    def follow_up(self, message: AgentMessage) -> None:
        self._follow_up_queue.enqueue(message)

    def clear_queues(self) -> None:
        self._steering_queue.clear()
        self._follow_up_queue.clear()

    def abort(self) -> None:
        self._abort_event.set()

    async def wait_for_idle(self) -> None:
        await self._idle_event.wait()

    async def restore(self) -> None:
        """从会话存储恢复当前分支的消息。"""
        if self._session_loaded:
            return
        if self._session_storage is not None:
            self._state.messages = await self._session_storage.get_context_messages()
        self._session_loaded = True

    async def switch_session(
        self,
        storage: SessionStorage,
        session_id: str | None = None,
    ) -> None:
        """切换会话存储，并恢复目标会话的当前分支。"""
        if self.is_busy:
            raise RuntimeError("Agent is already processing")
        self._session_storage = storage
        self._session_id = session_id
        self._session_loaded = False
        self.reset()
        await self.restore()

    def set_model(self, model: Model) -> None:
        """切换后续请求使用的模型。

        空闲时立即生效；运行中时排队到下一个 save point 应用。
        """
        if self.is_busy:
            async def apply() -> None:
                self._state.model = model
                self._context_token_limit = (
                    max(1, model.context_window - (self._max_tokens or model.max_tokens or 4096))
                    if model.context_window
                    else None
                )
                if self._session_storage is not None:
                    await self._session_storage.append_model_change(
                        model.provider, model.id,
                    )
            self._pending_writes.append(apply)
            return
        self._state.model = model
        self._context_token_limit = (
            max(1, model.context_window - (self._max_tokens or model.max_tokens or 4096))
            if model.context_window
            else None
        )

    def set_thinking_level(self, level: ModelThinkingLevel) -> None:
        """切换后续请求使用的思考级别。

        空闲时立即生效；运行中时排队到下一个 save point 应用。
        """
        if self.is_busy:
            async def apply() -> None:
                self._thinking_level = level
                self._state.thinking_level = level.value
                if self._session_storage is not None:
                    await self._session_storage.append_thinking_level_change(
                        level.value,
                    )
            self._pending_writes.append(apply)
            return
        self._thinking_level = level
        self._state.thinking_level = level.value

    @property
    def phase(self) -> AgentPhase:
        """返回当前运行时阶段。"""
        return self._phase

    def _create_turn_state(self) -> _TurnState:
        """创建当前配置的冻结快照，用于本次 turn 的全部请求。"""
        return _TurnState(
            model=self._state.model,
            thinking_level=self._thinking_level,
            system_prompt=self._state.system_prompt,
            tools=self._state.tools[:],
            tool_context=self._tool_context,
            context_token_limit=self._context_token_limit,
            compact_to_tokens=self._compact_to_tokens,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            thinking_budget_tokens=self._thinking_budget_tokens,
        )

    async def _flush_pending_writes(self) -> None:
        """在 save point 应用排队的配置变更。"""
        if not self._pending_writes:
            return
        for write in self._pending_writes:
            await write()
        self._pending_writes.clear()

    def reset(self) -> None:
        self._state.messages.clear()
        self._state.is_streaming = False
        self._state.streaming_message = None
        self._state.pending_tool_calls.clear()
        self._state.error_message = None
        self.clear_queues()

    async def prompt(self, text: str, images: list[ImageContent] | None = None) -> None:
        await self.restore()
        if self.is_busy:
            raise RuntimeError("Agent is already processing. Use steer() or wait.")
        msg = create_user_message(text, images)
        await self._run([msg])

    async def _run(self, messages: list[AgentMessage]) -> None:
        if self._stream_fn is None:
            raise RuntimeError("No stream function configured")
        if self._state.model is None:
            raise RuntimeError("No model configured")

        self._abort_event.clear()
        self._idle_event.clear()
        self._state.is_streaming = True
        self._phase = AgentPhase.TURN
        previous_messages = self._state.messages[:]
        self._state.messages.extend(messages)

        try:
            current_context = previous_messages
            pending_prompts = messages
            while pending_prompts:
                turn_state = self._create_turn_state()
                result: AgentLoopResult = await run_agent_loop(
                    prompts=pending_prompts,
                    context=AgentContext(
                        system_prompt=turn_state.system_prompt,
                        messages=current_context,
                        tools=turn_state.tools,
                    ),
                    model=turn_state.model,
                    stream_fn=self._stream_fn,
                    sink=self._process_event,
                    options=StreamOptions(
                        temperature=turn_state.temperature,
                        max_tokens=turn_state.max_tokens,
                        session_id=self._session_id,
                        abort_event=self._abort_event,
                        context_token_limit=turn_state.context_token_limit,
                        compact_to_tokens=turn_state.compact_to_tokens,
                        thinking_level=turn_state.thinking_level,
                        thinking_budget_tokens=turn_state.thinking_budget_tokens,
                    ),
                    tool_execution=self._tool_execution,
                    tool_context=turn_state.tool_context,
                    get_steering_messages=self._steering_queue.drain,
                    compact_fn=self._compact_with_model,
                    max_turns=self._max_turns,
                    before_tool_call=self._before_tool_call,
                    after_tool_call=self._after_tool_call,
                    )
                current_context = result.context_messages
                if self._session_storage is not None:
                    # P0-2: 始终追加原始消息，压缩时额外追加检查点
                    for message in result.messages:
                        await self._session_storage.append_message(message)
                    if result.compactions:
                        compaction = result.compactions[-1]
                        summary_text = ""
                        if current_context:
                            first = current_context[0]
                            if hasattr(first, "content") and isinstance(first.content, str):
                                summary_text = first.content
                        await self._session_storage.append_compaction(
                            current_context,
                            compaction.original_tokens,
                            compaction.compacted_tokens,
                            compaction.dropped_messages,
                            compaction.usage,
                            summary=summary_text,
                           retained_tail=current_context,
                        )
                # Save point: apply queued config changes for next turn
                await self._flush_pending_writes()
                last_assistant = next(
                    (
                        message
                        for message in reversed(result.messages)
                        if isinstance(message, AgentAssistantMessage)
                    ),
                    None,
                )
                if last_assistant and last_assistant.stop_reason in ("aborted", "error"):
                    break
                pending_prompts = self._steering_queue.drain()
                if not pending_prompts:
                    pending_prompts = self._follow_up_queue.drain()

            self._state.messages = current_context
        except Exception as exc:
            now = int(time.time() * 1000)
            failure = AgentAssistantMessage(
                content=[TextContent(text="")],
                api=self._state.model.api,
                provider=self._state.model.provider,
                model=self._state.model.id,
                stop_reason="error",
                error_message=str(exc),
                timestamp=now,
            )
            await self._process_event(MessageStartEvent(message=failure))
            await self._process_event(MessageEndEvent(message=failure))
            await self._process_event(TurnEndEvent(message=failure, tool_results=[]))
            await self._process_event(AgentEndEvent(messages=[failure]))
        finally:
            self._state.is_streaming = False
            self._phase = AgentPhase.IDLE
            self._state.streaming_message = None
            self._state.pending_tool_calls.clear()
            self._idle_event.set()

    async def _compact_with_model(
        self,
        messages: list[AgentMessage],
        target_tokens: int,
        original_tokens: int,
    ) -> CompactionResult:
        """使用当前模型总结旧轮次；请求失败时退回确定性压缩。"""
        if self._stream_fn is None or self._state.model is None:
            return compact_messages(messages, target_tokens, original_tokens)
        plan = prepare_compaction(messages, target_tokens, original_tokens)
        if not plan.dropped:
            return apply_compaction_summary(plan, "")
        transcript = format_messages_for_summary(plan.dropped)
        prompt = UserMessage(
            content=f"请总结以下早期会话：\n\n{transcript}",
            timestamp=int(time.time() * 1000),
        )
        try:
            stream = await self._stream_fn(
                self._state.model,
                Context(system_prompt=_COMPACTION_SYSTEM_PROMPT, messages=[prompt]),
                StreamOptions(
                    temperature=0,
                    max_tokens=min(4096, max(256, plan.summary_budget)),
                    session_id=self._session_id,
                    abort_event=self._abort_event,
                    thinking_level=ModelThinkingLevel.OFF,
                ),
            )
            final = None
            async for event in stream:
                if event.type == "done":
                    final = event.message
                elif event.type == "error":
                    raise RuntimeError(event.error.error_message or "Compaction request failed")
            if final is None:
                raise RuntimeError("Compaction request returned no final message")
            summary = "\n".join(
                block.text for block in final.content if isinstance(block, TextContent)
            ).strip()
            if not summary:
                raise RuntimeError("Compaction request returned an empty summary")
            return apply_compaction_summary(plan, summary, final.usage)
        except Exception:
            return compact_messages(messages, target_tokens, original_tokens)

    async def _process_event(self, event: AgentEvent) -> None:
        if event.type == "message_start":
            self._state.streaming_message = event.message
        elif event.type == "text_delta":
            # 流式文本增量：更新 streaming_message 的内容
            if self._state.streaming_message is not None:
                if self._state.streaming_message.content:
                    last = self._state.streaming_message.content[-1]
                    if hasattr(last, "text"):
                        last.text += event.delta
                    else:
                        self._state.streaming_message.content.append(TextContent(text=event.delta))
                else:
                    self._state.streaming_message.content.append(TextContent(text=event.delta))
        elif event.type == "thinking_delta":
            if self._state.streaming_message is not None:
                if self._state.streaming_message.content:
                    last = self._state.streaming_message.content[-1]
                    if isinstance(last, ThinkingContent):
                        last.thinking += event.delta
                    else:
                        self._state.streaming_message.content.append(
                            ThinkingContent(thinking=event.delta)
                        )
                else:
                    self._state.streaming_message.content.append(
                        ThinkingContent(thinking=event.delta)
                    )
        elif event.type == "message_end":
            self._state.streaming_message = None
            self._state.messages.append(event.message)
        elif event.type == "tool_execution_start":
            self._state.pending_tool_calls.add(event.tool_call_id)
        elif event.type == "tool_execution_end":
            self._state.pending_tool_calls.discard(event.tool_call_id)
        elif event.type == "turn_end":
            if event.message.error_message:
                self._state.error_message = event.message.error_message
        elif event.type == "agent_end":
            self._state.streaming_message = None

        for listener in self._listeners:
            await listener(event)
