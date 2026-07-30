"""Stateful Agent wrapper around the low-level agent loop.

Owns the current transcript, emits lifecycle events, executes tools,
and exposes queueing APIs for steering and follow-up messages.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

from pi.agent.agent_loop import run_agent_loop
from pi.agent.session.base import SessionStorage
from pi.agent.types import (
    AgentAssistantMessage,
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentMessage,
    AgentState,
    AgentTool,
    MessageEndEvent,
    MessageStartEvent,
    QueueMode,
    ToolExecutionMode,
    TurnEndEvent,
    create_user_message,
)
from pi.ai.streaming import EventStream
from pi.ai.types import (
    ImageContent,
    Model,
    ModelThinkingLevel,
    StreamOptions,
    TextContent,
    ThinkingContent,
)

StreamFn = Callable[
    [Model, Any, StreamOptions | None],
    Coroutine[Any, Any, EventStream],
]

AgentListener = Callable[[AgentEvent], Coroutine[Any, Any, None]]


@dataclass
class AgentOptions:
    """Options for constructing an Agent."""

    model: Model | None = None
    system_prompt: str = ""
    tools: list[AgentTool] = field(default_factory=list)
    stream_fn: StreamFn | None = None
    session_id: str | None = None
    session_storage: SessionStorage | None = None
    steering_mode: QueueMode = QueueMode.ONE_AT_A_TIME
    follow_up_mode: QueueMode = QueueMode.ONE_AT_A_TIME
    tool_execution: ToolExecutionMode = ToolExecutionMode.PARALLEL
    context_token_limit: int | None = None
    compact_to_tokens: int | None = None
    thinking_level: ModelThinkingLevel = ModelThinkingLevel.OFF
    thinking_budget_tokens: int | None = None


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
    """Stateful wrapper around the agent loop.

    Usage:
        agent = Agent(AgentOptions(model=..., stream_fn=..., tools=...))
        agent.subscribe(my_listener)
        await agent.prompt("Hello")
    """

    def __init__(self, options: AgentOptions) -> None:
        self._state = AgentState(
            system_prompt=options.system_prompt,
            model=options.model,
            tools=options.tools[:],
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
        model_context_limit = None
        if options.model is not None and options.model.context_window:
            model_context_limit = max(
                1,
                options.model.context_window - (options.model.max_tokens or 4096),
            )
        self._context_token_limit = options.context_token_limit or model_context_limit
        self._compact_to_tokens = options.compact_to_tokens
        self._thinking_level = options.thinking_level
        self._thinking_budget_tokens = options.thinking_budget_tokens
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
            branch = await self._session_storage.get_branch()
            self._state.messages = [
                entry.message
                for entry in branch
                if entry.type == "message" and entry.message is not None
            ]
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
        """在空闲状态切换后续请求使用的模型。"""
        if self.is_busy:
            raise RuntimeError("Agent is already processing")
        self._state.model = model
        self._context_token_limit = (
            max(1, model.context_window - (model.max_tokens or 4096))
            if model.context_window
            else None
        )

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
        previous_messages = self._state.messages[:]
        self._state.messages.extend(messages)

        try:
            new_messages = await run_agent_loop(
                prompts=messages,
                context=AgentContext(
                    system_prompt=self._state.system_prompt,
                    messages=previous_messages,
                    tools=self._state.tools[:],
                ),
                model=self._state.model,
                stream_fn=self._stream_fn,
                sink=self._process_event,
                options=StreamOptions(
                    session_id=self._session_id,
                    abort_event=self._abort_event,
                    context_token_limit=self._context_token_limit,
                    compact_to_tokens=self._compact_to_tokens,
                    thinking_level=self._thinking_level,
                    thinking_budget_tokens=self._thinking_budget_tokens,
                ),
                tool_execution=self._tool_execution,
            )
            self._state.messages = previous_messages + new_messages
            if self._session_storage is not None:
                for message in new_messages:
                    await self._session_storage.append_message(message)
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
            self._state.streaming_message = None
            self._state.pending_tool_calls.clear()
            self._idle_event.set()

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
