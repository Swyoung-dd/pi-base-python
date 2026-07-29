"""LLM 响应的流式事件协议与 EventStream。

镜像 pi-ai 的 AssistantMessageEventStream：提供商发射结构化事件
（文本增量、工具调用增量、思考增量），消费者据此响应。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal

from pi.ai.types import AssistantMessage, ToolCall


@dataclass
class StartEvent:
    type: Literal["start"] = "start"
    partial: AssistantMessage = field(default_factory=AssistantMessage)


@dataclass
class TextDeltaEvent:
    type: Literal["text_delta"] = "text_delta"
    content_index: int = 0
    delta: str = ""
    partial: AssistantMessage = field(default_factory=AssistantMessage)


@dataclass
class ThinkingDeltaEvent:
    type: Literal["thinking_delta"] = "thinking_delta"
    content_index: int = 0
    delta: str = ""
    partial: AssistantMessage = field(default_factory=AssistantMessage)


@dataclass
class ToolCallDeltaEvent:
    type: Literal["toolcall_delta"] = "toolcall_delta"
    content_index: int = 0
    delta: str = ""
    partial: AssistantMessage = field(default_factory=AssistantMessage)


@dataclass
class ToolCallEndEvent:
    type: Literal["toolcall_end"] = "toolcall_end"
    content_index: int = 0
    tool_call: ToolCall = field(default_factory=ToolCall)
    partial: AssistantMessage = field(default_factory=AssistantMessage)


@dataclass
class DoneEvent:
    type: Literal["done"] = "done"
    reason: str = "stop"  # "stop" | "length" | "toolUse"
    message: AssistantMessage = field(default_factory=AssistantMessage)


@dataclass
class ErrorEvent:
    type: Literal["error"] = "error"
    reason: str = "error"  # "error" | "aborted"
    error: AssistantMessage = field(default_factory=AssistantMessage)


AssistantMessageEvent = (
    StartEvent
    | TextDeltaEvent
    | ThinkingDeltaEvent
    | ToolCallDeltaEvent
    | ToolCallEndEvent
    | DoneEvent
    | ErrorEvent
)


class EventStream:
    """LLM 响应的异步事件流。

    生产者通过 push() 和 end() / error() 推送事件。
    消费者异步迭代：async for event in stream。
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[AssistantMessageEvent | None] = asyncio.Queue()
        self._result: AssistantMessage | None = None
        self._closed = False

    async def push(self, event: AssistantMessageEvent) -> None:
        if self._closed:
            return
        await self._queue.put(event)

    async def end(self, result: AssistantMessage) -> None:
        self._result = result
        self._closed = True
        await self._queue.put(None)

    def __aiter__(self) -> AsyncIterator[AssistantMessageEvent]:
        return self._aiter()

    async def _aiter(self) -> AsyncIterator[AssistantMessageEvent]:
        while True:
            event = await self._queue.get()
            if event is None:
                break
            yield event

    @property
    def result(self) -> AssistantMessage | None:
        return self._result

    @property
    def closed(self) -> bool:
        return self._closed
