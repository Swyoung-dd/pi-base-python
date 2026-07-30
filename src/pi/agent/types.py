"""Agent 层类型。

AgentMessage 扩展 LLM 消息类型，添加 agent 特定元数据。
AgentTool 将 pi-ai Tool 与异步 execute 函数封装在一起。
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from pi.ai.types import (
    ImageContent,
    Model,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    Usage,
)


class QueueMode(StrEnum):
    ONE_AT_A_TIME = "one-at-a-time"
    ALL = "all"


class ToolExecutionMode(StrEnum):
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"


# ---- Agent 消息（扩展 LLM 消息，添加 agent 元数据） ----


@dataclass
class AgentUserMessage:
    role: Literal["user"] = "user"
    content: str | list[TextContent | ImageContent] = ""
    timestamp: int = 0


@dataclass
class AgentAssistantMessage:
    role: Literal["assistant"] = "assistant"
    content: list[TextContent | ThinkingContent | ToolCall] = field(default_factory=list)
    api: str = ""
    provider: str = ""
    model: str = ""
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = "pending"
    error_message: str | None = None
    timestamp: int = 0


@dataclass
class AgentToolResultMessage:
    role: Literal["toolResult"] = "toolResult"
    tool_call_id: str = ""
    tool_name: str = ""
    content: list[TextContent | ImageContent] = field(default_factory=list)
    details: Any = None
    is_error: bool = False
    timestamp: int = 0


AgentMessage = AgentUserMessage | AgentAssistantMessage | AgentToolResultMessage


# ---- Agent 工具 ----

ToolExecuteFn = Callable[
    ["AgentToolCall", Any],
    Coroutine[Any, Any, "AgentToolResult"],
]


@dataclass
class AgentTool:
    """Agent 可调用的工具。将 pi-ai Tool 与 execute 函数封装。"""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    execute: ToolExecuteFn

    def to_ai_tool(self) -> Tool:
        from pi.ai.types import ToolParameterSchema

        if isinstance(self.parameters, dict):
            params = ToolParameterSchema(**self.parameters)
        else:
            params = self.parameters
        return Tool(
            name=self.name,
            description=self.description,
            parameters=params,
        )


@dataclass
class AgentToolCall:
    """来自助手的工具调用。"""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class AgentToolResult:
    """工具执行结果。"""

    tool_call_id: str
    tool_name: str
    content: list[TextContent | ImageContent] = field(default_factory=list)
    details: Any = None
    is_error: bool = False


# ---- Agent 状态 ----


@dataclass
class AgentState:
    system_prompt: str = ""
    model: Model | None = None
    thinking_level: str = "off"
    tools: list[AgentTool] = field(default_factory=list)
    messages: list[AgentMessage] = field(default_factory=list)
    is_streaming: bool = False
    streaming_message: AgentAssistantMessage | None = None
    pending_tool_calls: set[str] = field(default_factory=set)
    error_message: str | None = None


# ---- Agent 上下文（传递给循环的快照） ----


@dataclass
class AgentContext:
    system_prompt: str
    messages: list[AgentMessage]
    tools: list[AgentTool]


# ---- Agent 事件 ----


@dataclass
class MessageStartEvent:
    type: Literal["message_start"] = "message_start"
    message: AgentAssistantMessage = field(default_factory=AgentAssistantMessage)


@dataclass
class MessageUpdateEvent:
    type: Literal["message_update"] = "message_update"
    message: AgentAssistantMessage = field(default_factory=AgentAssistantMessage)


@dataclass
class MessageEndEvent:
    type: Literal["message_end"] = "message_end"
    message: AgentAssistantMessage = field(default_factory=AgentAssistantMessage)


@dataclass
class TextDeltaUpdateEvent:
    """流式文本增量事件，从 provider 透传给 UI 层实时渲染。"""

    type: Literal["text_delta"] = "text_delta"
    delta: str = ""
    content_index: int = 0


@dataclass
class ThinkingDeltaUpdateEvent:
    """流式思考增量事件。"""

    type: Literal["thinking_delta"] = "thinking_delta"
    delta: str = ""
    content_index: int = 0


@dataclass
class ToolExecutionStartEvent:
    type: Literal["tool_execution_start"] = "tool_execution_start"
    tool_call_id: str = ""
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolExecutionEndEvent:
    type: Literal["tool_execution_end"] = "tool_execution_end"
    tool_call_id: str = ""
    tool_name: str = ""
    result: AgentToolResult | None = None


@dataclass
class TurnEndEvent:
    type: Literal["turn_end"] = "turn_end"
    message: AgentAssistantMessage = field(default_factory=AgentAssistantMessage)
    tool_results: list[AgentToolResult] = field(default_factory=list)


@dataclass
class AgentEndEvent:
    type: Literal["agent_end"] = "agent_end"
    messages: list[AgentMessage] = field(default_factory=list)


@dataclass
class ContextCompactedEvent:
    """发送给模型前发生上下文压缩。"""

    type: Literal["context_compacted"] = "context_compacted"
    original_tokens: int = 0
    compacted_tokens: int = 0
    dropped_messages: int = 0


@dataclass
class ProviderRetryEvent:
    """提供商请求将在退避后重试。"""

    type: Literal["provider_retry"] = "provider_retry"
    attempt: int = 0
    max_retries: int = 0
    delay_ms: int = 0
    error: str = ""


AgentEvent = (
    MessageStartEvent
    | MessageUpdateEvent
    | MessageEndEvent
    | TextDeltaUpdateEvent
    | ThinkingDeltaUpdateEvent
    | ToolExecutionStartEvent
    | ToolExecutionEndEvent
    | TurnEndEvent
    | AgentEndEvent
    | ContextCompactedEvent
    | ProviderRetryEvent
)


AgentEventSink = Callable[[AgentEvent], Coroutine[Any, Any, None]]


# ---- 辅助函数 ----


def create_user_message(text: str, images: list[ImageContent] | None = None) -> AgentUserMessage:
    import time

    if images:
        content: list[TextContent | ImageContent] = [TextContent(text=text)]
        content.extend(images)
        return AgentUserMessage(content=content, timestamp=int(time.time() * 1000))
    return AgentUserMessage(content=text, timestamp=int(time.time() * 1000))
