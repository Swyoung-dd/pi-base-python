"""Agent 层类型。

AgentMessage 扩展 LLM 消息类型，添加 agent 特定元数据。
AgentTool 将 pi-ai Tool 与异步 execute 函数封装在一起。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Literal, Union

from pi.ai.types import (
    AssistantMessage,
    ImageContent,
    Model,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)


class QueueMode(str, Enum):
    ONE_AT_A_TIME = "one-at-a-time"
    ALL = "all"


class ToolExecutionMode(str, Enum):
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


AgentMessage = Union[AgentUserMessage, AgentAssistantMessage, AgentToolResultMessage]


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
        return Tool(
            name=self.name,
            description=self.description,
            parameters=ToolParameterSchema(**self.parameters) if isinstance(self.parameters, dict) else self.parameters,
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
class ToolExecutionStartEvent:
    type: Literal["tool_execution_start"] = "tool_execution_start"
    tool_call_id: str = ""
    tool_name: str = ""


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


AgentEvent = Union[
    MessageStartEvent,
    MessageUpdateEvent,
    MessageEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionEndEvent,
    TurnEndEvent,
    AgentEndEvent,
]


AgentEventSink = Callable[[AgentEvent], Coroutine[Any, Any, None]]


# ---- 辅助函数 ----

def create_user_message(text: str, images: list[ImageContent] | None = None) -> AgentUserMessage:
    import time
    if images:
        content: list[TextContent | ImageContent] = [TextContent(text=text)]
        content.extend(images)
        return AgentUserMessage(content=content, timestamp=int(time.time() * 1000))
    return AgentUserMessage(content=text, timestamp=int(time.time() * 1000))
