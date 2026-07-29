"""统一 LLM API 的核心类型。

镜像 @earendil-works/pi-ai 的类型系统：消息、内容块、
工具、模型、用量和流式事件。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class StopReason(StrEnum):
    PENDING = "pending"
    STOP = "stop"
    LENGTH = "length"
    TOOL_USE = "toolUse"
    ERROR = "error"
    ABORTED = "aborted"


class ThinkingLevel(StrEnum):
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


class ModelThinkingLevel(StrEnum):
    OFF = "off"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


class Transport(StrEnum):
    SSE = "sse"
    WEBSOCKET = "websocket"
    WEBSOCKET_CACHED = "websocket-cached"
    AUTO = "auto"


# ---- 内容块 ----

class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str
    text_signature: str | None = None


class ThinkingContent(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str
    thinking_signature: str | None = None
    redacted: bool | None = None


class ImageContent(BaseModel):
    type: Literal["image"] = "image"
    data: str  # base64 编码
    mime_type: str


# ---- 工具定义 ----

class ToolParameterSchema(BaseModel):
    """工具参数的 JSON Schema。"""
    type: str = "object"
    properties: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)
    additionalProperties: bool = False


class Tool(BaseModel):
    name: str
    description: str
    parameters: ToolParameterSchema = Field(default_factory=ToolParameterSchema)


class ToolCall(BaseModel):
    type: Literal["toolCall"] = "toolCall"
    id: str
    name: str
    arguments: dict[str, Any]
    thought_signature: str | None = None


# ---- 用量 ----

class UsageCost(BaseModel):
    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    total: float = 0.0


class Usage(BaseModel):
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    reasoning: int | None = None
    total_tokens: int = 0
    cost: UsageCost = Field(default_factory=UsageCost)


# ---- 消息 ----

class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str | list[TextContent | ImageContent]
    timestamp: int  # Unix 毫秒


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: list[TextContent | ThinkingContent | ToolCall]
    api: str
    provider: str
    model: str
    response_model: str | None = None
    response_id: str | None = None
    usage: Usage = Field(default_factory=Usage)
    stop_reason: StopReason = StopReason.PENDING
    error_message: str | None = None
    timestamp: int  # Unix 毫秒


class ToolResultMessage(BaseModel):
    role: Literal["toolResult"] = "toolResult"
    tool_call_id: str
    tool_name: str
    content: list[TextContent | ImageContent]
    details: Any = None
    usage: Usage | None = None
    added_tool_names: list[str] | None = None
    is_error: bool = False
    timestamp: int  # Unix 毫秒


Message = UserMessage | AssistantMessage | ToolResultMessage


# ---- 模型 ----

class ModelCost(BaseModel):
    input: float = 0.0  # 美元/百万 token
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0


class Model(BaseModel):
    id: str
    name: str
    api: str  # 如 "openai-responses"、"anthropic-messages"
    provider: str
    base_url: str = ""
    reasoning: bool = False
    input: list[str] = Field(default_factory=lambda: ["text"])
    cost: ModelCost = Field(default_factory=ModelCost)
    context_window: int = 0
    max_tokens: int = 0
    headers: dict[str, str] | None = None


# ---- 上下文（传递给提供商） ----

class Context(BaseModel):
    system_prompt: str | None = None
    messages: list[Message] = Field(default_factory=list)
    tools: list[Tool] | None = None


# ---- 流式选项 ----

class StreamOptions(BaseModel):
    temperature: float | None = None
    max_tokens: int | None = None
    api_key: str | None = None
    session_id: str | None = None
    timeout_ms: int | None = None
    max_retries: int | None = None
    headers: dict[str, str | None] | None = None
    model_config = {"arbitrary_types_allowed": True}
