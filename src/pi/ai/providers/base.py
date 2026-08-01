"""提供商基类接口。

每个提供商实现 stream() 方法，返回 AssistantMessageEvent 的 EventStream。
提供商负责：
- 将 pi 的统一 Context/messages 转换为各自 API 格式
- 以事件流形式返回增量结果
- 返回带 usage 和 stop_reason 的最终 AssistantMessage
"""

from __future__ import annotations

import abc
from typing import Any

from pi.ai.streaming import EventStream
from pi.ai.types import Context, Model, ModelThinkingLevel, StreamOptions


class BaseProvider(abc.ABC):
    """LLM 提供商抽象基类。"""

    @property
    def requires_api_key(self) -> bool:
        """该 provider 是否需要凭据。"""
        return True

    @property
    @abc.abstractmethod
    def provider_id(self) -> str:
        """提供商标识符，如 'openai'、'anthropic'。"""
        ...

    @abc.abstractmethod
    async def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> EventStream:
        """流式请求提供商补全。

        返回发射 AssistantMessageEvent 对象的 EventStream。
        流必须以 DoneEvent 或 ErrorEvent 终止。
        """
        ...

    async def resolve_api_key(self, options: StreamOptions | None = None) -> str | None:
        """从选项或环境变量解析 API 密钥。"""
        if options and options.api_key:
            return options.api_key
        from pi.ai.auth import get_provider_token

        return await get_provider_token(self.provider_id)

    def build_headers(self, api_key: str, options: StreamOptions | None = None) -> dict[str, str]:
        """构建 API 请求的 HTTP 头。由各提供商覆盖。"""
        return {}

    def merge_headers(
        self,
        base_headers: dict[str, str],
        model: Model,
        options: StreamOptions | None = None,
    ) -> dict[str, str]:
        """按模型和调用选项覆盖 headers；值为 None 时删除对应项。"""
        headers = dict(base_headers)
        if model.headers:
            headers.update(model.headers)
        if options and options.headers:
            for key, value in options.headers.items():
                if value is None:
                    headers.pop(key, None)
                else:
                    headers[key] = value
        return headers

    def convert_messages(self, context: Context) -> list[dict[str, Any]]:
        """将 pi 消息转换为提供商特定格式。由各提供商覆盖。"""
        raise NotImplementedError

    def convert_tools(self, context: Context) -> list[dict[str, Any]] | None:
        """将 pi 工具转换为提供商特定格式。由各提供商覆盖。"""
        if not context.tools:
            return None
        raise NotImplementedError

    def map_thinking_level(self, level: ModelThinkingLevel) -> Any:
        """Map unified thinking level to provider-specific format.

        Override per provider to return the format the provider expects
        (e.g., a string for OpenAI reasoning_effort, or budget tokens for Anthropic).
        """
        return level
