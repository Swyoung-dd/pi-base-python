"""DeepSeek V4 的 OpenAI 兼容 provider。"""

from __future__ import annotations

from typing import Any

from pi.ai.providers.openai import OpenAIProvider
from pi.ai.types import (
    AssistantMessage,
    Context,
    Model,
    ModelThinkingLevel,
    StreamOptions,
    ThinkingContent,
)


class DeepSeekProvider(OpenAIProvider):
    """处理 DeepSeek V4 思考模式及工具调用上下文。"""

    def __init__(self) -> None:
        super().__init__("deepseek")

    def convert_messages(self, context: Context) -> list[dict[str, Any]]:
        messages = super().convert_messages(context)
        output_index = 1 if context.system_prompt else 0
        for source in context.messages:
            if isinstance(source, AssistantMessage):
                reasoning = "".join(
                    block.thinking
                    for block in source.content
                    if isinstance(block, ThinkingContent)
                )
                if reasoning:
                    messages[output_index]["reasoning_content"] = reasoning
            output_index += 1
        return messages

    def build_payload(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> dict[str, Any]:
        payload = super().build_payload(model, context, options)
        level = options.thinking_level if options else ModelThinkingLevel.OFF
        thinking_enabled = model.reasoning and level != ModelThinkingLevel.OFF
        payload["thinking"] = {"type": "enabled" if thinking_enabled else "disabled"}
        payload.pop("reasoning_effort", None)
        if thinking_enabled:
            payload["reasoning_effort"] = (
                "max" if level in (ModelThinkingLevel.XHIGH, ModelThinkingLevel.MAX) else "high"
            )
        return payload
