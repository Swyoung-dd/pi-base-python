"""Anthropic 提供商 — 使用 Messages API 流式请求。

将 pi 的统一消息格式转换为 Anthropic API 格式，
以 AssistantMessageEvents 流式返回文本和工具调用增量。
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx

from pi.ai.providers.base import BaseProvider
from pi.ai.providers.retry import raise_for_status, run_with_retries
from pi.ai.streaming import (
    DoneEvent,
    ErrorEvent,
    EventStream,
    StartEvent,
    TextDeltaEvent,
    ThinkingDeltaEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
)
from pi.ai.types import (
    AssistantMessage,
    Context,
    Model,
    StopReason,
    StreamOptions,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)


class AnthropicProvider(BaseProvider):
    """Anthropic Messages API 提供商。"""

    @property
    def provider_id(self) -> str:
        return "anthropic"

    def build_headers(self, api_key: str, options: StreamOptions | None = None) -> dict[str, str]:
        return {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    def convert_messages(self, context: Context) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for msg in context.messages:
            if isinstance(msg, UserMessage):
                if isinstance(msg.content, str):
                    messages.append({"role": "user", "content": msg.content})
                else:
                    blocks = []
                    for block in msg.content:
                        if hasattr(block, "text"):
                            blocks.append({"type": "text", "text": block.text})
                        elif hasattr(block, "data"):
                            blocks.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": block.mime_type,
                                        "data": block.data,
                                    },
                                }
                            )
                    messages.append({"role": "user", "content": blocks})
            elif isinstance(msg, AssistantMessage):
                blocks: list[dict[str, Any]] = []
                for block in msg.content:
                    if isinstance(block, ThinkingContent) and block.redacted:
                        blocks.append({"type": "redacted_thinking", "data": block.thinking})
                    elif isinstance(block, ThinkingContent) and block.thinking_signature:
                        blocks.append(
                            {
                                "type": "thinking",
                                "thinking": block.thinking,
                                "signature": block.thinking_signature,
                            }
                        )
                    elif isinstance(block, TextContent):
                        blocks.append({"type": "text", "text": block.text})
                    elif isinstance(block, ToolCall):
                        blocks.append(
                            {
                                "type": "tool_use",
                                "id": block.id,
                                "name": block.name,
                                "input": block.arguments,
                            }
                        )
                messages.append({"role": "assistant", "content": blocks})
            elif isinstance(msg, ToolResultMessage):
                text = "".join(b.text for b in msg.content if hasattr(b, "text"))
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id,
                                "content": text,
                                "is_error": msg.is_error,
                            }
                        ],
                    }
                )
        return messages

    def convert_tools(self, context: Context) -> list[dict[str, Any]] | None:
        if not context.tools:
            return None
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters.model_dump(),
            }
            for tool in context.tools
        ]

    async def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> EventStream:
        stream_obj = EventStream()
        api_key = await self.resolve_api_key(options)
        if not api_key:
            await self._emit_error(stream_obj, model, "No API key for provider 'anthropic'")
            return stream_obj

        base_url = model.base_url or "https://api.anthropic.com"
        url = f"{base_url.rstrip('/')}/v1/messages"
        headers = self.merge_headers(self.build_headers(api_key, options), model, options)

        payload: dict[str, Any] = {
            "model": model.id,
            "messages": self.convert_messages(context),
            "stream": True,
            "max_tokens": options.max_tokens if options and options.max_tokens else 4096,
        }
        if context.system_prompt:
            payload["system"] = context.system_prompt
        tools = self.convert_tools(context)
        if tools:
            payload["tools"] = tools
        if options and options.temperature is not None and options.thinking_level.value == "off":
            payload["temperature"] = options.temperature
        if options and model.reasoning and options.thinking_level.value != "off":
            budget_map = {
                "minimal": 1024,
                "low": 2048,
                "medium": 4096,
                "high": 8192,
                "xhigh": 16384,
                "max": 32768,
            }
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": options.thinking_budget_tokens
                or budget_map[options.thinking_level.value],
            }

        max_retries = options.max_retries if options and options.max_retries is not None else 2
        timeout = options.timeout_ms / 1000 if options and options.timeout_ms else 600.0
        task = asyncio.create_task(
            _stream_anthropic(
                url,
                headers,
                payload,
                model,
                stream_obj,
                max_retries,
                timeout,
            )
        )
        stream_obj.set_producer_task(task)
        return stream_obj

    async def _emit_error(self, stream_obj: EventStream, model: Model, msg: str) -> None:
        now = int(time.time() * 1000)
        error_msg = AssistantMessage(
            content=[TextContent(text="")],
            api=model.api,
            provider=model.provider,
            model=model.id,
            stop_reason=StopReason.ERROR,
            error_message=msg,
            timestamp=now,
        )
        await stream_obj.push(ErrorEvent(reason="error", error=error_msg))
        await stream_obj.end(error_msg)


async def _stream_anthropic(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    model: Model,
    stream_obj: EventStream,
    max_retries: int,
    timeout: float,
) -> None:
    now = int(time.time() * 1000)
    content_blocks: dict[int, dict[str, Any]] = {}
    usage = Usage()
    stop_reason = StopReason.STOP

    partial = AssistantMessage(
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        timestamp=now,
    )
    await stream_obj.push(StartEvent(partial=partial))

    async def request_once() -> None:
        nonlocal stop_reason
        async with (
            httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client,
            client.stream("POST", url, headers=headers, json=payload) as response,
        ):
            await raise_for_status(response, "Anthropic")
            async for line in response.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                data = json.loads(line[6:])
                event_type = data.get("type", "")

                if event_type == "content_block_start":
                    index = data.get("index", len(content_blocks))
                    block = data.get("content_block", {})
                    block_type = block.get("type")
                    if block_type == "tool_use":
                        content_blocks[index] = {
                            "type": "tool_use",
                            "id": block.get("id", ""),
                            "name": block.get("name", ""),
                            "args": "",
                        }
                    elif block_type == "thinking":
                        content_blocks[index] = {
                            "type": "thinking",
                            "thinking": block.get("thinking", ""),
                            "signature": block.get("signature", ""),
                        }
                    elif block_type == "redacted_thinking":
                        content_blocks[index] = {
                            "type": "redacted_thinking",
                            "data": block.get("data", ""),
                        }
                    elif block_type == "text":
                        content_blocks[index] = {
                            "type": "text",
                            "text": block.get("text", ""),
                        }
                elif event_type == "content_block_delta":
                    index = data.get("index", 0)
                    delta = data.get("delta", {})
                    if delta.get("type") == "thinking_delta":
                        thinking = delta.get("thinking", "")
                        block = content_blocks.setdefault(
                            index,
                            {"type": "thinking", "thinking": "", "signature": ""},
                        )
                        block["thinking"] += thinking
                        await stream_obj.push(
                            ThinkingDeltaEvent(content_index=index, delta=thinking)
                        )
                    elif delta.get("type") == "signature_delta":
                        block = content_blocks.setdefault(
                            index,
                            {"type": "thinking", "thinking": "", "signature": ""},
                        )
                        block["signature"] += delta.get("signature", "")
                    elif delta.get("type") == "text_delta":
                        block = content_blocks.setdefault(index, {"type": "text", "text": ""})
                        block["text"] += delta.get("text", "")
                        await stream_obj.push(
                            TextDeltaEvent(content_index=index, delta=delta.get("text", ""))
                        )
                    elif delta.get("type") == "input_json_delta":
                        block = content_blocks.get(index)
                        if block is None or block.get("type") != "tool_use":
                            raise RuntimeError("Anthropic sent tool arguments before tool_use")
                        block["args"] += delta.get("partial_json", "")
                        await stream_obj.push(
                            ToolCallDeltaEvent(
                                content_index=index,
                                delta=delta.get("partial_json", ""),
                            )
                        )
                elif event_type == "message_delta":
                    delta = data.get("delta", {})
                    if delta.get("stop_reason") == "tool_use":
                        stop_reason = StopReason.TOOL_USE
                    elif delta.get("stop_reason") == "max_tokens":
                        stop_reason = StopReason.LENGTH
                    response_usage = data.get("usage", {})
                    usage.output = response_usage.get("output_tokens", usage.output)
                elif event_type == "message_start":
                    response_usage = data.get("message", {}).get("usage", {})
                    usage.input = response_usage.get("input_tokens", 0)
                    usage.cache_read = response_usage.get("cache_read_input_tokens", 0)
                    usage.cache_write = response_usage.get("cache_creation_input_tokens", 0)
                elif event_type == "error":
                    error = data.get("error", {})
                    raise RuntimeError(error.get("message") or "Anthropic stream error")

    try:
        await run_with_retries(
            request_once,
            lambda: bool(content_blocks),
            stream_obj,
            max_retries,
        )

        final_blocks: list[Any] = []
        for index in sorted(content_blocks):
            block = content_blocks[index]
            if block["type"] == "thinking":
                final_blocks.append(
                    ThinkingContent(
                        thinking=block["thinking"],
                        thinking_signature=block["signature"] or None,
                    )
                )
            elif block["type"] == "redacted_thinking":
                final_blocks.append(ThinkingContent(thinking=block["data"], redacted=True))
            elif block["type"] == "text":
                final_blocks.append(TextContent(text=block["text"]))
            elif block["type"] == "tool_use":
                try:
                    args = json.loads(block["args"]) if block["args"] else {}
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Anthropic returned invalid arguments for tool {block['name']}"
                    ) from exc
                call = ToolCall(id=block["id"], name=block["name"], arguments=args)
                final_blocks.append(call)
                await stream_obj.push(ToolCallEndEvent(content_index=index, tool_call=call))

        if not final_blocks:
            final_blocks = [TextContent(text="")]

        usage.total_tokens = usage.input + usage.output + usage.cache_read + usage.cache_write
        final_msg = AssistantMessage(
            content=final_blocks,
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=usage,
            stop_reason=stop_reason,
            timestamp=int(time.time() * 1000),
        )
        reason = (
            "toolUse"
            if stop_reason == StopReason.TOOL_USE
            else ("length" if stop_reason == StopReason.LENGTH else "stop")
        )
        await stream_obj.push(DoneEvent(reason=reason, message=final_msg))
        await stream_obj.end(final_msg)

    except Exception as exc:
        error_msg = AssistantMessage(
            content=[TextContent(text="")],
            api=model.api,
            provider=model.provider,
            model=model.id,
            stop_reason=StopReason.ERROR,
            error_message=str(exc),
            timestamp=int(time.time() * 1000),
        )
        await stream_obj.push(ErrorEvent(reason="error", error=error_msg))
        await stream_obj.end(error_msg)
