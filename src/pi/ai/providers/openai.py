"""OpenAI 提供商 — 使用 chat completions API 流式请求。

将 pi 的统一消息格式转换为 OpenAI API 格式，
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


class OpenAIProvider(BaseProvider):
    """OpenAI chat completions 提供商。"""

    def __init__(
        self,
        provider_id: str = "openai",
        requires_api_key: bool = True,
    ) -> None:
        self._provider_id = provider_id
        self._requires_api_key = requires_api_key

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def requires_api_key(self) -> bool:
        return self._requires_api_key

    def build_headers(self, api_key: str, options: StreamOptions | None = None) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def convert_messages(self, context: Context) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if context.system_prompt:
            messages.append({"role": "system", "content": context.system_prompt})
        for msg in context.messages:
            if isinstance(msg, UserMessage):
                if isinstance(msg.content, str):
                    messages.append({"role": "user", "content": msg.content})
                else:
                    parts = []
                    for block in msg.content:
                        if hasattr(block, "text"):
                            parts.append({"type": "text", "text": block.text})
                        elif hasattr(block, "data"):
                            parts.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{block.mime_type};base64,{block.data}"
                                    },
                                }
                            )
                    messages.append({"role": "user", "content": parts})
            elif isinstance(msg, AssistantMessage):
                # 重建带工具调用的助手消息
                content_parts = []
                tool_calls = []
                for block in msg.content:
                    if isinstance(block, TextContent):
                        content_parts.append(block.text)
                    elif isinstance(block, ToolCall):
                        tool_calls.append(
                            {
                                "id": block.id,
                                "type": "function",
                                "function": {
                                    "name": block.name,
                                    "arguments": json.dumps(block.arguments),
                                },
                            }
                        )
                entry: dict[str, Any] = {"role": "assistant"}
                if content_parts:
                    entry["content"] = "".join(content_parts)
                else:
                    entry["content"] = None
                if tool_calls:
                    entry["tool_calls"] = tool_calls
                messages.append(entry)
            elif isinstance(msg, ToolResultMessage):
                text = "".join(b.text for b in msg.content if hasattr(b, "text"))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": text,
                    }
                )
        return messages

    def convert_tools(self, context: Context) -> list[dict[str, Any]] | None:
        if not context.tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters.model_dump(),
                },
            }
            for tool in context.tools
        ]

    def build_payload(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> dict[str, Any]:
        """构建 OpenAI Chat Completions 请求体。"""
        payload: dict[str, Any] = {
            "model": model.id,
            "messages": self.convert_messages(context),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        tools = self.convert_tools(context)
        if tools:
            payload["tools"] = tools
        if options:
            if options.temperature is not None:
                payload["temperature"] = options.temperature
            if options.max_tokens is not None:
                payload["max_tokens"] = options.max_tokens
            if model.reasoning and options.thinking_level.value != "off":
                effort_map = {
                    "minimal": "low",
                    "low": "low",
                    "medium": "medium",
                    "high": "high",
                    "xhigh": "high",
                    "max": "high",
                }
                payload["reasoning_effort"] = effort_map[options.thinking_level.value]
        return payload

    async def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> EventStream:
        stream_obj = EventStream()
        api_key = await self.resolve_api_key(options)
        if not api_key and self._requires_api_key:
            await self._emit_error(
                stream_obj,
                model,
                f"No API key for provider '{self.provider_id}'",
            )
            return stream_obj

        base_url = model.base_url or "https://api.openai.com/v1"
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = self.merge_headers(
            self.build_headers(api_key or "", options),
            model,
            options,
        )

        payload = self.build_payload(model, context, options)

        max_retries = options.max_retries if options and options.max_retries is not None else 2
        timeout = options.timeout_ms / 1000 if options and options.timeout_ms else 600.0
        task = asyncio.create_task(
            _stream_openai(
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


async def _stream_openai(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    model: Model,
    stream_obj: EventStream,
    max_retries: int,
    timeout: float,
) -> None:
    """从 OpenAI 流式请求并推送事件的后台任务。"""

    now = int(time.time() * 1000)
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_buffers: dict[int, dict[str, str]] = {}
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
            await raise_for_status(response, "OpenAI")
            async for line in response.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                choices = chunk.get("choices", [])
                if chunk.get("usage"):
                    response_usage = chunk["usage"]
                    usage.input = response_usage.get("prompt_tokens", 0)
                    usage.output = response_usage.get("completion_tokens", 0)
                    usage.total_tokens = response_usage.get("total_tokens", 0)
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                finish = choices[0].get("finish_reason")

                reasoning_delta = delta.get("reasoning_content") or delta.get("reasoning")
                if reasoning_delta:
                    thinking_parts.append(reasoning_delta)
                    await stream_obj.push(
                        ThinkingDeltaEvent(content_index=0, delta=reasoning_delta)
                    )

                if delta.get("content"):
                    text_parts.append(delta["content"])
                    await stream_obj.push(TextDeltaEvent(content_index=0, delta=delta["content"]))

                for tool_delta in delta.get("tool_calls", []):
                    index = tool_delta.get("index", 0)
                    if index not in tool_buffers:
                        tool_buffers[index] = {"id": "", "name": "", "args": ""}
                    if tool_delta.get("id"):
                        tool_buffers[index]["id"] = tool_delta["id"]
                    function = tool_delta.get("function", {})
                    if function.get("name"):
                        tool_buffers[index]["name"] = function["name"]
                    if function.get("arguments"):
                        tool_buffers[index]["args"] += function["arguments"]
                        await stream_obj.push(
                            ToolCallDeltaEvent(
                                content_index=index,
                                delta=function["arguments"],
                            )
                        )

                if finish == "tool_calls":
                    stop_reason = StopReason.TOOL_USE
                elif finish == "length":
                    stop_reason = StopReason.LENGTH

    try:
        await run_with_retries(
            request_once,
            lambda: bool(thinking_parts or text_parts or tool_buffers),
            stream_obj,
            max_retries,
        )

        # 构建最终内容块
        final_blocks: list[Any] = []
        if thinking_parts:
            final_blocks.append(ThinkingContent(thinking="".join(thinking_parts)))
        if text_parts:
            final_blocks.append(TextContent(text="".join(text_parts)))
        for idx in sorted(tool_buffers.keys()):
            buf = tool_buffers[idx]
            try:
                args = json.loads(buf["args"]) if buf["args"] else {}
            except json.JSONDecodeError:
                args = {}
            tc = ToolCall(id=buf["id"], name=buf["name"], arguments=args)
            final_blocks.append(tc)
            await stream_obj.push(ToolCallEndEvent(content_index=idx, tool_call=tc))

        if not final_blocks:
            final_blocks = [TextContent(text="")]

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
