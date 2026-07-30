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
from pi.ai.streaming import (
    DoneEvent,
    ErrorEvent,
    EventStream,
    StartEvent,
    TextDeltaEvent,
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
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)


class OpenAIProvider(BaseProvider):
    """OpenAI chat completions 提供商。"""

    @property
    def provider_id(self) -> str:
        return "openai"

    def build_headers(self, api_key: str, options: StreamOptions | None = None) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
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

    async def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> EventStream:
        stream_obj = EventStream()
        api_key = self.resolve_api_key(options)
        if not api_key:
            await self._emit_error(stream_obj, model, "No API key for provider 'openai'")
            return stream_obj

        base_url = model.base_url or "https://api.openai.com/v1"
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = self.build_headers(api_key, options)

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

        task = asyncio.create_task(_stream_openai(url, headers, payload, model, stream_obj))
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
) -> None:
    """从 OpenAI 流式请求并推送事件的后台任务。"""

    now = int(time.time() * 1000)
    text_parts: list[str] = []
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

    try:
        async with (
            httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client,
            client.stream("POST", url, headers=headers, json=payload) as resp,
        ):
            if resp.status_code != 200:
                body = await resp.aread()
                raise RuntimeError(f"OpenAI API error {resp.status_code}: {body.decode()}")

            async for line in resp.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                choices = chunk.get("choices", [])
                if chunk.get("usage"):
                    u = chunk["usage"]
                    usage.input = u.get("prompt_tokens", 0)
                    usage.output = u.get("completion_tokens", 0)
                    usage.total_tokens = u.get("total_tokens", 0)
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                finish = choices[0].get("finish_reason")

                if "content" in delta and delta["content"]:
                    text_parts.append(delta["content"])
                    await stream_obj.push(
                        TextDeltaEvent(
                            content_index=0,
                            delta=delta["content"],
                        )
                    )

                if "tool_calls" in delta:
                    for tc in delta["tool_calls"]:
                        idx = tc.get("index", 0)
                        if idx not in tool_buffers:
                            tool_buffers[idx] = {"id": "", "name": "", "args": ""}
                        if tc.get("id"):
                            tool_buffers[idx]["id"] = tc["id"]
                        fn = tc.get("function", {})
                        if fn.get("name"):
                            tool_buffers[idx]["name"] = fn["name"]
                        if fn.get("arguments"):
                            tool_buffers[idx]["args"] += fn["arguments"]
                            await stream_obj.push(
                                ToolCallDeltaEvent(
                                    content_index=idx,
                                    delta=fn["arguments"],
                                )
                            )

                if finish:
                    if finish == "tool_calls":
                        stop_reason = StopReason.TOOL_USE
                    elif finish == "length":
                        stop_reason = StopReason.LENGTH

        # 构建最终内容块
        final_blocks: list[Any] = []
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
