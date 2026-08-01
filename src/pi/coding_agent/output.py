"""打印模式的文本、JSON 和 JSONL 渲染。"""

from __future__ import annotations

import json

import click
from pydantic import TypeAdapter

from pi.agent.types import AgentEvent, AgentMessage

_EVENT_ADAPTER = TypeAdapter(AgentEvent)
_MESSAGES_ADAPTER = TypeAdapter(list[AgentMessage])


def event_to_dict(event: AgentEvent) -> dict:
    return _EVENT_ADAPTER.dump_python(event, mode="json")


def _result_to_dict(messages: list[AgentMessage]) -> dict:
    usage = {
        "input": 0,
        "output": 0,
        "cache_read": 0,
        "cache_write": 0,
        "reasoning": 0,
        "total_tokens": 0,
    }
    error = None
    for message in messages:
        if message.role != "assistant":
            continue
        usage["input"] += message.usage.input
        usage["output"] += message.usage.output
        usage["cache_read"] += message.usage.cache_read
        usage["cache_write"] += message.usage.cache_write
        usage["reasoning"] += message.usage.reasoning or 0
        usage["total_tokens"] += message.usage.total_tokens
        error = message.error_message or error
    return {
        "type": "agent_result",
        "messages": _MESSAGES_ADAPTER.dump_python(messages, mode="json"),
        "usage": usage,
        "error": error,
    }


class PrintRenderer:
    """可直接作为 Agent 异步事件监听器使用。"""

    def __init__(self, output_format: str = "text") -> None:
        self.output_format = output_format
        self._thinking_started = False

    async def __call__(self, event: AgentEvent) -> None:
        if self.output_format == "jsonl":
            click.echo(json.dumps(event_to_dict(event), ensure_ascii=False, default=str))
            return
        if self.output_format == "json":
            if event.type == "agent_end":
                click.echo(
                    json.dumps(
                        _result_to_dict(event.messages),
                        ensure_ascii=False,
                        default=str,
                    )
                )
            return

        if event.type == "message_start":
            self._thinking_started = False
        elif event.type == "thinking_delta":
            if not self._thinking_started:
                click.echo("[thinking]", err=True)
                self._thinking_started = True
            click.echo(event.delta, nl=False, err=True)
        elif event.type == "text_delta":
            if self._thinking_started:
                click.echo(err=True)
                self._thinking_started = False
            click.echo(event.delta, nl=False)
        elif event.type == "message_end" and self._thinking_started:
            click.echo(err=True)
            self._thinking_started = False
        elif event.type == "tool_execution_start":
            from pi.tui.formatting import format_tool_display

            label = format_tool_display(event.tool_name, event.arguments)
            click.echo(f"\n[tool: {label}]", err=True)
        elif event.type == "tool_execution_end" and event.result and event.result.is_error:
            for block in event.result.content:
                if hasattr(block, "text"):
                    click.echo(f"  [error] {block.text}", err=True)
        elif event.type == "turn_end" and event.message.error_message:
            click.echo(f"Error: {event.message.error_message}", err=True)
