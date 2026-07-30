"""创建 compact 工具 —— 允许 AI agent 主动压缩对话上下文。"""

from __future__ import annotations

from typing import Any

from pi.agent.types import AgentTool, AgentToolCall, AgentToolResult
from pi.ai.types import TextContent


def create_compact_tool(get_agent: Any = None) -> AgentTool:
    """创建一个可供 AI 调用的上下文压缩工具。

    Args:
        get_agent: 返回当前 Agent 实例的可调用对象（支持懒引用避免循环导入）。
    """

    async def execute(call: AgentToolCall, _context: Any = None) -> AgentToolResult:
        target_tokens = call.arguments.get("target_tokens")
        if target_tokens is not None and not isinstance(target_tokens, int):
            # 可能被 JSON Schema 约束，但再确保一遍
            target_tokens = int(target_tokens)

        if get_agent is None:
            return AgentToolResult(
                tool_call_id=call.id,
                tool_name=call.name,
                content=[TextContent(text="Compact tool is not available in this runtime.")],
                is_error=True,
            )

        agent = get_agent()
        if agent is None:
            return AgentToolResult(
                tool_call_id=call.id,
                tool_name=call.name,
                content=[TextContent(text="Agent is not available for compaction.")],
                is_error=True,
            )

        if agent.is_busy:
            return AgentToolResult(
                tool_call_id=call.id,
                tool_name=call.name,
                content=[
                    TextContent(text="Agent is currently busy. Retry after the current turn.")
                ],
                is_error=True,
            )

        try:
            result = await agent.compact(target_tokens)
        except RuntimeError as exc:
            return AgentToolResult(
                tool_call_id=call.id,
                tool_name=call.name,
                content=[TextContent(text=str(exc))],
                is_error=True,
            )

        if result.dropped_messages == 0:
            return AgentToolResult(
                tool_call_id=call.id,
                tool_name=call.name,
                content=[
                    TextContent(
                        text=f"No compaction was needed. "
                        f"Current context: {result.compacted_tokens} tokens. "
                        f"All {result.original_tokens} tokens of history are already within budget."
                    )
                ],
            )
        return AgentToolResult(
            tool_call_id=call.id,
            tool_name=call.name,
            content=[
                TextContent(
                    text=f"Context compacted successfully. "
                    f"Tokens: {result.original_tokens} → {result.compacted_tokens}. "
                    f"Dropped {result.dropped_messages} older messages while preserving "
                    f"a summary and the most recent turns."
                )
            ],
        )

    return AgentTool(
        name="compact",
        description=(
            "Actively compact the conversation context to stay within the model's context window. "
            "Call this when you estimate the conversation history is growing large and you need to "
            "preserve key information while freeing up space. The compaction summarizes older "
            "messages and keeps recent turns intact. Optionally specify a target token count."
        ),
        parameters={
            "type": "object",
            "properties": {
                "target_tokens": {
                    "type": "integer",
                    "description": (
                        "Optional. Target number of tokens after compaction. "
                        "If not specified, uses the agent's default compaction threshold."
                    ),
                }
            },
        },
        execute=execute,
    )
