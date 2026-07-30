"""创建 compact 工具 —— 允许 AI agent 主动压缩对话上下文。"""

from __future__ import annotations

from typing import Any

from pi.agent.types import (
    AgentTool,
    AgentToolCall,
    AgentToolResult,
    ContextCompactionRequest,
)
from pi.ai.types import TextContent


def create_compact_tool() -> AgentTool:
    """创建一个向 agent loop 提交上下文压缩请求的工具。"""

    async def execute(call: AgentToolCall, _context: Any = None) -> AgentToolResult:
        target_tokens = call.arguments.get("target_tokens")
        if target_tokens is not None and not isinstance(target_tokens, int):
            # 可能被 JSON Schema 约束，但再确保一遍
            target_tokens = int(target_tokens)

        return AgentToolResult(
            tool_call_id=call.id,
            tool_name=call.name,
            content=[
                TextContent(
                    text="Context compaction was requested and will run before the next "
                    "model call."
                )
            ],
            details=ContextCompactionRequest(target_tokens=target_tokens),
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
