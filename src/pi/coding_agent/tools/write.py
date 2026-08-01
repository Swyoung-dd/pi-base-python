"""文件写入工具。"""

from __future__ import annotations

from pi.agent.tools.base import ToolContext
from pi.agent.types import AgentTool, AgentToolCall, AgentToolResult
from pi.ai.types import TextContent

PARAMETERS = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Path to the file to write. Relative to cwd.",
        },
        "content": {
            "type": "string",
            "description": "The full content to write to the file.",
        },
    },
    "required": ["path", "content"],
}


async def execute(call: AgentToolCall, ctx: ToolContext | None) -> AgentToolResult:
    path_arg = call.arguments.get("path", "")
    content = call.arguments.get("content", "")

    if ctx is not None:
        env = ctx.ensure_env()
    else:
        from pathlib import Path

        from pi.agent.tools.execution_env import LocalExecutionEnv

        env = LocalExecutionEnv(Path.cwd())

    try:
        await env.write(path_arg, content)
        line_count = content.count("\n") + 1 if content else 0
        return AgentToolResult(
            tool_call_id=call.id,
            tool_name="write",
            content=[TextContent(text=f"Wrote {line_count} lines to {path_arg}")],
        )
    except Exception as exc:
        return AgentToolResult(
            tool_call_id=call.id,
            tool_name="write",
            content=[TextContent(text=f"Error writing file: {exc}")],
            is_error=True,
        )


def create_write_tool() -> AgentTool:
    return AgentTool(
        name="write",
        description="Write content to a file. Creates or overwrites the file.",
        parameters=PARAMETERS,
        execute=execute,
    )
