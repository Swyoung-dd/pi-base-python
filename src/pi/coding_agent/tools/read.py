"""文件读取工具。"""

from __future__ import annotations

from pi.agent.tools.base import ToolContext, truncate_output
from pi.agent.types import AgentTool, AgentToolCall, AgentToolResult
from pi.ai.types import TextContent

PARAMETERS = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Path to the file to read. Relative to cwd.",
        },
        "offset": {
            "type": "integer",
            "description": "Line number to start reading from (1-based). Default: 1.",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of lines to read. Default: 2000.",
        },
    },
    "required": ["path"],
}


async def execute(call: AgentToolCall, ctx: ToolContext | None) -> AgentToolResult:
    path_arg = call.arguments.get("path", "")
    offset = call.arguments.get("offset", 1)
    limit = call.arguments.get("limit", 2000)

    if ctx is not None:
        env = ctx.ensure_env()
    else:
        from pathlib import Path

        from pi.agent.tools.execution_env import LocalExecutionEnv
        env = LocalExecutionEnv(Path.cwd())

    try:
        file_path = env.resolve(path_arg)
        if not file_path.exists():
            return AgentToolResult(
                tool_call_id=call.id,
                tool_name="read",
                content=[TextContent(text=f"File not found: {path_arg}")],
                is_error=True,
            )
        if file_path.is_dir():
            return AgentToolResult(
                tool_call_id=call.id,
                tool_name="read",
                content=[TextContent(text=f"Path is a directory: {path_arg}")],
                is_error=True,
            )

        text = await env.read(path_arg)
        lines = text.split("\n")

        start = max(0, offset - 1)
        end = min(len(lines), start + limit)
        selected = lines[start:end]

        numbered = []
        for i, line in enumerate(selected, start=start + 1):
            numbered.append(f"{i:6d}\t{line}")
        result = "\n".join(numbered)

        if end < len(lines):
            result += f"\n\n... ({len(lines) - end} more lines)"

        return AgentToolResult(
            tool_call_id=call.id,
            tool_name="read",
            content=[TextContent(text=truncate_output(result))],
        )
    except Exception as exc:
        return AgentToolResult(
            tool_call_id=call.id,
            tool_name="read",
            content=[TextContent(text=f"Error reading file: {exc}")],
            is_error=True,
        )


def create_read_tool() -> AgentTool:
    return AgentTool(
        name="read",
        description="Read the contents of a file. Supports line offset and limit.",
        parameters=PARAMETERS,
        execute=execute,
    )
