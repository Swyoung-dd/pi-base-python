"""目录列表工具。"""

from __future__ import annotations

from pi.agent.tools.base import ToolContext, truncate_output
from pi.agent.types import AgentTool, AgentToolCall, AgentToolResult
from pi.ai.types import TextContent

PARAMETERS = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Directory path to list. Default: cwd.",
        },
        "all": {
            "type": "boolean",
            "description": "Include hidden files. Default: false.",
        },
    },
    "required": [],
}


async def execute(call: AgentToolCall, ctx: ToolContext | None) -> AgentToolResult:
    path_arg = call.arguments.get("path", ".")
    show_all = call.arguments.get("all", False)

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
                tool_name="ls",
                content=[TextContent(text=f"Directory not found: {path_arg}")],
                is_error=True,
            )
        if not file_path.is_dir():
            return AgentToolResult(
                tool_call_id=call.id,
                tool_name="ls",
                content=[TextContent(text=f"Not a directory: {path_arg}")],
                is_error=True,
            )

        entries = await env.list_dir(path_arg, show_all=show_all)
        lines = []
        for entry in entries:
            marker = "/" if entry.is_dir else ""
            lines.append(f"{entry.name}{marker}")

        result = "\n".join(lines) if lines else "(empty)"
        return AgentToolResult(
            tool_call_id=call.id,
            tool_name="ls",
            content=[TextContent(text=truncate_output(result))],
        )
    except Exception as exc:
        return AgentToolResult(
            tool_call_id=call.id,
            tool_name="ls",
            content=[TextContent(text=f"Error listing directory: {exc}")],
            is_error=True,
        )


def create_ls_tool() -> AgentTool:
    return AgentTool(
        name="ls",
        description="List directory contents.",
        parameters=PARAMETERS,
        execute=execute,
    )
