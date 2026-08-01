"""文件查找工具 — 按名称模式查找文件。"""

from __future__ import annotations

from pi.agent.tools.base import ToolContext, truncate_output
from pi.agent.types import AgentTool, AgentToolCall, AgentToolResult
from pi.ai.types import TextContent

PARAMETERS = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Directory to search in. Default: cwd.",
        },
        "pattern": {
            "type": "string",
            "description": "Glob pattern to match file names. e.g. '*.py'",
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum results. Default: 100.",
        },
    },
    "required": ["pattern"],
}


async def execute(call: AgentToolCall, ctx: ToolContext | None) -> AgentToolResult:
    path_arg = call.arguments.get("path", ".")
    pattern = call.arguments.get("pattern", "*")
    max_results = call.arguments.get("max_results", 100)

    if ctx is not None:
        env = ctx.ensure_env()
    else:
        from pathlib import Path

        from pi.agent.tools.execution_env import LocalExecutionEnv

        env = LocalExecutionEnv(Path.cwd())

    try:
        search_dir = env.resolve(path_arg)
        if not search_dir.exists():
            return AgentToolResult(
                tool_call_id=call.id,
                tool_name="find",
                content=[TextContent(text=f"Directory not found: {path_arg}")],
                is_error=True,
            )

        results = await env.find(path_arg, pattern, max_results=max_results)

        if not results:
            result_text = "No files found"
        else:
            result_text = "\n".join(results)
            if len(results) >= max_results:
                result_text += f"\n\n... (truncated at {max_results} results)"

        return AgentToolResult(
            tool_call_id=call.id,
            tool_name="find",
            content=[TextContent(text=truncate_output(result_text))],
        )
    except Exception as exc:
        return AgentToolResult(
            tool_call_id=call.id,
            tool_name="find",
            content=[TextContent(text=f"Error finding files: {exc}")],
            is_error=True,
        )


def create_find_tool() -> AgentTool:
    return AgentTool(
        name="find",
        description="Find files by name pattern, recursively.",
        parameters=PARAMETERS,
        execute=execute,
    )
