"""Grep 工具 — 按正则表达式搜索文件内容。"""

from __future__ import annotations

import re

from pi.agent.tools.base import ToolContext, truncate_output
from pi.agent.types import AgentTool, AgentToolCall, AgentToolResult
from pi.ai.types import TextContent

PARAMETERS = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "description": "Regular expression pattern to search for.",
        },
        "path": {
            "type": "string",
            "description": "Directory or file to search in. Default: cwd.",
        },
        "include": {
            "type": "string",
            "description": "File name glob filter. e.g. '*.py'",
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum matching lines. Default: 50.",
        },
    },
    "required": ["pattern"],
}


async def execute(call: AgentToolCall, ctx: ToolContext | None) -> AgentToolResult:
    pattern = call.arguments.get("pattern", "")
    path_arg = call.arguments.get("path", ".")
    include = call.arguments.get("include")
    max_results = call.arguments.get("max_results", 50)

    if ctx is not None:
        env = ctx.ensure_env()
    else:
        from pathlib import Path

        from pi.agent.tools.execution_env import LocalExecutionEnv
        env = LocalExecutionEnv(Path.cwd())

    try:
        re.compile(pattern)
    except re.error as exc:
        return AgentToolResult(
            tool_call_id=call.id,
            tool_name="grep",
            content=[TextContent(text=f"Invalid regex: {exc}")],
            is_error=True,
        )

    try:
        results = await env.grep(
            pattern, path_arg, include=include, max_results=max_results,
        )

        if not results:
            result_text = "No matches found"
        else:
            result_text = "\n".join(results)
            if len(results) >= max_results:
                result_text += f"\n\n... (truncated at {max_results} results)"

        return AgentToolResult(
            tool_call_id=call.id,
            tool_name="grep",
            content=[TextContent(text=truncate_output(result_text))],
        )
    except Exception as exc:
        return AgentToolResult(
            tool_call_id=call.id,
            tool_name="grep",
            content=[TextContent(text=f"Error searching: {exc}")],
            is_error=True,
        )


def create_grep_tool() -> AgentTool:
    return AgentTool(
        name="grep",
        description="Search file contents by regex. Supports file name filtering.",
        parameters=PARAMETERS,
        execute=execute,
    )
