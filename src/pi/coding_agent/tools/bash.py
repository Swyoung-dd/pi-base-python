"""Bash/shell 命令执行工具。"""

from __future__ import annotations

import asyncio
from pathlib import Path

from pi.agent.tools.base import ToolContext, truncate_output
from pi.agent.types import AgentTool, AgentToolCall, AgentToolResult
from pi.ai.types import TextContent

PARAMETERS = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "description": "The shell command to execute.",
        },
        "timeout": {
            "type": "integer",
            "description": "Timeout in seconds. Default: 120.",
        },
    },
    "required": ["command"],
}


async def execute(call: AgentToolCall, ctx: ToolContext | None) -> AgentToolResult:
    command = call.arguments.get("command", "")
    timeout = call.arguments.get("timeout", 120)

    cwd = str(ctx.cwd) if ctx else None

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return AgentToolResult(
                tool_call_id=call.id,
                tool_name="bash",
                content=[TextContent(text=f"Command timed out after {timeout}s")],
                is_error=True,
            )

        output_parts = []
        if stdout:
            output_parts.append(stdout.decode("utf-8", errors="replace"))
        if stderr:
            output_parts.append(f"STDERR:\n{stderr.decode('utf-8', errors='replace')}")
        output_parts.append(f"\nExit code: {proc.returncode}")

        is_error = proc.returncode != 0
        return AgentToolResult(
            tool_call_id=call.id,
            tool_name="bash",
            content=[TextContent(text=truncate_output("\n".join(output_parts)))],
            is_error=is_error,
        )
    except Exception as exc:
        return AgentToolResult(
            tool_call_id=call.id,
            tool_name="bash",
            content=[TextContent(text=f"Error executing command: {exc}")],
            is_error=True,
        )


def create_bash_tool() -> AgentTool:
    return AgentTool(
        name="bash",
        description="Execute a shell command and return stdout, stderr, and exit code.",
        parameters=PARAMETERS,
        execute=execute,
    )
