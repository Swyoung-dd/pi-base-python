"""Bash/shell 命令执行工具（跨平台）。"""

from __future__ import annotations

import locale

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
        "shell": {
            "type": "string",
            "enum": ["auto", "powershell", "cmd", "bash", "sh"],
            "description": (
                "Shell to run the command with. 'auto' picks the platform default: "
                "PowerShell on Windows, bash on macOS/Linux. Default: auto."
            ),
        },
    },
    "required": ["command"],
}


def _decode_output(data: bytes) -> str:
    """优先按 UTF-8 解码；失败时回退到系统区域编码，避免 Windows 中文乱码。"""
    if not data:
        return ""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        encoding = locale.getpreferredencoding(False) or "utf-8"
        return data.decode(encoding, errors="replace")


async def execute(call: AgentToolCall, ctx: ToolContext | None) -> AgentToolResult:
    command = call.arguments.get("command", "")
    timeout = call.arguments.get("timeout", 120)
    shell = call.arguments.get("shell", "auto")

    if not isinstance(command, str) or not command.strip():
        return AgentToolResult(
            tool_call_id=call.id,
            tool_name="bash",
            content=[TextContent(text="Command must be a non-empty string")],
            is_error=True,
        )
    if not isinstance(timeout, int | float) or not 0 < timeout <= 3600:
        return AgentToolResult(
            tool_call_id=call.id,
            tool_name="bash",
            content=[TextContent(text="Timeout must be between 0 and 3600 seconds")],
            is_error=True,
        )
    if shell not in ("auto", "powershell", "cmd", "bash", "sh"):
        return AgentToolResult(
            tool_call_id=call.id,
            tool_name="bash",
            content=[TextContent(text=f"Invalid shell: {shell}")],
            is_error=True,
        )

    if ctx is not None:
        env = ctx.ensure_env()
        cwd = str(ctx.cwd)
    else:
        from pathlib import Path

        from pi.agent.tools.execution_env import LocalExecutionEnv

        env = LocalExecutionEnv(Path.cwd())
        cwd = str(Path.cwd())

    try:
        result = await env.exec(command, cwd=cwd, timeout=timeout, shell=shell)

        output_parts = []
        if result.stdout:
            output_parts.append(_decode_output(result.stdout))
        if result.stderr:
            output_parts.append(f"STDERR:\n{_decode_output(result.stderr)}")
        output_parts.append(f"\nExit code: {result.returncode}")

        is_error = result.returncode != 0
        return AgentToolResult(
            tool_call_id=call.id,
            tool_name="bash",
            content=[TextContent(text=truncate_output("\n".join(output_parts)))],
            is_error=is_error,
        )
    except TimeoutError:
        return AgentToolResult(
            tool_call_id=call.id,
            tool_name="bash",
            content=[TextContent(text=f"Command timed out after {timeout}s")],
            is_error=True,
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
        description=(
            "Execute a shell command. Supports PowerShell on Windows and bash/sh on "
            "Unix. Output is captured and returned. Use for running tests, git commands, "
            "or other CLI tools."
        ),
        parameters=PARAMETERS,
        execute=execute,
    )
