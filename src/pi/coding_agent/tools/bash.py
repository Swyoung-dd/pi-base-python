"""Bash/shell 命令执行工具。"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
from contextlib import suppress

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

    cwd = str(ctx.cwd) if ctx else None

    try:
        process_options = (
            {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
            if os.name == "nt"
            else {"start_new_session": True}
        )
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            **process_options,
        )

        communicate_task = asyncio.create_task(asyncio.to_thread(proc.communicate))
        try:
            stdout, stderr = await asyncio.wait_for(
                asyncio.shield(communicate_task),
                timeout=timeout,
            )
        except TimeoutError:
            await _terminate_process_tree(proc, communicate_task)
            return AgentToolResult(
                tool_call_id=call.id,
                tool_name="bash",
                content=[TextContent(text=f"Command timed out after {timeout}s")],
                is_error=True,
            )
        except asyncio.CancelledError:
            await asyncio.shield(_terminate_process_tree(proc, communicate_task))
            raise

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


async def _terminate_process_tree(
    proc: subprocess.Popen,
    communicate_task: asyncio.Task[tuple[bytes, bytes]],
) -> None:
    """终止 shell 及其派生进程，避免超时或取消后遗留后台任务。"""
    if proc.returncode is not None:
        return
    if os.name == "nt":
        await asyncio.to_thread(
            subprocess.run,
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        with suppress(ProcessLookupError):
            proc.kill()
    else:
        with suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
    try:
        await asyncio.wait_for(asyncio.shield(communicate_task), timeout=2)
    except TimeoutError:
        communicate_task.cancel()
        await asyncio.gather(communicate_task, return_exceptions=True)


def create_bash_tool() -> AgentTool:
    return AgentTool(
        name="bash",
        description="Execute a shell command and return stdout, stderr, and exit code.",
        parameters=PARAMETERS,
        execute=execute,
    )
