"""Bash/shell 命令执行工具（跨平台）。"""

from __future__ import annotations

import asyncio
import locale
import os
import shutil
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


def _resolve_shell(shell: str) -> str:
    """解析 shell 参数；auto 时按平台选择默认 shell。"""
    if shell == "auto":
        return "powershell" if os.name == "nt" else "bash"
    return shell


def _build_argv(shell: str, command: str) -> tuple[list[str], bool]:
    """构建要启动的 argv 与是否使用 shell=True。"""
    if shell == "cmd" and os.name == "nt":
        return [command], True
    if shell == "powershell":
        executable = shutil.which("pwsh") or shutil.which("powershell")
        if executable is None:
            raise RuntimeError("PowerShell not found on PATH")
        return [executable, "-NoProfile", "-NonInteractive", "-Command", command], False
    if shell in ("bash", "sh"):
        executable = shutil.which(shell) or shutil.which("bash") or shutil.which("sh")
        if executable is None:
            raise RuntimeError(
                f"{shell} not found on PATH; on Windows install Git for Windows "
                "or use shell='powershell'"
            )
        return [executable, "-c", command], False
    raise RuntimeError(f"Unsupported shell: {shell}")


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

    cwd = str(ctx.cwd) if ctx else None

    try:
        resolved_shell = _resolve_shell(shell)
        argv, use_shell = _build_argv(resolved_shell, command)
        process_options = (
            {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
            if os.name == "nt"
            else {"start_new_session": True}
        )
        proc = subprocess.Popen(
            argv,
            shell=use_shell,
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
            output_parts.append(_decode_output(stdout))
        if stderr:
            output_parts.append(f"STDERR:\n{_decode_output(stderr)}")
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
        with suppress(TimeoutError):
            await asyncio.wait_for(
                asyncio.to_thread(
                    subprocess.run,
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=10,
                ),
                timeout=15,
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
        description=(
            "Execute a shell command and return stdout, stderr, and exit code. "
            "Uses the platform default shell (Windows: PowerShell; macOS/Linux: bash); "
            "pass 'shell' to override with powershell, cmd, bash, or sh."
        ),
        parameters=PARAMETERS,
        execute=execute,
    )
