"""TUI 使用的无状态文本和工作区格式化函数。"""

from __future__ import annotations

import subprocess
from pathlib import Path


def format_tokens(tokens: int) -> str:
    """将 token 数格式化为适合状态栏的紧凑文本。"""
    if tokens < 1_000:
        return str(tokens)
    if tokens < 10_000:
        return f"{tokens / 1_000:.1f}k"
    if tokens < 1_000_000:
        return f"{round(tokens / 1_000)}k"
    if tokens < 10_000_000:
        return f"{tokens / 1_000_000:.1f}m"
    return f"{round(tokens / 1_000_000)}m"


def split_complete_markdown(text: str) -> tuple[str, str]:
    """按围栏外的空行切分可安全渲染的 Markdown 前缀。"""
    boundary = 0
    offset = 0
    fence: str | None = None
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        marker = "```" if stripped.startswith("```") else None
        if marker is None and stripped.startswith("~~~"):
            marker = "~~~"
        if marker is not None:
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
        offset += len(line)
        if fence is None and not line.strip():
            boundary = offset
    return text[:boundary], text[boundary:]


def read_git_status(cwd: Path) -> tuple[str, int] | None:
    """读取工作区 Git 分支和变更数量，非仓库目录返回空。"""
    try:
        result = subprocess.run(
            ["git", "status", "--short", "--branch"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=1,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = result.stdout.splitlines()
    if result.returncode != 0 or not lines or not lines[0].startswith("## "):
        return None
    branch = lines[0].removeprefix("## ").split("...", maxsplit=1)[0].strip()
    return branch, len(lines) - 1


def format_tool_display(tool_name: str, arguments: dict) -> str:
    """根据工具名称和参数提取有意义的简短描述。"""
    path = arguments.get("path", "")
    pattern = arguments.get("pattern", "")
    command = arguments.get("command", "")

    if tool_name == "bash":
        return f"bash: {command}" if command else "bash"
    if tool_name in {"read", "write", "edit", "ls"}:
        return f"{tool_name}: {path}" if path else tool_name
    if tool_name == "find":
        suffix = f": {pattern}" if pattern else ""
        dir_info = f" in {path}" if path and path != "." else ""
        return f"find{suffix}{dir_info}"
    if tool_name == "grep":
        suffix = f": /{pattern}/" if pattern else ""
        return f"grep{suffix}"
    if tool_name == "subagent":
        task = arguments.get("task", "")
        return f"subagent: {task}" if task else "subagent"
    return tool_name


def format_tool_target(tool_name: str, arguments: dict) -> str:
    """生成工具树子项文本，去除分组标题已表达的工具名称。"""
    display = format_tool_display(tool_name, arguments)
    return display.removeprefix(f"{tool_name}: ")

