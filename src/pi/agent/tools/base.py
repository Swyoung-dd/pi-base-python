"""工具执行上下文与基础工具函数。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ToolContext:
    """传递给工具 execute 函数的上下文。

    提供工作目录和可选的共享状态。
    与文件系统交互的工具以 cwd 为基础路径。
    """
    cwd: Path = field(default_factory=Path.cwd)
    state: dict[str, Any] = field(default_factory=dict)


def truncate_output(text: str, max_chars: int = 30000) -> str:
    """将工具输出截断到最大字符数。"""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return (
        text[:half]
        + f"\n\n... ({len(text) - max_chars} characters truncated) ...\n\n"
        + text[-half:]
    )
