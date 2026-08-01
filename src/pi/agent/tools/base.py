"""工具执行上下文与基础工具函数。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ToolContext:
    """传递给工具 execute 函数的上下文。

    提供工作目录、可选的共享状态和可注入的执行环境。
    当 env 存在时，工具应优先使用 env 进行文件系统和进程操作；
    当 env 为 None 时，工具回退到使用 cwd 直接操作文件系统。
    """

    cwd: Path = field(default_factory=Path.cwd)
    state: dict[str, Any] = field(default_factory=dict)
    env: Any = None  # ExecutionEnv | None，避免循环导入用 Any

    def get_env(self) -> Any:
        """返回执行环境，如果不存在则返回 None。"""
        return self.env

    def ensure_env(self) -> Any:
        """返回执行环境；不存在时创建 LocalExecutionEnv。"""
        if self.env is not None:
            return self.env
        from pi.agent.tools.execution_env import LocalExecutionEnv

        self.env = LocalExecutionEnv(self.cwd)
        return self.env


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
