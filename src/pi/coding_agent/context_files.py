"""AGENTS.md 与 CLAUDE.md 上下文文件发现。"""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ContextFile:
    path: Path
    content: str


def _first_context_file(directory: Path) -> Path | None:
    for name in ("AGENTS.md", "CLAUDE.md"):
        candidate = directory / name
        if candidate.is_file():
            return candidate.resolve()
    return None


def load_context_files(
    cwd: Path,
    user_config_dir: Path | None = None,
) -> list[ContextFile]:
    """按用户级、父目录到当前目录的顺序加载上下文文件。"""
    project_dir = cwd.resolve()
    user_dir = (user_config_dir or (Path.home() / ".piy")).resolve()
    candidates: list[Path] = []
    user_file = _first_context_file(user_dir)
    if user_file is not None:
        candidates.append(user_file)
    directories = [project_dir, *project_dir.parents]
    for directory in reversed(directories):
        project_file = _first_context_file(directory)
        if project_file is not None:
            candidates.append(project_file)

    loaded: list[ContextFile] = []
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        loaded.append(ContextFile(path=path, content=path.read_text(encoding="utf-8")))
    return loaded


def format_context_files(files: list[ContextFile]) -> str:
    """将上下文文件包装为带来源路径的系统提示词片段。"""
    blocks = []
    for item in files:
        path = html.escape(str(item.path), quote=True)
        blocks.append(f'<context_file path="{path}">\n{item.content}\n</context_file>')
    return "\n\n".join(blocks)
