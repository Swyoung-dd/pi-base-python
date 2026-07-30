"""编码 agent 的会话发现与选择。"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pi.agent.session import JsonlStorage, SessionEntry
from pi.agent.types import AgentUserMessage

_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class SessionInfo:
    """用于会话列表展示的摘要。"""

    session_id: str
    updated_at: datetime
    message_count: int
    preview: str


def validate_session_id(session_id: str) -> str:
    """校验会话 ID，确保它只能映射到 sessions 目录内的文件。"""
    if not _SESSION_ID_PATTERN.fullmatch(session_id):
        raise ValueError("会话 ID 只能包含字母、数字、下划线和连字符")
    return session_id


def session_path(sessions_dir: Path, session_id: str) -> Path:
    return sessions_dir / f"{validate_session_id(session_id)}.jsonl"


def new_session_id() -> str:
    return uuid.uuid4().hex


def resolve_entry_id(entries: list[SessionEntry], value: str) -> str:
    """将完整 ID 或唯一前缀解析为条目 ID。"""
    exact = next((entry.id for entry in entries if entry.id == value), None)
    if exact is not None:
        return exact
    matches = [entry.id for entry in entries if entry.id.startswith(value)]
    if not matches:
        raise ValueError(f"会话条目不存在: {value}")
    if len(matches) > 1:
        raise ValueError(f"会话条目 ID 前缀不唯一: {value}")
    return matches[0]


def format_session_tree(entries: list[SessionEntry], leaf_id: str | None) -> str:
    """把只追加的会话条目格式化为紧凑 ASCII 树。"""
    if not entries:
        return "No session entries."
    children: dict[str | None, list[SessionEntry]] = {}
    for entry in entries:
        children.setdefault(entry.parent_id, []).append(entry)

    lines: list[str] = []

    def label(entry: SessionEntry) -> str:
        if entry.type == "message" and isinstance(entry.message, AgentUserMessage):
            content = entry.message.content
            if isinstance(content, str):
                return f"user: {content.replace(chr(10), ' ')[:48]}"
            return "user message"
        if entry.type == "message" and entry.message is not None:
            return entry.message.role
        if entry.type == "model_change" and entry.data:
            return f"model: {entry.data.get('provider')}/{entry.data.get('model_id')}"
        if entry.type == "branch":
            return "branch"
        return entry.type

    def visit(entry: SessionEntry, prefix: str, is_last: bool) -> None:
        connector = "`- " if is_last else "+- "
        marker = "*" if entry.id == leaf_id else " "
        lines.append(f"{prefix}{connector}{marker} {entry.id[:8]} {label(entry)}")
        nested = children.get(entry.id, [])
        child_prefix = prefix + ("   " if is_last else "|  ")
        for index, child in enumerate(nested):
            visit(child, child_prefix, index == len(nested) - 1)

    roots = children.get(None, [])
    for index, root in enumerate(roots):
        visit(root, "", index == len(roots) - 1)
    return "\n".join(lines)


async def list_sessions(sessions_dir: Path) -> list[SessionInfo]:
    """按最近修改时间倒序列出可恢复的会话。"""
    sessions: list[SessionInfo] = []
    if not sessions_dir.exists():
        return sessions
    for path in sessions_dir.glob("*.jsonl"):
        try:
            storage = JsonlStorage(path)
            branch = await storage.get_branch()
            messages = [entry.message for entry in branch if entry.message is not None]
            preview = ""
            if messages:
                content = messages[0].content
                if isinstance(content, str):
                    preview = content.replace("\n", " ")[:60]
            sessions.append(
                SessionInfo(
                    session_id=path.stem,
                    updated_at=datetime.fromtimestamp(path.stat().st_mtime, UTC),
                    message_count=len(messages),
                    preview=preview,
                )
            )
        except (OSError, ValueError):
            continue
    return sorted(sessions, key=lambda item: item.updated_at, reverse=True)


async def resolve_session_id(
    sessions_dir: Path,
    requested_id: str | None = None,
    continue_latest: bool = False,
) -> str:
    if requested_id:
        return validate_session_id(requested_id)
    if continue_latest:
        sessions = await list_sessions(sessions_dir)
        if sessions:
            return sessions[0].session_id
    return new_session_id()
