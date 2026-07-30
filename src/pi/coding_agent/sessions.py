"""编码 agent 的会话发现与选择。"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pi.agent.session import JsonlStorage

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
