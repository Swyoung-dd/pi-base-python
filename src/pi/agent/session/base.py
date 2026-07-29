"""会话存储接口。

会话将 agent 的对话记录存储为条目序列。
每个条目是消息、模型变更或其他会话事件。
存储为只追加模式，支持会话树的分支。
"""

from __future__ import annotations

import abc
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pi.agent.types import AgentMessage


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _gen_id() -> str:
    return uuid.uuid4().hex


@dataclass
class SessionEntry:
    """会话记录中的单个条目。"""
    id: str = field(default_factory=_gen_id)
    parent_id: str | None = None
    type: str = "message"
    message: AgentMessage | None = None
    data: dict[str, Any] | None = None
    timestamp: str = field(default_factory=_utc_now_iso)


class SessionStorage(abc.ABC):
    """会话存储后端的抽象基类。"""

    @abc.abstractmethod
    async def append(self, entry: SessionEntry) -> str:
        """追加条目。返回条目 ID。"""
        ...

    @abc.abstractmethod
    async def get(self, entry_id: str) -> SessionEntry | None:
        """按 ID 获取条目。"""
        ...

    @abc.abstractmethod
    async def get_branch(self) -> list[SessionEntry]:
        """获取从根到当前叶节点的所有条目。"""
        ...

    @abc.abstractmethod
    async def get_leaf_id(self) -> str | None:
        """获取当前叶节点条目 ID。"""
        ...

    @abc.abstractmethod
    async def set_leaf_id(self, entry_id: str) -> None:
        """设置当前叶节点条目 ID。"""
        ...

    async def append_message(self, message: AgentMessage, parent_id: str | None = None) -> str:
        """便捷方法：追加消息条目。"""
        leaf = await self.get_leaf_id()
        entry = SessionEntry(
            parent_id=parent_id or leaf,
            type="message",
            message=message,
        )
        entry_id = await self.append(entry)
        await self.set_leaf_id(entry_id)
        return entry_id
