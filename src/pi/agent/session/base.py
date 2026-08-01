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

from pydantic import TypeAdapter

from pi.agent.types import AgentMessage
from pi.ai.types import Usage

_MESSAGE_ADAPTER = TypeAdapter(AgentMessage)
_MESSAGES_ADAPTER = TypeAdapter(list[AgentMessage])
SESSION_VERSION = 2


def create_session_header(cwd: str | None = None) -> dict[str, Any]:
    """创建会话文件头行（v2 格式）。"""
    return {
        "type": "session",
        "version": SESSION_VERSION,
        "id": _gen_id(),
        "timestamp": _utc_now_iso(),
        "cwd": cwd or "",
    }


def is_session_header(data: dict[str, Any]) -> bool:
    """判断 JSON 字典是否为会话头行。"""
    return data.get("type") == "session" and "version" in data


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

    def to_dict(self) -> dict[str, Any]:
        """转换为可写入 JSON 的字典。"""
        message = None
        if self.message is not None:
            message = _MESSAGE_ADAPTER.dump_python(self.message, mode="json")
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "type": self.type,
            "message": message,
            "data": self.data,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SessionEntry:
        """从 JSON 字典恢复并校验会话条目。"""
        raw_message = value.get("message")
        message = _MESSAGE_ADAPTER.validate_python(raw_message) if raw_message else None
        return cls(
            id=value["id"],
            parent_id=value.get("parent_id"),
            type=value.get("type", "message"),
            message=message,
            data=value.get("data"),
            timestamp=value.get("timestamp") or _utc_now_iso(),
        )


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
    async def get_entries(self) -> list[SessionEntry]:
        """按写入顺序获取全部条目。"""
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

    async def append_compaction(
        self,
        messages: list[AgentMessage],
        original_tokens: int,
        compacted_tokens: int,
        dropped_messages: int,
        usage: Usage | None = None,
        *,
        summary: str | None = None,
        retained_tail: list[AgentMessage] | None = None,
    ) -> str:
        """追加压缩检查点。

        v2 格式：存储 summary（摘要文本）和 retained_tail（保留的消息尾部），
        同时保留 messages 字段以兼容旧版读取。
        """
        leaf = await self.get_leaf_id()
        data: dict[str, Any] = {
            "messages": _MESSAGES_ADAPTER.dump_python(messages, mode="json"),
            "original_tokens": original_tokens,
            "compacted_tokens": compacted_tokens,
            "dropped_messages": dropped_messages,
            "usage": usage.model_dump(mode="json") if usage is not None else None,
        }
        if summary is not None:
            data["summary"] = summary
        if retained_tail is not None:
            data["retained_tail"] = _MESSAGES_ADAPTER.dump_python(
                retained_tail,
                mode="json",
            )
        entry = SessionEntry(
            parent_id=leaf,
            type="compaction",
            data=data,
        )
        entry_id = await self.append(entry)
        await self.set_leaf_id(entry_id)
        return entry_id

        entry_id = await self.append(entry)
        await self.set_leaf_id(entry_id)
        return entry_id

    async def append_thinking_level_change(self, level: str) -> str:
        """记录思考级别变更。"""
        leaf = await self.get_leaf_id()
        entry = SessionEntry(
            parent_id=leaf,
            type="thinking_level_change",
            data={"level": level},
        )
        entry_id = await self.append(entry)
        await self.set_leaf_id(entry_id)
        return entry_id

    async def append_branch_summary(self, summary: str) -> str:
        """记录分支摘要。"""
        leaf = await self.get_leaf_id()
        entry = SessionEntry(
            parent_id=leaf,
            type="branch_summary",
            data={"summary": summary},
        )
        entry_id = await self.append(entry)
        await self.set_leaf_id(entry_id)
        return entry_id

    async def append_custom_entry(
        self,
        entry_type: str,
        data: dict[str, Any],
    ) -> str:
        """追加自定义类型条目。"""
        leaf = await self.get_leaf_id()
        entry = SessionEntry(
            parent_id=leaf,
            type=entry_type,
            data=data,
        )
        entry_id = await self.append(entry)
        await self.set_leaf_id(entry_id)
        return entry_id

    async def append_session_info(self, key: str, value: Any) -> str:
        """记录或更新会话信息键值对。"""
        leaf = await self.get_leaf_id()
        entry = SessionEntry(
            parent_id=leaf,
            type="session_info",
            data={"key": key, "value": value},
        )
        entry_id = await self.append(entry)
        await self.set_leaf_id(entry_id)
        return entry_id

    async def append_model_change(self, provider: str, model_id: str) -> str:
        """在当前分支记录后续请求使用的模型。"""
        leaf = await self.get_leaf_id()
        entry = SessionEntry(
            parent_id=leaf,
            type="model_change",
            data={"provider": provider, "model_id": model_id},
        )
        entry_id = await self.append(entry)
        await self.set_leaf_id(entry_id)
        return entry_id

    async def branch_from(self, entry_id: str) -> str:
        """从指定条目创建可持久恢复的新分支。"""
        if await self.get(entry_id) is None:
            raise KeyError(f"未知会话条目: {entry_id}")
        entry = SessionEntry(
            parent_id=entry_id,
            type="branch",
            data={"source_entry_id": entry_id},
        )
        branch_id = await self.append(entry)
        await self.set_leaf_id(branch_id)
        return branch_id

    async def get_model_selection(self) -> tuple[str, str] | None:
        """返回当前分支最近一次模型选择。"""
        for entry in reversed(await self.get_branch()):
            if entry.type != "model_change" or not entry.data:
                continue
            provider = entry.data.get("provider")
            model_id = entry.data.get("model_id")
            if isinstance(provider, str) and isinstance(model_id, str):
                return provider, model_id
        return None

    async def get_context_messages(self) -> list[AgentMessage]:
        """从最近压缩检查点恢复有效上下文，并追加其后的消息。

        v2 格式优先使用 summary + retained_tail 重建上下文；
        旧格式（仅 messages 字段）仍然兼容。
        """
        messages: list[AgentMessage] = []
        for entry in await self.get_branch():
            if entry.type == "compaction" and entry.data:
                if entry.data.get("retained_tail") is not None:
                    # v2: retained_tail 存储完整的压缩后上下文
                    messages = _MESSAGES_ADAPTER.validate_python(
                        entry.data["retained_tail"],
                    )
                elif entry.data.get("messages") is not None:
                    # v1: 直接使用存储的完整消息列表
                    messages = _MESSAGES_ADAPTER.validate_python(
                        entry.data["messages"],
                    )
            elif entry.type == "message" and entry.message is not None:
                messages.append(entry.message)
        return messages
