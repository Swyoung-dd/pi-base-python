"""内存会话存储。"""

from __future__ import annotations

from pi.agent.session.base import SessionEntry, SessionStorage


class MemoryStorage(SessionStorage):
    """简单的内存会话存储。进程退出后丢失。"""

    def __init__(self) -> None:
        self._entries: dict[str, SessionEntry] = {}
        self._leaf_id: str | None = None

    async def append(self, entry: SessionEntry) -> str:
        self._entries[entry.id] = entry
        return entry.id

    async def get(self, entry_id: str) -> SessionEntry | None:
        return self._entries.get(entry_id)

    async def get_branch(self) -> list[SessionEntry]:
        if self._leaf_id is None:
            return []
        chain: list[SessionEntry] = []
        current = self._entries.get(self._leaf_id)
        while current is not None:
            chain.append(current)
            current = self._entries.get(current.parent_id) if current.parent_id else None
        chain.reverse()
        return chain

    async def get_leaf_id(self) -> str | None:
        return self._leaf_id

    async def set_leaf_id(self, entry_id: str) -> None:
        if entry_id not in self._entries:
            raise KeyError(f"未知会话条目: {entry_id}")
        self._leaf_id = entry_id
