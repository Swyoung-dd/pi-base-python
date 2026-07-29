"""基于 JSONL 文件的会话存储。

每个会话是一个 .jsonl 文件，每行是一个 JSON 编码的 SessionEntry。
支持只追加写入和分支重建。
"""

from __future__ import annotations

import json
from pathlib import Path

from pi.agent.session.base import SessionEntry, SessionStorage


class JsonlStorage(SessionStorage):
    """基于 JSON Lines 格式的文件会话存储。"""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, SessionEntry] = {}
        self._leaf_id: str | None = None
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                entry = SessionEntry(
                    id=data["id"],
                    parent_id=data.get("parent_id"),
                    type=data.get("type", "message"),
                    message=data.get("message"),
                    data=data.get("data"),
                    timestamp=data.get("timestamp", ""),
                )
                self._entries[entry.id] = entry
                self._leaf_id = entry.id

    def _write_entry(self, entry: SessionEntry) -> None:
        data = {
            "id": entry.id,
            "parent_id": entry.parent_id,
            "type": entry.type,
            "message": entry.message,
            "data": entry.data,
            "timestamp": entry.timestamp,
        }
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, default=str) + "\n")

    async def append(self, entry: SessionEntry) -> str:
        self._entries[entry.id] = entry
        self._write_entry(entry)
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
        self._leaf_id = entry_id
