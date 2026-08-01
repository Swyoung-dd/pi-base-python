"""基于 JSONL 文件的会话存储。

每个会话是一个 .jsonl 文件，每行是一个 JSON 编码的 SessionEntry。
支持只追加写入和分支重建。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from pi.agent.session.base import SessionEntry, SessionStorage


class JsonlStorage(SessionStorage):
    """基于 JSON Lines 格式的文件会话存储。"""

    def __init__(self, path: str | Path, cwd: str | None = None) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, SessionEntry] = {}
        self._leaf_id: str | None = None
        self._load()
        if not self._entries and not self._path.exists():
            self._write_header(cwd)

    def _load(self) -> None:
        if not self._path.exists():
            return
        lines = self._path.read_text(encoding="utf-8").splitlines()
        nonempty_indexes = [index for index, line in enumerate(lines) if line.strip()]
        last_index = nonempty_indexes[-1] if nonempty_indexes else -1
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                if index == last_index:
                    break
                raise ValueError(f"会话文件第 {index + 1} 行损坏: {self._path}") from exc
            if data.get("type") == "session":
                continue
            entry = SessionEntry.from_dict(data)
            self._entries[entry.id] = entry
            self._leaf_id = entry.id

    def _write_entry(self, entry: SessionEntry) -> None:
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    async def append(self, entry: SessionEntry) -> str:
        self._write_entry(entry)
        self._entries[entry.id] = entry
        return entry.id

    async def get(self, entry_id: str) -> SessionEntry | None:
        return self._entries.get(entry_id)

    async def get_entries(self) -> list[SessionEntry]:
        return list(self._entries.values())

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

    def _write_header(self, cwd: str | None = None) -> None:
        from pi.agent.session.base import create_session_header
        header = create_session_header(cwd)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(header, ensure_ascii=False))
            f.write(chr(10))
            f.flush()
            os.fsync(f.fileno())
