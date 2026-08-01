"""SQLite session storage backend."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from pi.agent.session.base import (
    SessionEntry,
    SessionStorage,
)


class SqliteStorage(SessionStorage):
    """Persistent session storage backed by SQLite."""

    def __init__(self, path: str | Path, cwd: str | None = None) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._leaf_id: str | None = None
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()
        self._load_leaf()

    def _init_db(self) -> None:
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS entries (
                    id TEXT PRIMARY KEY,
                    parent_id TEXT,
                    type TEXT NOT NULL DEFAULT "message",
                    message TEXT,
                    data TEXT,
                    timestamp TEXT,
                    seq INTEGER AUTOINCREMENT
                )
            """)
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_parent ON entries(parent_id)")
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            self._conn.commit()

    def _load_leaf(self) -> None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM meta WHERE key = ?", ("leaf_id",)
            ).fetchone()
            if row:
                self._leaf_id = row["value"]

    async def append(self, entry: SessionEntry) -> str:
        d = entry.to_dict()
        with self._lock:
            self._conn.execute(
                (
                    "INSERT INTO entries "
                    "(id, parent_id, type, message, data, timestamp) "
                    "VALUES (?, ?, ?, ?, ?, ?)"
                ),
                (d["id"], d.get("parent_id"), d.get("type", "message"),
                 json.dumps(d.get("message"), ensure_ascii=False) if d.get("message") else None,
                 json.dumps(d.get("data"), ensure_ascii=False) if d.get("data") else None,
                 d.get("timestamp", "")),
            )
            self._conn.commit()
        return entry.id

    async def get(self, entry_id: str) -> SessionEntry | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM entries WHERE id = ?", (entry_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_entry(row)

    async def get_entries(self) -> list[SessionEntry]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM entries ORDER BY rowid").fetchall()
        return [self._row_to_entry(r) for r in rows]

    async def get_branch(self) -> list[SessionEntry]:
        if self._leaf_id is None:
            return []
        chain: list[SessionEntry] = []
        current_id = self._leaf_id
        while current_id:
            with self._lock:
                row = self._conn.execute(
                    "SELECT * FROM entries WHERE id = ?", (current_id,)
                ).fetchone()
            if row is None:
                break
            entry = self._row_to_entry(row)
            chain.append(entry)
            current_id = entry.parent_id
        chain.reverse()
        return chain

    async def get_leaf_id(self) -> str | None:
        return self._leaf_id

    async def set_leaf_id(self, entry_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                ("leaf_id", entry_id),
            )
            self._conn.commit()
        self._leaf_id = entry_id

    def _row_to_entry(self, row: sqlite3.Row) -> SessionEntry:
        data = {}
        if row["message"]:
            data["message"] = json.loads(row["message"])
        if row["data"]:
            data["data"] = json.loads(row["data"])
        data["id"] = row["id"]
        data["parent_id"] = row["parent_id"]
        data["type"] = row["type"] or "message"
        data["timestamp"] = row["timestamp"] or ""
        return SessionEntry.from_dict(data)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
