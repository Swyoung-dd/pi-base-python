"""SQLite session storage tests."""

from __future__ import annotations

import pytest

from pi.agent.session.sqlite import SqliteStorage
from pi.agent.types import AgentUserMessage


@pytest.fixture
def storage(tmp_path):
    s = SqliteStorage(tmp_path / "test.sqlite")
    yield s
    s.close()


async def test_sqlite_append_and_get(storage):
    msg = AgentUserMessage(content="hello")
    entry_id = await storage.append_message(msg)
    assert entry_id is not None
    entry = await storage.get(entry_id)
    assert entry is not None
    assert entry.message is not None


async def test_sqlite_get_entries(storage):
    await storage.append_message(AgentUserMessage(content="msg1"))
    await storage.append_message(AgentUserMessage(content="msg2"))
    entries = await storage.get_entries()
    assert len(entries) == 2


async def test_sqlite_get_branch(storage):
    await storage.append_message(AgentUserMessage(content="msg1"))
    await storage.append_message(AgentUserMessage(content="msg2"))
    branch = await storage.get_branch()
    assert len(branch) == 2


async def test_sqlite_leaf_id(storage):
    assert await storage.get_leaf_id() is None
    entry_id = await storage.append_message(AgentUserMessage(content="hello"))
    assert await storage.get_leaf_id() == entry_id


async def test_sqlite_branch_from(storage):
    entry_id = await storage.append_message(AgentUserMessage(content="msg1"))
    branch_id = await storage.branch_from(entry_id)
    assert branch_id != entry_id
    branch = await storage.get_branch()
    assert len(branch) == 2


async def test_sqlite_model_selection(storage):
    await storage.append_model_change("openai", "gpt-4o")
    selection = await storage.get_model_selection()
    assert selection == ("openai", "gpt-4o")


async def test_sqlite_context_messages(storage):
    await storage.append_message(AgentUserMessage(content="msg1"))
    await storage.append_message(AgentUserMessage(content="msg2"))
    messages = await storage.get_context_messages()
    assert len(messages) == 2


async def test_sqlite_get_nonexistent(storage):
    result = await storage.get("nonexistent-id")
    assert result is None
