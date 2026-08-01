"""Memory session storage tests."""

from __future__ import annotations

from pi.agent.session.memory import MemoryStorage
from pi.agent.types import AgentUserMessage


async def test_memory_append_and_get():
    storage = MemoryStorage()
    entry_id = await storage.append_message(AgentUserMessage(content="hello"))
    entry = await storage.get(entry_id)
    assert entry is not None
    assert entry.message is not None


async def test_memory_get_entries():
    storage = MemoryStorage()
    await storage.append_message(AgentUserMessage(content="msg1"))
    await storage.append_message(AgentUserMessage(content="msg2"))
    entries = await storage.get_entries()
    assert len(entries) == 2


async def test_memory_get_branch():
    storage = MemoryStorage()
    await storage.append_message(AgentUserMessage(content="msg1"))
    await storage.append_message(AgentUserMessage(content="msg2"))
    branch = await storage.get_branch()
    assert len(branch) == 2


async def test_memory_leaf_id():
    storage = MemoryStorage()
    assert await storage.get_leaf_id() is None
    entry_id = await storage.append_message(AgentUserMessage(content="hello"))
    assert await storage.get_leaf_id() == entry_id


async def test_memory_branch_from():
    storage = MemoryStorage()
    entry_id = await storage.append_message(AgentUserMessage(content="msg1"))
    branch_id = await storage.branch_from(entry_id)
    assert branch_id != entry_id
    branch = await storage.get_branch()
    assert len(branch) == 2
    assert branch[0].id == entry_id
    assert branch[1].id == branch_id
    assert branch[1].parent_id == entry_id


async def test_memory_context_messages():
    storage = MemoryStorage()
    await storage.append_message(AgentUserMessage(content="msg1"))
    await storage.append_message(AgentUserMessage(content="msg2"))
    messages = await storage.get_context_messages()
    assert len(messages) == 2


async def test_memory_get_nonexistent():
    storage = MemoryStorage()
    result = await storage.get("nonexistent")
    assert result is None
    branch = await storage.get_branch()
    assert branch == []
