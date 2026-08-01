"""Compact tool tests."""

from __future__ import annotations

from pi.agent.types import AgentToolCall, ContextCompactionRequest
from pi.coding_agent.tools.compact import create_compact_tool


async def test_compact_tool_basic():
    tool = create_compact_tool()
    result = await tool.execute(AgentToolCall(id="c1", name="compact", arguments={}), None)
    assert result.tool_name == "compact"
    assert not result.is_error
    assert isinstance(result.details, ContextCompactionRequest)


async def test_compact_tool_with_target_tokens():
    tool = create_compact_tool()
    result = await tool.execute(
        AgentToolCall(id="c2", name="compact", arguments={"target_tokens": 8000}),
        None,
    )
    assert result.details.target_tokens == 8000


async def test_compact_tool_registered_in_runtime():
    from pi.coding_agent.runtime import build_tools

    tools = build_tools()
    names = [t.name for t in tools]
    assert "compact" in names
    assert "subagent" in names
    assert "read" in names
