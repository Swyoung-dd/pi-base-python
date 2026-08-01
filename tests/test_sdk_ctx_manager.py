"""CodingAgent async context manager tests."""

from __future__ import annotations

from pi.coding_agent.sdk import CodingAgent, create_coding_agent


async def test_coding_agent_async_context_manager(tmp_path):
    async with await create_coding_agent(cwd=tmp_path) as agent:
        assert isinstance(agent, CodingAgent)
        assert agent._started is True
    assert agent._started is False


async def test_coding_agent_close_closes_tracer(tmp_path):
    agent = await create_coding_agent(cwd=tmp_path)
    await agent.start()
    agent.tracer._file = None  # ensure no file
    await agent.close()
    # close should be idempotent
    await agent.close()
