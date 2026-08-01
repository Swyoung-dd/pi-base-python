"""RPC 协议扩展测试。"""

from __future__ import annotations

from pi.agent.agent import Agent, AgentOptions
from pi.ai.streaming import DoneEvent, EventStream, TextDeltaEvent
from pi.ai.types import AssistantMessage, Model, StopReason, TextContent
from pi.coding_agent.config import Config
from pi.coding_agent.extensions import ExtensionContext
from pi.coding_agent.rpc import RPC_PROTOCOL_VERSION, RpcServer
from pi.coding_agent.sdk import CodingAgent
from pi.coding_agent.themes import Theme


def _model():
    return Model(id="fake-model", name="Fake", api="fake", provider="fake", context_window=8192)


async def _stream(model, context, options):
    stream = EventStream()
    message = AssistantMessage(
        content=[TextContent(text="ok")],
        api=model.api,
        provider=model.provider,
        model=model.id,
        stop_reason=StopReason.STOP,
        timestamp=1,
    )
    await stream.push(TextDeltaEvent(delta="ok"))
    await stream.push(DoneEvent(message=message))
    await stream.end(message)
    return stream


def _runtime(tmp_path):
    agent = Agent(AgentOptions(model=_model(), stream_fn=_stream))
    return CodingAgent(
        agent=agent,
        config=Config(
            model="fake-model",
            provider="fake",
            sessions_dir=tmp_path / ".piy" / "sessions",
        ),
        cwd=tmp_path,
        extensions=ExtensionContext(),
        skills=[],
        prompt_templates=[],
        theme=Theme(),
    )


async def test_rpc_protocol_version(tmp_path):
    sent = []

    async def send(payload):
        sent.append(payload)

    server = RpcServer(_runtime(tmp_path), send)
    await server.start()
    assert sent[0]["protocol_version"] == RPC_PROTOCOL_VERSION
    await server.handle({"type": "protocol_version", "id": "pv"})
    resp = next(s for s in sent if s.get("id") == "pv")
    assert resp["version"] == RPC_PROTOCOL_VERSION
    await server.close()


async def test_rpc_set_thinking_level(tmp_path):
    sent = []

    async def send(payload):
        sent.append(payload)

    server = RpcServer(_runtime(tmp_path), send)
    await server.start()
    await server.handle({"type": "set_thinking_level", "id": "tl", "level": "high"})
    resp = next(s for s in sent if s.get("id") == "tl")
    assert resp["ok"] is True
    assert server.runtime.agent.thinking_level.value == "high"
    await server.close()


async def test_rpc_list_models(tmp_path):
    sent = []

    async def send(payload):
        sent.append(payload)

    server = RpcServer(_runtime(tmp_path), send)
    await server.start()
    await server.handle({"type": "list_models", "id": "lm"})
    resp = next(s for s in sent if s.get("id") == "lm")
    assert "models" in resp
    assert isinstance(resp["models"], list)
    await server.close()


async def test_rpc_get_active_tools(tmp_path):
    sent = []

    async def send(payload):
        sent.append(payload)

    server = RpcServer(_runtime(tmp_path), send)
    await server.start()
    await server.handle({"type": "get_active_tools", "id": "tools"})
    resp = next(s for s in sent if s.get("id") == "tools")
    assert "tools" in resp
    assert isinstance(resp["tools"], list)
    assert "compact" in resp["tools"]
    await server.close()


async def test_rpc_list_sessions_empty(tmp_path):
    sent = []

    async def send(payload):
        sent.append(payload)

    server = RpcServer(_runtime(tmp_path), send)
    await server.start()
    await server.handle({"type": "list_sessions", "id": "ls"})
    resp = next(s for s in sent if s.get("id") == "ls")
    assert resp["sessions"] == []
    await server.close()


async def test_rpc_compact_when_idle(tmp_path):
    sent = []

    async def send(payload):
        sent.append(payload)

    server = RpcServer(_runtime(tmp_path), send)
    await server.start()
    await server.handle({"type": "compact", "id": "cmp", "target_tokens": 100})
    resp = next(s for s in sent if s.get("id") == "cmp")
    assert "compaction" in resp or "error" in resp
    await server.close()
