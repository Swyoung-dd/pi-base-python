"""程序化 SDK 与 JSONL RPC 协议测试。"""

from pi.agent.agent import Agent, AgentOptions
from pi.ai.models import list_models
from pi.ai.streaming import DoneEvent, EventStream, TextDeltaEvent
from pi.ai.types import AssistantMessage, Model, StopReason, TextContent
from pi.coding_agent.config import Config
from pi.coding_agent.extensions import ExtensionContext
from pi.coding_agent.rpc import RpcServer
from pi.coding_agent.sdk import CodingAgent, create_coding_agent
from pi.coding_agent.themes import Theme


def _model():
    return Model(
        id="fake-model",
        name="Fake",
        api="fake",
        provider="fake",
        context_window=8192,
    )


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
        config=Config(model="fake-model", provider="fake"),
        cwd=tmp_path,
        extensions=ExtensionContext(),
        skills=[],
        prompt_templates=[],
        theme=Theme(),
    )


async def test_coding_agent_sdk_returns_new_messages(tmp_path):
    runtime = _runtime(tmp_path)

    messages = await runtime.prompt("hello")
    await runtime.close()

    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[-1].content[0].text == "ok"


async def test_create_coding_agent_builds_default_runtime(tmp_path, monkeypatch):
    model = list_models()[0]
    config = Config(
        model=model.id,
        provider=model.provider,
        config_dir=tmp_path / ".piy",
        sessions_dir=tmp_path / ".piy" / "sessions",
    )
    context = ExtensionContext()

    async def build_resources(config, cwd):
        return [], "system", context, [], [], Theme()

    monkeypatch.setattr(
        "pi.coding_agent.sdk.build_runtime_resources",
        build_resources,
    )
    monkeypatch.setattr("pi.coding_agent.sdk.make_stream_fn", lambda: _stream)

    runtime = await create_coding_agent(config=config, cwd=tmp_path)
    messages = await runtime.prompt("hello")
    await runtime.close()

    assert runtime.cwd == tmp_path.resolve()
    assert runtime.agent.state.system_prompt == "system"
    assert messages[-1].content[0].text == "ok"


async def test_rpc_server_streams_events_and_returns_state(tmp_path):
    sent = []

    async def send(payload):
        sent.append(payload)

    server = RpcServer(_runtime(tmp_path), send)
    await server.start()
    await server.handle({"type": "prompt", "id": "one", "message": "hello"})
    await server.wait()
    await server.handle({"type": "get_state", "id": "state"})
    await server.close()

    assert sent[0]["type"] == "ready"
    assert any(item["type"] == "event" and item["id"] == "one" for item in sent)
    response = next(item for item in sent if item.get("id") == "one" and item["type"] == "response")
    assert response["messages"][-1]["content"][0]["text"] == "ok"
    state = next(item for item in sent if item.get("id") == "state")
    assert state["state"]["busy"] is False
    assert state["state"]["model"] == {"provider": "fake", "id": "fake-model"}
