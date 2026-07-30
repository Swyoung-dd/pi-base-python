"""JSONL RPC 命令行端到端测试。"""

import json
import os
import subprocess
import sys
from pathlib import Path


def test_rpc_cli_accepts_prompt_and_emits_jsonl(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config_dir = tmp_path / ".piy"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        """model: fake-model
provider: fake
extensions:
  - fake_provider.py
""",
        encoding="utf-8",
    )
    (config_dir / "models.yaml").write_text(
        """models:
  - id: fake-model
    name: Fake Model
    api: fake
    provider: fake
    context_window: 8192
    max_tokens: 1024
""",
        encoding="utf-8",
    )
    (config_dir / "fake_provider.py").write_text(
        """from pi.ai.providers.base import BaseProvider
from pi.ai.streaming import DoneEvent, EventStream, TextDeltaEvent
from pi.ai.types import AssistantMessage, StopReason, TextContent

class FakeProvider(BaseProvider):
    @property
    def provider_id(self):
        return "fake"

    @property
    def requires_api_key(self):
        return False

    async def stream(self, model, context, options=None):
        stream = EventStream()
        message = AssistantMessage(
            content=[TextContent(text="rpc ok")],
            api=model.api,
            provider=model.provider,
            model=model.id,
            stop_reason=StopReason.STOP,
            timestamp=1,
        )
        await stream.push(TextDeltaEvent(delta="rpc ok"))
        await stream.push(DoneEvent(message=message))
        await stream.end(message)
        return stream

def setup(context):
    context.add_provider(FakeProvider())
""",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    requests = "\n".join(
        [
            json.dumps({"type": "prompt", "id": "one", "message": "hello"}),
            json.dumps({"type": "shutdown", "id": "stop"}),
            "",
        ]
    )

    result = subprocess.run(
        [sys.executable, "-m", "pi", "--rpc", "--approve"],
        cwd=tmp_path,
        env=environment,
        input=requests,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    output = [json.loads(line) for line in result.stdout.splitlines()]
    assert output[0]["type"] == "ready"
    assert any(item["type"] == "accepted" and item["id"] == "one" for item in output)
    assert any(
        item["type"] == "event" and item["id"] == "one" and item["event"]["type"] == "text_delta"
        for item in output
    )
    prompt_response = next(
        item for item in output if item["type"] == "response" and item["id"] == "one"
    )
    assert prompt_response["messages"][-1]["content"][0]["text"] == "rpc ok"
    assert output[-1] == {"type": "response", "id": "stop", "ok": True}
