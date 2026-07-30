"""从项目配置到 JSON 结果的 CLI 集成测试。"""

import json
from pathlib import Path

from click.testing import CliRunner

from pi.ai.models import clear_custom_models
from pi.coding_agent.cli import main


def test_print_mode_with_project_model_and_extension_provider():
    runner = CliRunner()
    with runner.isolated_filesystem():
        config_dir = Path(".piy")
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

    async def stream(self, model, context, options=None):
        stream = EventStream()
        message = AssistantMessage(
            content=[TextContent(text="integration ok")],
            api=model.api,
            provider=model.provider,
            model=model.id,
            stop_reason=StopReason.STOP,
            timestamp=1,
        )
        await stream.push(TextDeltaEvent(delta="integration ok"))
        await stream.push(DoneEvent(message=message))
        await stream.end(message)
        return stream

def setup(context):
    context.add_provider(FakeProvider())
""",
            encoding="utf-8",
        )

        try:
            result = runner.invoke(
                main,
                ["-p", "hello", "--output", "json", "--approve"],
            )
        finally:
            clear_custom_models()

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["type"] == "agent_result"
    assert payload["messages"][-1]["content"][0]["text"] == "integration ok"
