"""首次运行配置与模型切换测试。"""

from pi.agent.agent import Agent, AgentOptions
from pi.ai.models import list_models
from pi.ai.types import Model
from pi.coding_agent.config import Config, load_config
from pi.coding_agent.setup import run_setup


def test_setup_selects_and_persists_model(tmp_path, monkeypatch):
    config = Config(
        config_dir=tmp_path,
        sessions_dir=tmp_path / "sessions",
    )
    monkeypatch.setattr("pi.coding_agent.setup.click.prompt", lambda *args, **kwargs: 1)
    run_setup(config)

    selected = list_models()[0]
    restored = load_config(tmp_path)
    assert restored.is_configured
    assert (restored.provider, restored.model) == (selected.provider, selected.id)


def test_agent_model_switch_updates_context_limit():
    initial = Model(
        id="initial",
        name="Initial",
        api="test",
        provider="test",
        context_window=10_000,
        max_tokens=1_000,
    )
    replacement = Model(
        id="replacement",
        name="Replacement",
        api="test",
        provider="test",
        context_window=20_000,
        max_tokens=2_000,
    )
    agent = Agent(AgentOptions(model=initial))

    agent.set_model(replacement)

    assert agent.state.model is replacement
    assert agent.context_token_limit == 18_000
