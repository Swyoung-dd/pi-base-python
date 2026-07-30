"""首次运行配置与模型切换测试。"""

from pi.agent.agent import Agent, AgentOptions
from pi.ai.models import list_models
from pi.ai.oauth import CredentialStore, resolve_stored_api_key
from pi.ai.types import Model
from pi.coding_agent.config import Config, load_config, save_config
from pi.coding_agent.setup import run_setup


async def test_setup_selects_model_api_key_and_persists_both(tmp_path, monkeypatch):
    config = Config(
        config_dir=tmp_path,
        sessions_dir=tmp_path / "sessions",
    )
    models = list_models()
    selected = next(model for model in models if model.provider == "deepseek")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("PIY_API_KEY_DEEPSEEK", raising=False)

    async def choose(title, options, **kwargs):
        return selected if title == "Select model" else "high"

    monkeypatch.setattr("pi.coding_agent.setup.select_option", choose)

    async def prompt_api_key(message):
        return "setup-key"

    monkeypatch.setattr("pi.coding_agent.model_auth._prompt_api_key", prompt_api_key)
    store = CredentialStore(tmp_path / "auth.json")
    await run_setup(config, store)

    restored = load_config(tmp_path)
    assert restored.is_configured
    assert (restored.provider, restored.model) == (selected.provider, selected.id)
    assert restored.thinking_level == "high"
    assert await resolve_stored_api_key(selected.provider, store) == "setup-key"


def test_default_config_uses_piy_directory_and_environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PIY_MODEL", "configured-model")
    monkeypatch.setenv("PIY_PROVIDER", "configured-provider")
    monkeypatch.setenv("PIY_THINKING", "high")

    config = load_config()

    assert config.config_dir == tmp_path / ".piy"
    assert config.sessions_dir == tmp_path / ".piy" / "sessions"
    assert config.model == "configured-model"
    assert config.provider == "configured-provider"
    assert config.thinking_level == "high"
    assert config.enable_context_files


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


def test_save_config_preserves_resource_settings(tmp_path):
    config_dir = tmp_path / ".piy"
    config_dir.mkdir()
    path = config_dir / "config.yaml"
    path.write_text(
        """extensions:
  - extension.py
enable_skills: false
""",
        encoding="utf-8",
    )
    config = Config(
        model="new-model",
        provider="new-provider",
        thinking_level="high",
        max_tokens=2048,
        temperature=0.3,
        config_dir=config_dir,
        sessions_dir=config_dir / "sessions",
    )

    save_config(config)
    restored = load_config(config_dir)

    assert restored.extension_paths == [(config_dir / "extension.py").resolve()]
    assert not restored.enable_skills
    assert restored.max_tokens == 2048
    assert restored.temperature == 0.3
