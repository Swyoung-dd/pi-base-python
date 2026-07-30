"""项目本地资源信任测试。"""

from pathlib import Path

from pi.coding_agent.config import load_config
from pi.coding_agent.project_trust import (
    ProjectTrustStore,
    has_protected_project_resources,
    resolve_project_trust,
)


def test_untrusted_config_ignores_executable_resources(tmp_path):
    config_dir = tmp_path / ".piy"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        """model: gpt-4o-mini
provider: openai
system_prompt: injected
extensions:
  - extension.py
skills:
  - skills
prompts:
  - prompts
""",
        encoding="utf-8",
    )

    config = load_config(config_dir, project_trusted=False)

    assert config.model == "gpt-4o-mini"
    assert config.provider == "openai"
    assert config.system_prompt == ""
    assert config.extension_paths == []
    assert config.skill_paths == []
    assert config.prompt_paths == []


def test_trust_store_inherits_nearest_parent_decision(tmp_path):
    store = ProjectTrustStore(tmp_path / "trust.json")
    parent = tmp_path / "workspace"
    child = parent / "project"
    child.mkdir(parents=True)

    store.set(parent, True)

    assert store.get(child) is True


def test_noninteractive_project_resources_default_to_untrusted(tmp_path):
    config_dir = tmp_path / ".piy"
    config_dir.mkdir()
    (config_dir / "models.yaml").write_text("models: []\n", encoding="utf-8")
    store = ProjectTrustStore(tmp_path / "trust.json")

    trusted = resolve_project_trust(
        tmp_path,
        config_dir,
        override=None,
        interactive=False,
        store=store,
    )

    assert has_protected_project_resources(config_dir)
    assert trusted is False


def test_explicit_project_trust_does_not_persist(tmp_path):
    config_dir = tmp_path / ".piy"
    config_dir.mkdir()
    (config_dir / "models.yaml").write_text("models: []\n", encoding="utf-8")
    store = ProjectTrustStore(tmp_path / "trust.json")

    trusted = resolve_project_trust(
        tmp_path,
        config_dir,
        override=True,
        interactive=False,
        store=store,
    )

    assert trusted is True
    assert store.get(Path(tmp_path)) is None
