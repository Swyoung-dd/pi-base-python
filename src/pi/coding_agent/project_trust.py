"""项目本地资源的信任判断与持久化。"""

from __future__ import annotations

import json
import os
from contextlib import suppress
from pathlib import Path

import click
import yaml

_PROTECTED_CONFIG_KEYS = {
    "system_prompt",
    "extensions",
    "enable_entrypoint_extensions",
    "skills",
    "enable_skills",
    "prompts",
    "enable_prompt_templates",
}


def _canonical(path: Path) -> Path:
    return path.resolve()


def has_protected_project_resources(config_dir: Path) -> bool:
    """判断项目配置目录是否包含会执行或注入内容的资源。"""
    if (config_dir / "models.yaml").is_file():
        return True
    skills_dir = config_dir / "skills"
    if skills_dir.is_dir() and any(skills_dir.iterdir()):
        return True
    prompts_dir = config_dir / "prompts"
    if prompts_dir.is_dir() and any(prompts_dir.iterdir()):
        return True
    config_file = config_dir / "config.yaml"
    if not config_file.is_file():
        return False
    try:
        value = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return True
    return isinstance(value, dict) and any(key in value for key in _PROTECTED_CONFIG_KEYS)


class ProjectTrustStore:
    """按规范化项目路径保存信任决定。"""

    def __init__(self, path: Path | None = None) -> None:
        default_path = Path.home() / ".piy" / "trust.json"
        self.path = path or Path(os.environ.get("PIY_TRUST_FILE", default_path))

    def _read(self) -> dict[str, bool]:
        if not self.path.is_file():
            return {}
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        paths = value.get("paths", {}) if isinstance(value, dict) else {}
        return {
            str(path): decision
            for path, decision in paths.items()
            if isinstance(path, str) and isinstance(decision, bool)
        }

    def get(self, project_dir: Path) -> bool | None:
        """返回当前目录或最近父目录的信任决定。"""
        decisions = self._read()
        current = _canonical(project_dir)
        for candidate in (current, *current.parents):
            key = os.path.normcase(str(candidate))
            if key in decisions:
                return decisions[key]
        return None

    def set(self, project_dir: Path, trusted: bool) -> None:
        decisions = self._read()
        decisions[os.path.normcase(str(_canonical(project_dir)))] = trusted
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps({"paths": decisions}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with suppress(OSError):
            temporary.chmod(0o600)
        os.replace(temporary, self.path)


def resolve_project_trust(
    project_dir: Path,
    config_dir: Path,
    override: bool | None,
    interactive: bool,
    store: ProjectTrustStore | None = None,
) -> bool:
    """解析项目资源信任状态；非交互模式在无决定时默认拒绝。"""
    if not has_protected_project_resources(config_dir):
        return True
    trust_store = store or ProjectTrustStore()
    if override is not None:
        return override
    saved = trust_store.get(project_dir)
    if saved is not None:
        return saved
    if not interactive:
        return False
    trusted = click.confirm(
        f"Trust project resources in {project_dir}?",
        default=False,
    )
    trust_store.set(project_dir, trusted)
    return trusted
