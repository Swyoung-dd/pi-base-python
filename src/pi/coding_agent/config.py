"""编码 agent 的配置。

从 .piy/ 目录、环境变量和默认值解析配置。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from pi.ai.models import load_model_file


@dataclass
class Config:
    """编码 agent 配置。"""

    model: str = "gpt-4o-mini"
    provider: str | None = None
    system_prompt: str = ""
    max_tokens: int | None = None
    temperature: float | None = None
    thinking_level: str = "off"
    extension_paths: list[Path] = field(default_factory=list)
    enable_entrypoint_extensions: bool = False
    skill_paths: list[Path] = field(default_factory=list)
    enable_skills: bool = True
    enable_context_files: bool = True
    is_configured: bool = False
    config_dir: Path = field(default_factory=lambda: Path.cwd() / ".piy")
    sessions_dir: Path = field(default_factory=lambda: Path.cwd() / ".piy" / "sessions")


def load_config(
    config_dir: Path | None = None,
    project_trusted: bool = True,
) -> Config:
    """从 .piy/config.yaml 加载配置，回退到默认值。"""
    if config_dir is None:
        config_dir = Path.cwd() / ".piy"

    config = Config(config_dir=config_dir, sessions_dir=config_dir / "sessions")

    config_file = config_dir / "config.yaml"
    if config_file.exists():
        config.is_configured = True
        with open(config_file, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if "model" in data:
            config.model = data["model"]
        if "provider" in data:
            config.provider = data["provider"]
        if project_trusted and "system_prompt" in data:
            config.system_prompt = data["system_prompt"]
        if "max_tokens" in data:
            config.max_tokens = data["max_tokens"]
        if "temperature" in data:
            config.temperature = data["temperature"]
        if "thinking_level" in data:
            config.thinking_level = data["thinking_level"]
        if project_trusted and "extensions" in data:
            config.extension_paths = [
                (config_dir / path).resolve() if not Path(path).is_absolute() else Path(path)
                for path in data["extensions"]
            ]
        if project_trusted and "enable_entrypoint_extensions" in data:
            config.enable_entrypoint_extensions = bool(data["enable_entrypoint_extensions"])
        if project_trusted and "skills" in data:
            config.skill_paths = [
                (config_dir / path).resolve() if not Path(path).is_absolute() else Path(path)
                for path in data["skills"]
            ]
        if project_trusted and "enable_skills" in data:
            config.enable_skills = bool(data["enable_skills"])
        if "enable_context_files" in data:
            config.enable_context_files = bool(data["enable_context_files"])

    models_file = config_dir / "models.yaml"
    if project_trusted and models_file.exists():
        load_model_file(models_file)

    # 环境变量覆盖
    if env_model := os.environ.get("PIY_MODEL"):
        config.model = env_model
    if env_provider := os.environ.get("PIY_PROVIDER"):
        config.provider = env_provider
    if env_thinking := os.environ.get("PIY_THINKING"):
        config.thinking_level = env_thinking

    config.sessions_dir.mkdir(parents=True, exist_ok=True)
    return config


def save_config(config: Config) -> Path:
    """原子写入用户可编辑的核心配置。"""
    config.config_dir.mkdir(parents=True, exist_ok=True)
    path = config.config_dir / "config.yaml"
    temporary = path.with_suffix(".yaml.tmp")
    data = {}
    if path.is_file():
        existing = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(existing, dict):
            data.update(existing)
    data.update(
        {
            "model": config.model,
            "provider": config.provider,
            "thinking_level": config.thinking_level,
        }
    )
    if config.max_tokens is not None:
        data["max_tokens"] = config.max_tokens
    if config.temperature is not None:
        data["temperature"] = config.temperature
    temporary.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    temporary.replace(path)
    config.is_configured = True
    return path
