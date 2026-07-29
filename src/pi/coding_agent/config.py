"""编码 agent 的配置。

从 .pi/ 目录、环境变量和默认值解析配置。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Config:
    """编码 agent 配置。"""
    model: str = "gpt-4o-mini"
    provider: str | None = None
    system_prompt: str = ""
    max_tokens: int | None = None
    temperature: float | None = None
    config_dir: Path = field(default_factory=lambda: Path.cwd() / ".pi")
    sessions_dir: Path = field(default_factory=lambda: Path.cwd() / ".pi" / "sessions")


def load_config(config_dir: Path | None = None) -> Config:
    """从 .pi/config.yaml 加载配置，回退到默认值。"""
    if config_dir is None:
        config_dir = Path.cwd() / ".pi"

    config = Config(config_dir=config_dir, sessions_dir=config_dir / "sessions")

    config_file = config_dir / "config.yaml"
    if config_file.exists():
        with open(config_file, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if "model" in data:
            config.model = data["model"]
        if "provider" in data:
            config.provider = data["provider"]
        if "system_prompt" in data:
            config.system_prompt = data["system_prompt"]
        if "max_tokens" in data:
            config.max_tokens = data["max_tokens"]
        if "temperature" in data:
            config.temperature = data["temperature"]

    # 环境变量覆盖
    if env_model := os.environ.get("PI_MODEL"):
        config.model = env_model
    if env_provider := os.environ.get("PI_PROVIDER"):
        config.provider = env_provider

    config.sessions_dir.mkdir(parents=True, exist_ok=True)
    return config
