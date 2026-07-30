"""首次运行模型选择与配置写入。"""

from __future__ import annotations

import click

from pi.ai.models import list_models
from pi.ai.oauth import CredentialStore
from pi.ai.types import ModelThinkingLevel
from pi.coding_agent.config import Config, save_config
from pi.coding_agent.model_auth import ensure_model_auth
from pi.tui.selector import select_option


async def run_setup(
    config: Config,
    credential_store: CredentialStore | None = None,
) -> Config:
    """交互选择模型，按需收集凭据并保存默认配置。"""
    models = list_models()
    if not models:
        raise click.ClickException("No models are available")
    model = await select_option(
        "Select model",
        [
            (
                candidate,
                f"{candidate.provider}/{candidate.id}"
                f"{'  reasoning' if candidate.reasoning else ''}",
            )
            for candidate in models
        ],
        default=models[0],
    )
    if model is None:
        click.echo("Setup cancelled.")
        return config
    if not await ensure_model_auth(model, credential_store):
        return config
    config.model = model.id
    config.provider = model.provider
    if model.reasoning:
        levels = [level.value for level in ModelThinkingLevel]
        default_level = (
            config.thinking_level
            if config.thinking_level in levels and config.thinking_level != "off"
            else ModelThinkingLevel.MEDIUM.value
        )
        thinking_level = await select_option(
            "Thinking level",
            default=default_level,
            options=[(level, level) for level in levels],
        )
        if thinking_level is None:
            click.echo("Setup cancelled.")
            return config
        config.thinking_level = thinking_level
    else:
        config.thinking_level = ModelThinkingLevel.OFF.value
    save_config(config)
    click.echo(f"Configured: {model.provider}/{model.id}")
    return config
