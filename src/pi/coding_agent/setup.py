"""首次运行模型选择与配置写入。"""

from __future__ import annotations

import click

from pi.ai.models import list_models
from pi.ai.oauth import CredentialStore
from pi.coding_agent.config import Config, save_config
from pi.coding_agent.model_auth import ensure_model_auth


async def run_setup(
    config: Config,
    credential_store: CredentialStore | None = None,
) -> Config:
    """交互选择模型，按需收集凭据并保存默认配置。"""
    models = list_models()
    if not models:
        raise click.ClickException("No models are available")
    click.echo("Available models:")
    for index, model in enumerate(models, start=1):
        reasoning = " reasoning" if model.reasoning else ""
        click.echo(f"  {index:>2}. {model.provider}/{model.id}{reasoning}")
    selected = click.prompt(
        "Select model",
        type=click.IntRange(1, len(models)),
        default=1,
    )
    model = models[selected - 1]
    if not await ensure_model_auth(model, credential_store):
        return config
    config.model = model.id
    config.provider = model.provider
    save_config(config)
    click.echo(f"Configured: {model.provider}/{model.id}")
    return config
