"""首次运行模型选择与配置写入。"""

from __future__ import annotations

import click

from pi.ai.models import list_models
from pi.coding_agent.config import Config, save_config


def run_setup(config: Config) -> Config:
    """交互选择模型并保存，不收集或写入 API 密钥。"""
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
    config.model = model.id
    config.provider = model.provider
    save_config(config)
    click.echo(f"Configured: {model.provider}/{model.id}")
    return config
