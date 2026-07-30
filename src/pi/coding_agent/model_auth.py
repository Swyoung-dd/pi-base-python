"""模型选择时的 provider 凭据检查与录入。"""

from __future__ import annotations

import click

from pi.ai.auth import get_provider_token
from pi.ai.oauth import CredentialStore, save_api_key
from pi.ai.providers.registry import get_provider
from pi.ai.types import Model


async def ensure_model_auth(
    model: Model,
    store: CredentialStore | None = None,
) -> bool:
    """确保模型 provider 有可用凭据，缺失时隐藏输入并保存 API Key。"""
    provider = get_provider(model.provider)
    if provider is None:
        raise click.ClickException(f"No provider registered for: {model.provider}")
    if not provider.requires_api_key:
        return True
    if await get_provider_token(model.provider, store):
        return True

    try:
        api_key = click.prompt(
            f"{model.provider} API key",
            hide_input=True,
            confirmation_prompt=False,
        )
    except (click.Abort, EOFError):
        click.echo("Model selection cancelled: API key is required")
        return False

    await save_api_key(model.provider, api_key, store)
    click.echo(f"API key saved for: {model.provider}")
    return True
