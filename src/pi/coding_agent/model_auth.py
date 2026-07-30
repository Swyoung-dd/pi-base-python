"""模型选择时的 provider 凭据检查与录入。"""

from __future__ import annotations

import re

import click

from pi.ai.auth import get_provider_token
from pi.ai.oauth import CredentialStore, save_api_key
from pi.ai.providers.registry import get_provider
from pi.ai.types import Model

_API_KEY_PATTERN = re.compile(r"(?:^|\s)sk-[A-Za-z0-9_-]{20,}(?:$|\s)")


def contains_likely_api_key(value: str) -> bool:
    """识别不应作为普通提示发送的常见 API Key。"""
    return _API_KEY_PATTERN.search(value) is not None


async def ensure_model_auth(
    model: Model,
    store: CredentialStore | None = None,
) -> bool:
    """确保模型 provider 有可用凭据，并允许隐藏录入或替换 API Key。"""
    provider = get_provider(model.provider)
    if provider is None:
        raise click.ClickException(f"No provider registered for: {model.provider}")
    if not provider.requires_api_key:
        return True
    existing_token = await get_provider_token(model.provider, store)

    try:
        if existing_token:
            api_key = click.prompt(
                f"{model.provider} API key (press Enter to keep current)",
                default="",
                show_default=False,
                hide_input=True,
                confirmation_prompt=False,
            )
            if not api_key.strip():
                return True
        else:
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
