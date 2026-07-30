"""模型选择时的 provider 凭据检查与录入。"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable

import click
from prompt_toolkit import PromptSession
from prompt_toolkit.history import DummyHistory

from pi.ai.auth import get_provider_token
from pi.ai.oauth import CredentialStore, save_api_key
from pi.ai.providers.registry import get_provider
from pi.ai.types import Model

_API_KEY_PATTERN = re.compile(r"(?:^|\s)sk-[A-Za-z0-9_-]{20,}(?:$|\s)")
ApiKeyPrompt = Callable[[str], Awaitable[str]]


def contains_likely_api_key(value: str) -> bool:
    """识别不应作为普通提示发送的常见 API Key。"""
    return _API_KEY_PATTERN.search(value) is not None


async def _prompt_api_key(message: str) -> str:
    """使用带掩码、无历史记录的异步密码输入框。"""
    session: PromptSession[str] = PromptSession(
        history=DummyHistory(),
        is_password=True,
    )
    return await session.prompt_async(message)


async def ensure_model_auth(
    model: Model,
    store: CredentialStore | None = None,
    api_key_prompt: ApiKeyPrompt | None = None,
) -> bool:
    """确保模型 provider 有可用凭据，并允许隐藏录入或替换 API Key。"""
    provider = get_provider(model.provider)
    if provider is None:
        raise click.ClickException(f"No provider registered for: {model.provider}")
    if not provider.requires_api_key:
        return True
    existing_token = await get_provider_token(model.provider, store)
    prompt_api_key = api_key_prompt or _prompt_api_key

    try:
        if existing_token:
            api_key = await prompt_api_key(
                f"{model.provider} API key (press Enter to keep current): "
            )
            if not api_key.strip():
                return True
        else:
            api_key = await prompt_api_key(f"{model.provider} API key: ")
    except (click.Abort, EOFError, KeyboardInterrupt):
        click.echo("Model selection cancelled: API key is required")
        return False

    await save_api_key(model.provider, api_key, store)
    click.echo(f"API key saved for: {model.provider}")
    return True
