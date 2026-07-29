"""提供商注册表。

将提供商 ID 映射到提供商实例，用于模型分发。
"""

from __future__ import annotations

from pi.ai.providers.base import BaseProvider

_providers: dict[str, BaseProvider] = {}


def register_provider(provider: BaseProvider) -> None:
    """注册一个提供商实例。"""
    _providers[provider.provider_id] = provider


def get_provider(provider_id: str) -> BaseProvider | None:
    """按 ID 查找已注册的提供商。"""
    return _providers.get(provider_id)


def list_providers() -> list[str]:
    """返回所有已注册的提供商 ID。"""
    return list(_providers.keys())


def _register_builtins() -> None:
    """首次导入时注册内置提供商。"""
    from pi.ai.providers.anthropic import AnthropicProvider
    from pi.ai.providers.openai import OpenAIProvider

    if "openai" not in _providers:
        register_provider(OpenAIProvider())
    if "anthropic" not in _providers:
        register_provider(AnthropicProvider())


_register_builtins()
