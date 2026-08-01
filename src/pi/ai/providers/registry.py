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
    return sorted(_providers)


def register_openai_compatible(
    provider_id: str,
    requires_api_key: bool = True,
) -> BaseProvider:
    """注册使用 OpenAI Chat Completions 协议的提供商。"""
    from pi.ai.providers.openai import OpenAIProvider

    provider = OpenAIProvider(provider_id, requires_api_key)
    register_provider(provider)
    return provider


def _register_builtins() -> None:
    """首次导入时注册内置提供商。"""
    from pi.ai.providers.anthropic import AnthropicProvider
    from pi.ai.providers.deepseek import DeepSeekProvider
    from pi.ai.providers.google import GoogleProvider
    from pi.ai.providers.kimi import KimiProvider
    from pi.ai.providers.openai import OpenAIProvider
    from pi.ai.providers.qwen import QwenProvider
    from pi.ai.providers.zai import ZAIProvider

    if "openai" not in _providers:
        register_provider(OpenAIProvider())
    if "anthropic" not in _providers:
        register_provider(AnthropicProvider())
    if "deepseek" not in _providers:
        register_provider(DeepSeekProvider())
    if "google" not in _providers:
        register_provider(GoogleProvider())
    if "qwen" not in _providers:
        register_provider(QwenProvider())
    if "zai" not in _providers:
        register_provider(ZAIProvider())
    if "kimi" not in _providers:
        register_provider(KimiProvider())
    for provider_id in ("groq", "mistral", "openrouter", "xai"):
        if provider_id not in _providers:
            register_openai_compatible(provider_id)
    for provider_id in ("lmstudio", "ollama"):
        if provider_id not in _providers:
            register_openai_compatible(provider_id, requires_api_key=False)


_register_builtins()
