"""API 密钥解析。

按提供商特定的命名约定，从环境变量解析提供商 API 密钥。
"""

from __future__ import annotations

import os

_ENV_KEY_MAP: dict[str, list[str]] = {
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "google": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    "mistral": ["MISTRAL_API_KEY"],
    "deepseek": ["DEEPSEEK_API_KEY"],
    "xai": ["XAI_API_KEY"],
    "groq": ["GROQ_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
}


def get_api_key(provider: str) -> str | None:
    """从环境变量解析提供商的 API 密钥。

    先检查提供商特定的环境变量，再回退到通用的
    PIY_API_KEY_{PROVIDER} 模式。
    """
    env_keys = _ENV_KEY_MAP.get(provider, [])
    for key in env_keys:
        val = os.environ.get(key)
        if val:
            return val
    generic = os.environ.get(f"PIY_API_KEY_{provider.upper()}")
    if generic:
        return generic
    return None


async def get_provider_token(provider: str) -> str | None:
    """优先解析并刷新 OAuth token，再回退到环境变量 API key。"""
    from pi.ai.oauth import resolve_oauth_access_token
    from pi.ai.oauth_xai import register_xai_oauth

    register_xai_oauth()
    oauth_token = await resolve_oauth_access_token(provider)
    return oauth_token or get_api_key(provider)
