"""API 密钥解析。

按提供商特定的命名约定，从环境变量解析提供商 API 密钥。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from pi.ai.oauth import CredentialStore, resolve_oauth_access_token, resolve_stored_api_key
from pi.ai.oauth_xai import register_xai_oauth

_ENV_KEY_MAP: dict[str, list[str]] = {
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "google": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    "mistral": ["MISTRAL_API_KEY"],
    "deepseek": ["DEEPSEEK_API_KEY"],
    "xai": ["XAI_API_KEY"],
    "groq": ["GROQ_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
    "qwen": ["DASHSCOPE_API_KEY", "QWEN_API_KEY"],
    "zai": ["ZAI_API_KEY"],
    "kimi": ["MOONSHOT_API_KEY", "KIMI_API_KEY"],
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


async def get_provider_token(
    provider: str,
    store: CredentialStore | None = None,
) -> str | None:
    """优先解析保存凭据，再回退到环境变量 API key。"""
    register_xai_oauth()
    stored_api_key = await resolve_stored_api_key(provider, store)
    if stored_api_key:
        return stored_api_key
    oauth_token = await resolve_oauth_access_token(provider, store)
    return oauth_token or get_api_key(provider)


@dataclass
class AuthDiagnostic:
    """API key source diagnostic for a provider."""

    provider: str
    source: str  # "env" | "stored" | "oauth" | "none"
    key_name: str | None = None
    available: bool = False


def diagnose_auth(provider: str, store: CredentialStore | None = None) -> AuthDiagnostic:
    """Diagnose where the API key for a provider comes from."""
    env_keys = _ENV_KEY_MAP.get(provider, [])
    for key in env_keys:
        val = os.environ.get(key)
        if val:
            return AuthDiagnostic(provider=provider, source="env", key_name=key, available=True)
    generic = os.environ.get(f"PIY_API_KEY_{provider.upper()}")
    if generic:
        key_name = f"PIY_API_KEY_{provider.upper()}"
        return AuthDiagnostic(
            provider=provider,
            source="env",
            key_name=key_name,
            available=True,
        )
    return AuthDiagnostic(provider=provider, source="none", available=False)
