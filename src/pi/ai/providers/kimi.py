"""Kimi / Moonshot provider - OpenAI-compatible API."""

from __future__ import annotations

from pi.ai.providers.openai import OpenAIProvider


class KimiProvider(OpenAIProvider):
    """Kimi via Moonshot OpenAI-compatible endpoint."""

    def __init__(self) -> None:
        super().__init__("kimi")
