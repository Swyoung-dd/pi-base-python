"""Z.AI provider - OpenAI-compatible API."""

from __future__ import annotations

from pi.ai.providers.openai import OpenAIProvider


class ZAIProvider(OpenAIProvider):
    """Z.AI via OpenAI-compatible endpoint."""

    def __init__(self) -> None:
        super().__init__("zai")
