"""Qwen / DashScope provider - OpenAI-compatible API."""

from __future__ import annotations

from pi.ai.providers.openai import OpenAIProvider


class QwenProvider(OpenAIProvider):
    """Qwen via DashScope OpenAI-compatible endpoint."""

    def __init__(self) -> None:
        super().__init__("qwen")

    def build_payload(self, model, context, options=None):
        payload = super().build_payload(model, context, options)
        payload.pop("stream_options", None)
        return payload
