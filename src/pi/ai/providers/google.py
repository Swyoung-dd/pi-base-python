"""Google Gemini provider - stub implementation."""

from __future__ import annotations

import time
from typing import Any

from pi.ai.providers.base import BaseProvider
from pi.ai.streaming import ErrorEvent, EventStream
from pi.ai.types import AssistantMessage, Context, StopReason, TextContent


class GoogleProvider(BaseProvider):
    """Google Gemini API provider (stub)."""

    @property
    def provider_id(self) -> str:
        return "google"

    @property
    def requires_api_key(self) -> bool:
        return True

    def convert_messages(self, context: Context) -> list[dict[str, Any]]:
        """Convert pi messages to Gemini format (scaffolding)."""
        messages: list[dict[str, Any]] = []
        for msg in context.messages:
            if hasattr(msg, "content") and isinstance(msg.content, str):
                role = "user" if msg.role == "user" else "model"
                messages.append({"role": role, "parts": [{"text": msg.content}]})
        return messages

    async def stream(self, model, context, options=None):
        """Stream from Gemini API. Not yet implemented."""
        stream_obj = EventStream()
        now = int(time.time() * 1000)
        error_msg = AssistantMessage(
            content=[TextContent(text="")],
            api=model.api,
            provider=model.provider,
            model=model.id,
            stop_reason=StopReason.ERROR,
            error_message="Google Gemini provider not yet implemented.",
            timestamp=now,
        )
        await stream_obj.push(ErrorEvent(reason="error", error=error_msg))
        await stream_obj.end(error_msg)
        return stream_obj
