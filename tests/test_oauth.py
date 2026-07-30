"""OAuth 凭据存储、刷新和传输集成测试。"""

import asyncio
import time

import pytest

from pi.ai.auth import get_api_key
from pi.ai.oauth import (
    OAuthCredential,
    OAuthCredentialStore,
    register_oauth_flow,
    resolve_oauth_access_token,
)
from pi.ai.oauth_xai import _validate_verification_uri
from pi.ai.providers import openai
from pi.ai.streaming import DoneEvent
from pi.ai.types import AssistantMessage, Context, Model, StopReason, TextContent


class _FakeFlow:
    name = "Test OAuth"

    def __init__(self) -> None:
        self.refresh_count = 0

    async def login(self, notify):
        raise NotImplementedError

    async def refresh(self, credential):
        self.refresh_count += 1
        await asyncio.sleep(0)
        return OAuthCredential(
            access="fresh-token",
            refresh=credential.refresh,
            expires=int(time.time() * 1000) + 3_600_000,
        )


def test_generic_api_key_uses_piy_prefix(monkeypatch):
    monkeypatch.setenv("PIY_API_KEY_CUSTOM", "custom-key")

    assert get_api_key("custom") == "custom-key"


async def test_concurrent_oauth_resolution_refreshes_once(tmp_path):
    store = OAuthCredentialStore(tmp_path / "auth.json")
    expired = OAuthCredential(access="old", refresh="refresh", expires=1)
    await store.modify("test-oauth", lambda current: expired)
    flow = _FakeFlow()
    register_oauth_flow("test-oauth", flow)

    tokens = await asyncio.gather(
        resolve_oauth_access_token("test-oauth", store),
        resolve_oauth_access_token("test-oauth", store),
    )

    assert tokens == ["fresh-token", "fresh-token"]
    assert flow.refresh_count == 1
    assert await store.list() == ["test-oauth"]


async def test_provider_uses_stored_oauth_bearer_token(tmp_path, monkeypatch):
    auth_file = tmp_path / "auth.json"
    monkeypatch.setenv("PIY_AUTH_FILE", str(auth_file))
    store = OAuthCredentialStore(auth_file)
    await store.modify(
        "xai",
        lambda current: OAuthCredential(
            access="oauth-access",
            refresh="oauth-refresh",
            expires=int(time.time() * 1000) + 3_600_000,
        ),
    )
    captured = {}

    async def fake_worker(url, headers, payload, model, stream, max_retries, timeout):
        captured.update(headers)
        message = AssistantMessage(
            content=[TextContent(text="ok")],
            api=model.api,
            provider=model.provider,
            model=model.id,
            stop_reason=StopReason.STOP,
            timestamp=1,
        )
        await stream.push(DoneEvent(message=message))
        await stream.end(message)

    monkeypatch.setattr(openai, "_stream_openai", fake_worker)
    provider = openai.OpenAIProvider("xai")
    model = Model(
        id="grok-test",
        name="Grok Test",
        api="openai-chat-completions",
        provider="xai",
        base_url="https://api.x.ai/v1",
    )

    stream = await provider.stream(model, Context())
    events = [event async for event in stream]

    assert events[-1].type == "done"
    assert captured["Authorization"] == "Bearer oauth-access"


def test_xai_device_flow_rejects_untrusted_verification_uri():
    with pytest.raises(ValueError):
        _validate_verification_uri("http://example.test/verify")
