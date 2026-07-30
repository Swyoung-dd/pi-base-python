"""xAI OAuth 2.0 device-code 登录流程。"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from urllib.parse import urlparse

import httpx

from pi.ai.oauth import (
    OAuthCredential,
    OAuthEvent,
    OAuthNotifier,
    get_oauth_flow,
    register_oauth_flow,
)

XAI_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
XAI_SCOPE = "openid profile email offline_access grok-cli:access api:access"
XAI_DEVICE_CODE_URL = "https://auth.x.ai/oauth2/device/code"
XAI_TOKEN_URL = "https://auth.x.ai/oauth2/token"
REFRESH_SKEW_MS = 5 * 60 * 1000


def _required_string(body: dict[str, Any], field: str) -> str:
    value = body.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Invalid xAI OAuth response field: {field}")
    return value


def _validate_verification_uri(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("Untrusted verification URI in xAI OAuth response")
    return value


def _credential_from_response(
    body: dict[str, Any],
    previous_refresh: str | None = None,
) -> OAuthCredential:
    refresh = body.get("refresh_token") or previous_refresh
    if not isinstance(refresh, str) or not refresh:
        raise ValueError("Invalid xAI OAuth response field: refresh_token")
    expires_in = body.get("expires_in", 3600)
    if not isinstance(expires_in, (int, float)) or expires_in <= 0:
        raise ValueError("Invalid xAI OAuth response field: expires_in")
    return OAuthCredential(
        access=_required_string(body, "access_token"),
        refresh=refresh,
        expires=int(time.time() * 1000 + expires_in * 1000 - REFRESH_SKEW_MS),
    )


async def _post_form(url: str, data: dict[str, str]) -> tuple[int, dict[str, Any]]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            url,
            data=data,
            headers={"Accept": "application/json"},
        )
    try:
        body = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"xAI OAuth returned invalid JSON (HTTP {response.status_code})"
        ) from exc
    return response.status_code, body if isinstance(body, dict) else {}


class XaiOAuthFlow:
    name = "xAI (Grok/X subscription)"

    async def login(self, notify: OAuthNotifier) -> OAuthCredential:
        status, body = await _post_form(
            XAI_DEVICE_CODE_URL,
            {"client_id": XAI_CLIENT_ID, "scope": XAI_SCOPE, "referrer": "pi"},
        )
        if status >= 400:
            raise RuntimeError(f"xAI OAuth device authorization failed (HTTP {status})")
        device_code = _required_string(body, "device_code")
        user_code = _required_string(body, "user_code")
        verification_uri = _validate_verification_uri(
            str(body.get("verification_uri_complete") or _required_string(body, "verification_uri"))
        )
        interval = max(1.0, float(body.get("interval", 5)))
        expires_in = max(1.0, float(body.get("expires_in", 900)))
        notify(
            OAuthEvent(
                type="device_code",
                user_code=user_code,
                verification_uri=verification_uri,
                interval_seconds=interval,
                expires_in_seconds=expires_in,
            )
        )
        deadline = time.monotonic() + expires_in
        await asyncio.sleep(interval)
        while time.monotonic() < deadline:
            status, token_body = await _post_form(
                XAI_TOKEN_URL,
                {
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "client_id": XAI_CLIENT_ID,
                    "device_code": device_code,
                },
            )
            if status < 400:
                return _credential_from_response(token_body)
            error = token_body.get("error")
            if error == "slow_down":
                interval = max(interval + 5, float(token_body.get("interval", 0)))
            elif error != "authorization_pending":
                raise RuntimeError(f"xAI OAuth device token polling failed: {error}")
            await asyncio.sleep(min(interval, max(0, deadline - time.monotonic())))
        raise TimeoutError("xAI OAuth device flow timed out")

    async def refresh(self, credential: OAuthCredential) -> OAuthCredential:
        status, body = await _post_form(
            XAI_TOKEN_URL,
            {
                "grant_type": "refresh_token",
                "client_id": XAI_CLIENT_ID,
                "refresh_token": credential.refresh,
            },
        )
        if status >= 400:
            raise RuntimeError(f"xAI OAuth token refresh failed (HTTP {status})")
        return _credential_from_response(body, credential.refresh)


def register_xai_oauth() -> None:
    if get_oauth_flow("xai") is None:
        register_oauth_flow("xai", XaiOAuthFlow())
