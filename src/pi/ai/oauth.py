"""OAuth 凭据存储、流程注册和并发安全刷新。"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel


class OAuthCredential(BaseModel):
    type: Literal["oauth"] = "oauth"
    access: str
    refresh: str
    expires: int
    model_config = {"extra": "allow"}


class OAuthEvent(BaseModel):
    type: Literal["info", "device_code", "progress"]
    message: str = ""
    user_code: str | None = None
    verification_uri: str | None = None
    interval_seconds: float | None = None
    expires_in_seconds: float | None = None


OAuthNotifier = Callable[[OAuthEvent], None]


class OAuthFlow(Protocol):
    name: str

    async def login(self, notify: OAuthNotifier) -> OAuthCredential: ...

    async def refresh(self, credential: OAuthCredential) -> OAuthCredential: ...


CredentialModifier = Callable[
    [OAuthCredential | None],
    OAuthCredential | None | Awaitable[OAuthCredential | None],
]


class OAuthCredentialStore:
    """使用原子替换写入 auth.json，并按 provider 串行化进程内修改。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._locks: dict[str, asyncio.Lock] = {}

    def _read_all(self) -> dict[str, OAuthCredential]:
        if not self.path.exists():
            return {}
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return {
            provider: OAuthCredential.model_validate(value)
            for provider, value in raw.items()
            if isinstance(value, dict) and value.get("type") == "oauth"
        }

    def _write_all(self, credentials: dict[str, OAuthCredential]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        data = {
            provider: credential.model_dump(mode="json")
            for provider, credential in credentials.items()
        }
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with suppress(OSError):
            temporary.chmod(0o600)
        os.replace(temporary, self.path)

    async def read(self, provider_id: str) -> OAuthCredential | None:
        credentials = await asyncio.to_thread(self._read_all)
        return credentials.get(provider_id)

    async def list(self) -> list[str]:
        return sorted((await asyncio.to_thread(self._read_all)).keys())

    async def modify(
        self,
        provider_id: str,
        modifier: CredentialModifier,
    ) -> OAuthCredential | None:
        lock = self._locks.setdefault(provider_id, asyncio.Lock())
        async with lock:
            credentials = await asyncio.to_thread(self._read_all)
            updated = modifier(credentials.get(provider_id))
            if inspect.isawaitable(updated):
                updated = await updated
            if updated is not None:
                credentials[provider_id] = updated
                await asyncio.to_thread(self._write_all, credentials)
            return credentials.get(provider_id)

    async def delete(self, provider_id: str) -> None:
        lock = self._locks.setdefault(provider_id, asyncio.Lock())
        async with lock:
            credentials = await asyncio.to_thread(self._read_all)
            if credentials.pop(provider_id, None) is not None:
                await asyncio.to_thread(self._write_all, credentials)


_flows: dict[str, OAuthFlow] = {}
_default_store: OAuthCredentialStore | None = None


def register_oauth_flow(provider_id: str, flow: OAuthFlow) -> None:
    _flows[provider_id] = flow


def get_oauth_flow(provider_id: str) -> OAuthFlow | None:
    return _flows.get(provider_id)


def list_oauth_flows() -> list[str]:
    return sorted(_flows)


def get_default_oauth_store() -> OAuthCredentialStore:
    global _default_store
    path = Path(os.environ.get("PI_AUTH_FILE", Path.home() / ".pi" / "auth.json"))
    if _default_store is None or _default_store.path != path:
        _default_store = OAuthCredentialStore(path)
    return _default_store


async def login_oauth(
    provider_id: str,
    notify: OAuthNotifier,
    store: OAuthCredentialStore | None = None,
) -> OAuthCredential:
    flow = get_oauth_flow(provider_id)
    if flow is None:
        raise ValueError(f"Provider does not support OAuth login: {provider_id}")
    credential = await flow.login(notify)
    target = store or get_default_oauth_store()
    await target.modify(provider_id, lambda current: credential)
    return credential


async def resolve_oauth_access_token(
    provider_id: str,
    store: OAuthCredentialStore | None = None,
    minimum_validity_ms: int = 5 * 60 * 1000,
) -> str | None:
    """解析有效 access token；临近过期时在 provider 锁内只刷新一次。"""
    target = store or get_default_oauth_store()
    credential = await target.read(provider_id)
    if credential is None:
        return None

    def expires_soon(value: OAuthCredential) -> bool:
        return value.expires <= int(time.time() * 1000) + minimum_validity_ms

    if expires_soon(credential):
        flow = get_oauth_flow(provider_id)
        if flow is None:
            raise RuntimeError(
                f"OAuth token expired and no refresh flow is registered: {provider_id}"
            )

        async def refresh(current: OAuthCredential | None) -> OAuthCredential | None:
            if current is None or not expires_soon(current):
                return None
            return await flow.refresh(current)

        credential = await target.modify(provider_id, refresh)
        if credential is None:
            return None
    return credential.access
