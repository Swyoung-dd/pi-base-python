"""提供商流式请求的有限重试策略。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

import httpx

from pi.ai.streaming import EventStream, RetryEvent

RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


class RetryableProviderError(RuntimeError):
    """可安全重试且尚未产生正文的提供商错误。"""

    def __init__(self, message: str, retry_after: str | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def retry_delay_seconds(attempt: int, retry_after: str | None = None) -> float:
    """解析 Retry-After，否则采用上限为 8 秒的指数退避。"""
    if retry_after:
        try:
            return min(60.0, max(0.0, float(retry_after)))
        except ValueError:
            pass
    return min(8.0, 0.5 * (2 ** (attempt - 1)))


async def raise_for_status(response: httpx.Response, provider: str) -> None:
    if response.status_code == 200:
        return
    body = (await response.aread()).decode(errors="replace")
    message = f"{provider} API error {response.status_code}: {body}"
    if response.status_code in RETRYABLE_STATUS_CODES:
        raise RetryableProviderError(message, response.headers.get("Retry-After"))
    raise RuntimeError(message)


async def run_with_retries(
    operation: Callable[[], Coroutine[Any, Any, None]],
    has_output: Callable[[], bool],
    stream: EventStream,
    max_retries: int,
) -> None:
    """运行请求；收到部分输出后禁止重试，防止重复增量。"""
    for retry_count in range(max_retries + 1):
        try:
            await operation()
            return
        except (httpx.TransportError, RetryableProviderError) as exc:
            if retry_count >= max_retries or has_output():
                raise
            attempt = retry_count + 1
            retry_after = exc.retry_after if isinstance(exc, RetryableProviderError) else None
            delay = retry_delay_seconds(attempt, retry_after)
            await stream.push(
                RetryEvent(
                    attempt=attempt,
                    max_retries=max_retries,
                    delay_ms=int(delay * 1000),
                    error=str(exc),
                )
            )
            await asyncio.sleep(delay)
