"""生成模型、自定义模型的注册与发现。"""

from __future__ import annotations

from pathlib import Path

import yaml

from pi.ai.models_generated import MODELS
from pi.ai.types import Model

_custom_models: dict[tuple[str, str], Model] = {}


def register_model(model: Model) -> None:
    """注册或覆盖一个运行时模型定义。"""
    _custom_models[(model.provider, model.id)] = model


def load_model_file(path: str | Path) -> list[Model]:
    """从 YAML 文件加载项目级模型定义。"""
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    records = value.get("models", []) if isinstance(value, dict) else value
    models = [Model.model_validate(record) for record in records]
    for model in models:
        register_model(model)
    return models


def clear_custom_models() -> None:
    """清空运行时模型，主要用于隔离嵌入方和测试。"""
    _custom_models.clear()


def list_models() -> list[Model]:
    """返回所有已知模型。"""
    merged = {(model.provider, model.id): model for model in MODELS}
    merged.update(_custom_models)
    return sorted(merged.values(), key=lambda model: (model.provider, model.id))


def get_model(model_id: str, provider: str | None = None) -> Model | None:
    """按 ID 查找模型。"""
    matches = [
        model
        for model in list_models()
        if model.id == model_id and (provider is None or model.provider == provider)
    ]
    return matches[0] if len(matches) == 1 else None


def models_by_provider(provider: str) -> list[Model]:
    """返回指定提供商的所有模型。"""
    return [model for model in list_models() if model.provider == provider]


async def refresh_models(provider_id: str | None = None) -> list[Model]:
    """Refresh models from provider API (e.g., OpenAI /models endpoint).

    For providers that support model listing, fetch and register new models.
    Returns the list of refreshed models. Failures are non-fatal.
    """
    import httpx

    from pi.ai.auth import get_api_key
    from pi.ai.providers.registry import get_provider

    targets = [provider_id] if provider_id else list_providers_safe()
    for pid in targets:
        provider = get_provider(pid)
        if provider is None:
            continue
        api_key = get_api_key(pid)
        if not api_key:
            continue
        base_url = "https://api.openai.com/v1"
        for m in list_models():
            if m.provider == pid and m.base_url:
                base_url = m.base_url
                break
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(f"{base_url.rstrip('/')}/models", headers=headers)
                resp.raise_for_status()
                data = resp.json()
                for item in data.get("data", []):
                    model_id = item.get("id")
                    if not model_id:
                        continue
                    existing = get_model(model_id, pid)
                    if existing is None:
                        register_model(Model(
                            id=model_id,
                            name=model_id,
                            api="openai-chat",
                            provider=pid,
                            base_url=base_url,
                        ))
        except Exception:
            pass
    return list_models()


def list_providers_safe() -> list[str]:
    """Return registered provider IDs, handling import errors."""
    try:
        from pi.ai.providers.registry import list_providers
        return list_providers()
    except Exception:
        return []
