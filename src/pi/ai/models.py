"""模型注册表与发现。

在 TypeScript 版本中，模型数据由提供商目录自动生成。
这里使用静态注册表手动维护，后续可添加生成器。
"""

from __future__ import annotations

from pi.ai.types import Model, ModelCost

# 内置模型定义。
# 在 pi-ai 中这些由提供商目录自动生成。
_MODELS: list[Model] = [
    Model(
        id="gpt-4o",
        name="GPT-4o",
        api="openai-responses",
        provider="openai",
        base_url="https://api.openai.com/v1",
        reasoning=False,
        input=["text", "image"],
        cost=ModelCost(input=2.5, output=10.0),
        context_window=128_000,
        max_tokens=16_384,
    ),
    Model(
        id="gpt-4o-mini",
        name="GPT-4o mini",
        api="openai-responses",
        provider="openai",
        base_url="https://api.openai.com/v1",
        reasoning=False,
        input=["text", "image"],
        cost=ModelCost(input=0.15, output=0.6),
        context_window=128_000,
        max_tokens=16_384,
    ),
    Model(
        id="claude-sonnet-4-20250514",
        name="Claude Sonnet 4",
        api="anthropic-messages",
        provider="anthropic",
        base_url="https://api.anthropic.com",
        reasoning=False,
        input=["text", "image"],
        cost=ModelCost(input=3.0, output=15.0),
        context_window=200_000,
        max_tokens=16_384,
    ),
    Model(
        id="claude-3-5-haiku-20241022",
        name="Claude 3.5 Haiku",
        api="anthropic-messages",
        provider="anthropic",
        base_url="https://api.anthropic.com",
        reasoning=False,
        input=["text", "image"],
        cost=ModelCost(input=0.8, output=4.0),
        context_window=200_000,
        max_tokens=8_192,
    ),
]


def list_models() -> list[Model]:
    """返回所有已知模型。"""
    return list(_MODELS)


def get_model(model_id: str) -> Model | None:
    """按 ID 查找模型。"""
    for m in _MODELS:
        if m.id == model_id:
            return m
    return None


def models_by_provider(provider: str) -> list[Model]:
    """返回指定提供商的所有模型。"""
    return [m for m in _MODELS if m.provider == provider]
