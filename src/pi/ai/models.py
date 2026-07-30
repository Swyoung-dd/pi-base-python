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
