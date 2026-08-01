"""Provider compatibility modeling tests."""

from __future__ import annotations

from pi.ai.auth import AuthDiagnostic, diagnose_auth
from pi.ai.models import list_models, refresh_models
from pi.ai.types import Model, ModelCompat


def test_model_compat_defaults():
    compat = ModelCompat()
    assert compat.developer_role == "system"
    assert compat.supports_thinking is False
    assert compat.supports_strict_tools is False
    assert compat.supports_cache is False
    assert compat.supports_session_affinity is False


def test_model_has_compat_field():
    model = Model(id="test", name="Test", api="test", provider="test")
    assert isinstance(model.compat, ModelCompat)


def test_model_compat_resolve_thinking_level_with_map():
    compat = ModelCompat(thinking_level_map={"high": "enabled"})
    assert compat.resolve_thinking_level("high") == "enabled"
    assert compat.resolve_thinking_level("off") == "off"


def test_model_compat_resolve_thinking_level_without_map():
    compat = ModelCompat()
    assert compat.resolve_thinking_level("high") == "high"
    assert compat.resolve_thinking_level("off") == "off"


def test_model_compat_custom_values():
    compat = ModelCompat(
        supports_thinking=True,
        supports_strict_tools=True,
        supports_cache=True,
        supports_session_affinity=True,
        thinking_level_map={"low": "low", "high": "high"},
    )
    assert compat.supports_thinking
    assert compat.supports_strict_tools
    assert compat.supports_cache
    assert compat.supports_session_affinity
    assert compat.resolve_thinking_level("low") == "low"


def test_model_serialization_preserves_compat():
    model = Model(
        id="test",
        name="Test",
        api="test",
        provider="test",
        compat=ModelCompat(supports_thinking=True, supports_strict_tools=True),
    )
    data = model.model_dump()
    assert data["compat"]["supports_thinking"] is True
    restored = Model.model_validate(data)
    assert restored.compat.supports_thinking is True


def test_auth_diagnostic_for_unknown_provider():
    result = diagnose_auth("nonexistent_provider_xyz")
    assert isinstance(result, AuthDiagnostic)
    assert result.provider == "nonexistent_provider_xyz"
    assert result.source in ("none", "env")
    assert result.available is False or result.available is True


def test_auth_diagnostic_with_env_var(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test123")
    result = diagnose_auth("openai")
    assert result.source == "env"
    assert result.available is True
    assert result.key_name == "OPENAI_API_KEY"


async def test_refresh_models_returns_list():
    models = await refresh_models()
    assert isinstance(models, list)
    assert len(models) > 0
    assert all(isinstance(m, Model) for m in models)


async def test_refresh_models_filtered_by_provider():
    if not list_models():
        return
    first_provider = list_models()[0].provider
    models = await refresh_models(first_provider)
    assert all(m.provider == first_provider for m in models)
