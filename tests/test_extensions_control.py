"""Extension control plane tests."""

from __future__ import annotations

import pytest

from pi.coding_agent.extensions import ExtensionContext, load_extensions


async def test_new_event_types_accepted(tmp_path):
    ext = tmp_path / "evt.py"
    code = (
        "def handler(event, agent):\n"
        "    return\n"
        "\n"
        "def setup(context):\n"
        "    context.on('before_request', handler)\n"
        "    context.on('after_response', handler)\n"
        "    context.on('before_tool', handler)\n"
        "    context.on('after_tool', handler)\n"
        "    context.on('before_compaction', handler)\n"
        "    context.on('before_navigation', handler)\n"
    )
    ext.write_text(code, encoding="utf-8")
    ctx = await load_extensions([ext])
    assert "before_request" in ctx.event_handlers
    assert "after_response" in ctx.event_handlers
    assert "before_tool" in ctx.event_handlers
    assert "after_tool" in ctx.event_handlers
    assert "before_compaction" in ctx.event_handlers
    assert "before_navigation" in ctx.event_handlers


async def test_context_transformer_registration(tmp_path):
    ext = tmp_path / "transform.py"
    code = (
        "def transformer(data):\n"
        "    return data\n"
        "\n"
        "def setup(context):\n"
        "    context.add_context_transformer(transformer)\n"
    )
    ext.write_text(code, encoding="utf-8")
    ctx = await load_extensions([ext])
    assert len(ctx.context_transformers) == 1


async def test_sources_returns_loaded_extensions(tmp_path):
    ext = tmp_path / "src_info.py"
    code = "def setup(context):\n    context.add_system_prompt('test')\n"
    ext.write_text(code, encoding="utf-8")
    ctx = await load_extensions([ext])
    sources = ctx.sources()
    assert len(sources) >= 1


async def test_unload_extension(tmp_path):
    ext = tmp_path / "unload.py"
    code = "def setup(context):\n    context.add_system_prompt('test')\n"
    ext.write_text(code, encoding="utf-8")
    ctx = await load_extensions([ext])
    assert len(ctx.system_prompt_sections) == 1
    sources = ctx.sources()
    source_name = sources[0].name
    result = ctx.unload(source_name)
    assert result is True
    result2 = ctx.unload("nonexistent")
    assert result2 is False


async def test_reload_extension(tmp_path):
    ext = tmp_path / "reload.py"
    code = "def setup(context):\n    context.add_system_prompt('v1')\n"
    ext.write_text(code, encoding="utf-8")
    ctx = await load_extensions([ext])
    assert ctx.system_prompt_sections == ["v1"]
    failures = await ctx.reload()
    assert isinstance(failures, list)


async def test_conflicts_detection(tmp_path):
    ext = tmp_path / "conflict.py"
    code = (
        "def setup(context):\n"
        "    context.add_system_prompt('one')\n"
        "    context.add_system_prompt('two')\n"
    )
    ext.write_text(code, encoding="utf-8")
    ctx = await load_extensions([ext])
    conflicts = ctx.conflicts()
    assert isinstance(conflicts, list)


def test_extension_rejects_unknown_event():
    ctx = ExtensionContext()
    with pytest.raises(ValueError, match="Unknown extension event"):
        ctx.on("totally_unknown_event", lambda e, a: None)
