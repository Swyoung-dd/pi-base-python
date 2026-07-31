"""Tests for the local piY Web server boundary."""

from pathlib import Path

import pytest

from pi.agent.session import JsonlStorage
from pi.agent.types import create_user_message
from pi.coding_agent.config import Config
from pi.coding_agent.sessions import session_path
from pi.web.server import (
    WebApiError,
    list_workspace_directory,
    resolve_workspace_path,
    session_payload,
)


def test_resolve_workspace_path_stays_inside_project(tmp_path: Path) -> None:
    source = tmp_path / "src" / "app.py"
    source.parent.mkdir()
    source.write_text("print('ok')\n", encoding="utf-8")

    assert resolve_workspace_path(tmp_path, "src/app.py") == source
    with pytest.raises(WebApiError, match="outside the project"):
        resolve_workspace_path(tmp_path, "../outside.txt")


def test_list_workspace_directory_sorts_and_hides_runtime_directories(tmp_path: Path) -> None:
    (tmp_path / "zeta.txt").write_text("z", encoding="utf-8")
    (tmp_path / "alpha").mkdir()
    (tmp_path / ".git").mkdir()

    payload = list_workspace_directory(tmp_path)

    assert payload["path"] == ""
    assert payload["parent"] is None
    assert [entry["name"] for entry in payload["entries"]] == ["alpha", "zeta.txt"]


async def test_session_payload_uses_existing_piy_session_format(tmp_path: Path) -> None:
    config = Config(
        config_dir=tmp_path / ".piy",
        sessions_dir=tmp_path / ".piy" / "sessions",
    )
    storage = JsonlStorage(session_path(config.sessions_dir, "web-session"))
    await storage.append_model_change("openai", "gpt-4o-mini")
    await storage.append_message(create_user_message("hello from web"))

    payload = await session_payload(config, "web-session")

    assert payload["model"] == {"provider": "openai", "id": "gpt-4o-mini"}
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][0]["content"] == "hello from web"
