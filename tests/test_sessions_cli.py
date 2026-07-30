"""会话发现和 CLI 选择测试。"""

import os

import pytest

from pi.agent.session import JsonlStorage
from pi.agent.types import create_user_message
from pi.coding_agent.sessions import (
    list_sessions,
    resolve_session_id,
    session_path,
    validate_session_id,
)


async def test_list_sessions_and_continue_latest(tmp_path):
    first_path = session_path(tmp_path, "first")
    second_path = session_path(tmp_path, "second")
    first = JsonlStorage(first_path)
    second = JsonlStorage(second_path)
    await first.append_message(create_user_message("first prompt"))
    await second.append_message(create_user_message("second prompt"))
    os.utime(first_path, (1, 1))
    os.utime(second_path, (2, 2))

    sessions = await list_sessions(tmp_path)

    assert [item.session_id for item in sessions] == ["second", "first"]
    assert sessions[0].preview == "second prompt"
    assert await resolve_session_id(tmp_path, continue_latest=True) == "second"


def test_session_id_rejects_path_traversal(tmp_path):
    with pytest.raises(ValueError):
        validate_session_id("../outside")
    with pytest.raises(ValueError):
        session_path(tmp_path, "folder/session")
