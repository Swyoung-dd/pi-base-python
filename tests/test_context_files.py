"""项目上下文文件发现测试。"""

from pi.coding_agent.context_files import format_context_files, load_context_files


def test_context_files_load_user_and_project_hierarchy(tmp_path):
    user_dir = tmp_path / "user"
    project = tmp_path / "workspace" / "project"
    user_dir.mkdir()
    project.mkdir(parents=True)
    (user_dir / "AGENTS.md").write_text("user rules", encoding="utf-8")
    (tmp_path / "workspace" / "CLAUDE.md").write_text("parent rules", encoding="utf-8")
    (project / "AGENTS.md").write_text("project rules", encoding="utf-8")

    files = load_context_files(project, user_dir)

    assert [item.content for item in files] == ["user rules", "parent rules", "project rules"]
    formatted = format_context_files(files)
    assert formatted.index("user rules") < formatted.index("parent rules")
    assert formatted.index("parent rules") < formatted.index("project rules")


def test_agents_file_takes_precedence_over_claude_file(tmp_path):
    project = tmp_path / "project"
    user_dir = tmp_path / "user"
    project.mkdir()
    user_dir.mkdir()
    (project / "AGENTS.md").write_text("agents", encoding="utf-8")
    (project / "CLAUDE.md").write_text("claude", encoding="utf-8")

    files = load_context_files(project, user_dir)

    assert [item.path.name for item in files] == ["AGENTS.md"]
    assert files[0].content == "agents"
