"""终端主题加载与校验测试。"""

import pytest

from pi.coding_agent.themes import load_theme, load_theme_file


def test_builtin_and_custom_theme_loading(tmp_path):
    themes = tmp_path / ".piy" / "themes"
    themes.mkdir(parents=True)
    (themes / "ocean.yaml").write_text(
        """name: ocean
styles:
  primary: cyan
  muted: dim white
  thinking: italic cyan
  error: bold red
  warning: yellow
  success: green
""",
        encoding="utf-8",
    )

    custom = load_theme("ocean", [themes])

    assert load_theme("high-contrast").name == "high-contrast"
    assert custom.name == "ocean"
    assert custom.primary == "cyan"
    assert custom.thinking == "italic cyan"


def test_invalid_theme_style_is_rejected(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text("primary: not-a-real-color\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid theme style primary"):
        load_theme_file(path)


def test_unknown_theme_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Theme not found"):
        load_theme("missing", [tmp_path])
