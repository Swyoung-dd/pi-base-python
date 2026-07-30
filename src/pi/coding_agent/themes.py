"""终端主题的加载和 Rich 样式校验。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from rich.errors import StyleSyntaxError
from rich.style import Style


@dataclass(frozen=True)
class Theme:
    """piY 交互终端使用的语义样式。"""

    name: str = "default"
    primary: str = "blue"
    muted: str = "dim"
    thinking: str = "dim italic"
    error: str = "bold red"
    warning: str = "yellow"
    success: str = "green"


_BUILTIN_THEMES = {
    "default": Theme(),
    "high-contrast": Theme(
        name="high-contrast",
        primary="bold bright_cyan",
        muted="white",
        thinking="italic bright_black",
        error="bold bright_red",
        warning="bold bright_yellow",
        success="bold bright_green",
    ),
    "mono": Theme(
        name="mono",
        primary="bold white",
        muted="dim white",
        thinking="dim italic white",
        error="bold white",
        warning="italic white",
        success="white",
    ),
}
_STYLE_FIELDS = {"primary", "muted", "thinking", "error", "warning", "success"}


def list_builtin_themes() -> list[str]:
    return sorted(_BUILTIN_THEMES)


def load_theme_file(path: Path) -> Theme:
    """读取自定义主题文件并校验全部 Rich 样式。"""
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Theme must be a mapping: {path}")
    styles = value.get("styles", value)
    if not isinstance(styles, dict):
        raise ValueError(f"Theme styles must be a mapping: {path}")
    unknown = set(styles) - _STYLE_FIELDS - {"name"}
    if unknown:
        raise ValueError(f"Unknown theme style: {', '.join(sorted(unknown))}")
    base = _BUILTIN_THEMES["default"]
    resolved = {}
    for field_name in _STYLE_FIELDS:
        style = str(styles.get(field_name, getattr(base, field_name)))
        try:
            Style.parse(style)
        except StyleSyntaxError as exc:
            raise ValueError(f"Invalid theme style {field_name}: {style}") from exc
        resolved[field_name] = style
    return Theme(
        name=str(value.get("name") or styles.get("name") or path.stem),
        **resolved,
    )


def load_theme(name: str, search_paths: list[Path] | None = None) -> Theme:
    """按名称加载内置或自定义主题。"""
    if name in _BUILTIN_THEMES:
        return _BUILTIN_THEMES[name]
    requested = Path(name)
    candidates = []
    if requested.is_absolute():
        candidates.append(requested)
    else:
        for root in search_paths or []:
            if requested.suffix.lower() in {".yaml", ".yml"}:
                candidates.append(root / requested)
            else:
                candidates.extend((root / f"{name}.yaml", root / f"{name}.yml"))
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise ValueError(f"Theme not found: {name}")
    return load_theme_file(path)
