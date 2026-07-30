"""提示文本中的 @file 引用解析。"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

_REFERENCE_PATTERN = re.compile(r"(?<![\w.@])@(?!@)(?:\"([^\"]+)\"|'([^']+)'|([^\s]+))")
_LINE_RANGE_PATTERN = re.compile(r"^(.*):(\d+)(?:-(\d+))?$")


class FileReferenceError(ValueError):
    pass


@dataclass(frozen=True)
class FileReference:
    path: Path
    display_path: str
    start_line: int | None = None
    end_line: int | None = None


def _parse_reference(raw: str, cwd: Path) -> FileReference:
    start_line = None
    end_line = None
    path_text = raw
    line_match = _LINE_RANGE_PATTERN.fullmatch(raw)
    if line_match:
        path_text = line_match.group(1)
        start_line = int(line_match.group(2))
        end_line = int(line_match.group(3) or start_line)
        if start_line < 1 or end_line < start_line:
            raise FileReferenceError(f"Invalid line range: @{raw}")

    root = cwd.resolve()
    candidate = Path(path_text)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise FileReferenceError(f"File reference is outside the working directory: @{raw}")
    if not resolved.is_file():
        raise FileReferenceError(f"Referenced file not found: @{raw}")
    return FileReference(
        path=resolved,
        display_path=resolved.relative_to(root).as_posix(),
        start_line=start_line,
        end_line=end_line,
    )


def _read_reference(reference: FileReference, max_bytes: int) -> str:
    if reference.path.stat().st_size > max_bytes:
        raise FileReferenceError(f"Referenced file is too large: {reference.display_path}")
    data = reference.path.read_bytes()
    if b"\x00" in data:
        raise FileReferenceError(f"Referenced file appears to be binary: {reference.display_path}")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FileReferenceError(
            f"Referenced file is not valid UTF-8: {reference.display_path}"
        ) from exc
    if reference.start_line is not None:
        lines = text.splitlines()
        text = "\n".join(lines[reference.start_line - 1 : reference.end_line])
    return text


def expand_file_references(
    prompt: str,
    cwd: Path,
    max_bytes: int = 1_000_000,
) -> str:
    """将提示中的文件引用展开为带路径边界的附加上下文。"""
    references: list[FileReference] = []
    seen: set[tuple[Path, int | None, int | None]] = set()
    for match in _REFERENCE_PATTERN.finditer(prompt):
        raw = next(group for group in match.groups() if group is not None)
        reference = _parse_reference(raw, cwd)
        key = (reference.path, reference.start_line, reference.end_line)
        if key not in seen:
            references.append(reference)
            seen.add(key)
    if not references:
        return prompt.replace("@@", "@")

    blocks = []
    for reference in references:
        content = _read_reference(reference, max_bytes)
        line_suffix = ""
        if reference.start_line is not None:
            line_suffix = f":{reference.start_line}-{reference.end_line}"
        label = html.escape(reference.display_path + line_suffix, quote=True)
        blocks.append(f'<file path="{label}">\n{content}\n</file>')
    return prompt.replace("@@", "@") + "\n\n" + "\n\n".join(blocks)
