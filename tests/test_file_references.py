"""@file 引用解析测试。"""

import pytest

from pi.coding_agent.file_references import FileReferenceError, expand_file_references


def test_expands_file_and_line_range_once(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("one\ntwo\nthree\n", encoding="utf-8")

    expanded = expand_file_references("Review @sample.py:2-3 and @sample.py:2-3", tmp_path)

    assert expanded.count('<file path="sample.py:2-3">') == 1
    assert "two\nthree" in expanded
    assert "\none\n" not in expanded


def test_quoted_path_email_and_escaped_at_sign(tmp_path):
    path = tmp_path / "my file.txt"
    path.write_text("content", encoding="utf-8")

    expanded = expand_file_references(
        'Read @"my file.txt" and email user@example.com; literal @@missing.txt',
        tmp_path,
    )

    assert '<file path="my file.txt">' in expanded
    assert "user@example.com" in expanded
    assert "literal @missing.txt" in expanded


def test_rejects_reference_outside_working_directory(tmp_path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    with pytest.raises(FileReferenceError, match="outside the working directory"):
        expand_file_references("Read @../outside.txt", tmp_path)


def test_rejects_binary_file(tmp_path):
    path = tmp_path / "binary.bin"
    path.write_bytes(b"abc\x00def")

    with pytest.raises(FileReferenceError, match="binary"):
        expand_file_references("Read @binary.bin", tmp_path)
