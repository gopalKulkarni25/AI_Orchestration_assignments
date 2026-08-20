"""Pytest tests for the validated file service registry and workspace policy."""

from __future__ import annotations

from pathlib import Path

import pytest

from validated_file_service import (
    FileToolRegistry,
    WorkspaceManager,
    _plain_filename,
)


# ---------------------------------------------------------------------------
# Filename validator
# ---------------------------------------------------------------------------

class TestPlainFilename:
    def test_valid_filename(self) -> None:
        assert _plain_filename("hello.txt") == "hello.txt"

    def test_valid_filename_with_numbers(self) -> None:
        assert _plain_filename("report_2025.csv") == "report_2025.csv"

    def test_rejects_empty_string(self) -> None:
        with pytest.raises(ValueError, match="Invalid filename"):
            _plain_filename("")

    def test_rejects_dot(self) -> None:
        with pytest.raises(ValueError, match="Invalid filename"):
            _plain_filename(".")

    def test_rejects_dotdot(self) -> None:
        with pytest.raises(ValueError, match="Invalid filename"):
            _plain_filename("..")

    def test_rejects_forward_slash_traversal(self) -> None:
        with pytest.raises(ValueError):
            _plain_filename("../evil.txt")

    def test_rejects_nested_forward_slash(self) -> None:
        with pytest.raises(ValueError):
            _plain_filename("subdir/file.txt")

    def test_rejects_backslash_traversal(self) -> None:
        with pytest.raises(ValueError):
            _plain_filename("subdir\\file.txt")

    def test_rejects_double_dotdot_prefix(self) -> None:
        with pytest.raises(ValueError):
            _plain_filename("../../root.txt")


# ---------------------------------------------------------------------------
# FileToolRegistry dispatch
# ---------------------------------------------------------------------------

class TestFileToolRegistry:
    @pytest.fixture()
    def registry(self, tmp_path: Path) -> FileToolRegistry:
        manager = WorkspaceManager(workspace=tmp_path)
        return FileToolRegistry(manager=manager)

    def test_write_and_read_roundtrip(self, registry: FileToolRegistry) -> None:
        registry.dispatch("write_file", {"filepath": "test.txt", "content": "hello"})
        result = registry.dispatch("read_file", {"filepath": "test.txt"})
        assert result == "hello"

    def test_write_with_custom_encoding(self, registry: FileToolRegistry) -> None:
        result = registry.dispatch("write_file", {"filepath": "utf.txt", "content": "hi", "encoding": "utf-8"})
        assert "utf.txt" in result

    def test_list_directory_shows_written_file(self, registry: FileToolRegistry) -> None:
        registry.dispatch("write_file", {"filepath": "a.txt", "content": "x"})
        result = registry.dispatch("list_directory", {})
        assert "a.txt" in result

    def test_list_directory_empty_workspace(self, registry: FileToolRegistry) -> None:
        result = registry.dispatch("list_directory", {})
        assert "empty" in result.lower()

    def test_traversal_write_is_rejected(self, registry: FileToolRegistry) -> None:
        result = registry.dispatch("write_file", {"filepath": "../evil.txt", "content": "bad"})
        assert "error" in result.lower() or "validation" in result.lower()

    def test_traversal_read_is_rejected(self, registry: FileToolRegistry) -> None:
        result = registry.dispatch("read_file", {"filepath": "../passwd"})
        assert "error" in result.lower() or "validation" in result.lower()

    def test_unknown_tool_returns_error(self, registry: FileToolRegistry) -> None:
        result = registry.dispatch("delete_all", {})
        assert "unknown tool" in result.lower()

    def test_read_missing_file_returns_error(self, registry: FileToolRegistry) -> None:
        result = registry.dispatch("read_file", {"filepath": "missing.txt"})
        assert "error" in result.lower()

    def test_dotdot_filename_rejected(self, registry: FileToolRegistry) -> None:
        result = registry.dispatch("write_file", {"filepath": "..", "content": "bad"})
        assert "error" in result.lower() or "validation" in result.lower()

    def test_empty_filename_rejected(self, registry: FileToolRegistry) -> None:
        result = registry.dispatch("write_file", {"filepath": "", "content": "bad"})
        assert "error" in result.lower() or "validation" in result.lower()


# ---------------------------------------------------------------------------
# WorkspaceManager boundaries
# ---------------------------------------------------------------------------

class TestWorkspaceManager:
    @pytest.fixture()
    def manager(self, tmp_path: Path) -> WorkspaceManager:
        return WorkspaceManager(workspace=tmp_path)

    def test_write_creates_file(self, manager: WorkspaceManager, tmp_path: Path) -> None:
        manager.write_file("out.txt", "data")
        assert (tmp_path / "out.txt").read_text() == "data"

    def test_read_returns_content(self, manager: WorkspaceManager) -> None:
        manager.write_file("r.txt", "content")
        assert manager.read_file("r.txt") == "content"

    def test_read_missing_raises(self, manager: WorkspaceManager) -> None:
        with pytest.raises(FileNotFoundError):
            manager.read_file("nope.txt")

    def test_list_directory_sorted(self, manager: WorkspaceManager) -> None:
        manager.write_file("z.txt", "z")
        manager.write_file("a.txt", "a")
        result = manager.list_directory()
        lines = result.splitlines()
        assert lines == sorted(lines)
