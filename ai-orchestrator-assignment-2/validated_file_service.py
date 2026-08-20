"""
Validated FastMCP file service with Pydantic-backed tool registry.

Part 1 of Assignment 2: typed, validated file workspace exposed through FastMCP.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError, field_validator

WORKSPACE_DIR = Path(__file__).parent / "workspace"


def _plain_filename(value: str) -> str:
    """Reject empty names, '.', '..', traversal attempts, nested paths, and path separators."""
    if not value or value in {".", ".."}:
        raise ValueError(f"Invalid filename: {value!r}")
    # Explicitly reject backslashes (Windows separator) — on POSIX, Path treats
    # them as literal characters rather than separators, so they slip through
    # the Path().name check otherwise.
    if "\\" in value:
        raise ValueError("filepath must not contain backslashes")
    # Path(value).name strips any directory components and resolves separators.
    # If the result differs from value, a separator or traversal component was present.
    if Path(value).name != value:
        raise ValueError("filepath must be a plain filename with no traversal or path separators")
    return value


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------

class WriteFile(BaseModel):
    """Write content to a new or existing workspace file."""

    filepath: str = Field(description="Plain filename only; paths are forbidden.")
    content: str = Field(description="Complete text to write.")
    encoding: str = Field(default="utf-8", description="Text encoding.")

    @field_validator("filepath")
    @classmethod
    def validate_filepath(cls, value: str) -> str:
        return _plain_filename(value)


class ReadFile(BaseModel):
    """Read one workspace file."""

    filepath: str = Field(description="Plain filename only.")

    @field_validator("filepath")
    @classmethod
    def validate_filepath(cls, value: str) -> str:
        return _plain_filename(value)


class ListDirectory(BaseModel):
    """List files from the managed workspace."""


# ---------------------------------------------------------------------------
# Workspace manager — bounded filesystem access
# ---------------------------------------------------------------------------

class WorkspaceManager:
    """Filesystem operations confined to a single managed directory."""

    def __init__(self, workspace: Path = WORKSPACE_DIR) -> None:
        self.workspace = workspace
        self.workspace.mkdir(parents=True, exist_ok=True)

    def write_file(self, filepath: str, content: str, encoding: str = "utf-8") -> str:
        target = self.workspace / filepath
        target.write_text(content, encoding=encoding)
        return f"Wrote {len(content)} chars to {filepath!r}."

    def read_file(self, filepath: str) -> str:
        target = self.workspace / filepath
        if not target.exists():
            raise FileNotFoundError(f"File not found: {filepath!r}")
        return target.read_text(encoding="utf-8")

    def list_directory(self) -> str:
        files = sorted(f.name for f in self.workspace.iterdir() if f.is_file())
        return "\n".join(files) if files else "(workspace is empty)"


# ---------------------------------------------------------------------------
# Tool registry — single dispatch/validation boundary
# ---------------------------------------------------------------------------

class ToolRegistration(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    model: type[BaseModel]
    executor: Callable[[BaseModel], str]


class FileToolRegistry:
    """Maps tool names to validated Pydantic models and executor functions."""

    def __init__(self, manager: WorkspaceManager | None = None) -> None:
        self.manager = manager or WorkspaceManager()
        self._registry: dict[str, ToolRegistration] = {
            "write_file": ToolRegistration(
                model=WriteFile,
                executor=lambda a: self.manager.write_file(**a.model_dump()),
            ),
            "read_file": ToolRegistration(
                model=ReadFile,
                executor=lambda a: self.manager.read_file(a.filepath),
            ),
            "list_directory": ToolRegistration(
                model=ListDirectory,
                executor=lambda a: self.manager.list_directory(),
            ),
        }

    def dispatch(self, tool_name: str, raw_arguments: dict[str, Any]) -> str:
        registration = self._registry.get(tool_name)
        if registration is None:
            return f"Error: unknown tool {tool_name!r}"
        try:
            arguments = registration.model.model_validate(raw_arguments)
            return registration.executor(arguments)
        except ValidationError as error:
            return f"Validation error: {error.errors(include_url=False)}"
        except (OSError, ValueError, FileNotFoundError) as error:
            return f"Tool error: {type(error).__name__}: {error}"


# ---------------------------------------------------------------------------
# FastMCP server — delegates entirely to the registry; no repeated validation
# ---------------------------------------------------------------------------

def create_server(registry: FileToolRegistry | None = None) -> Any:
    try:
        from fastmcp import FastMCP
    except ImportError as error:
        raise RuntimeError("Install dependencies from requirements.txt first.") from error

    registry = registry or FileToolRegistry()
    mcp = FastMCP("Validated File Server")

    @mcp.tool()
    def write_file(filepath: str, content: str, encoding: str = "utf-8") -> str:
        """Write content to a validated plain filename in the managed workspace."""
        return registry.dispatch("write_file", {"filepath": filepath, "content": content, "encoding": encoding})

    @mcp.tool()
    def read_file(filepath: str) -> str:
        """Read a validated plain filename from the managed workspace."""
        return registry.dispatch("read_file", {"filepath": filepath})

    @mcp.tool()
    def list_files() -> str:
        """List all files in the managed workspace."""
        return registry.dispatch("list_directory", {})

    @mcp.resource("workspace://files")
    def workspace_files() -> str:
        """Read-only snapshot of file names in the managed workspace."""
        return registry.dispatch("list_directory", {})

    return mcp


# ---------------------------------------------------------------------------
# In-process demo (--explore / --demo)
# ---------------------------------------------------------------------------

async def explore_server(mcp: Any) -> None:
    try:
        from fastmcp import Client
    except ImportError as error:
        raise RuntimeError("Install dependencies from requirements.txt first.") from error

    async with Client(mcp) as client:
        print("=== Discovering tools ===")
        tools = await client.list_tools()
        for tool in tools:
            print(f"  [{tool.name}] {tool.description}")

        print("\n=== Write a valid file ===")
        result = await client.call_tool("write_file", {"filepath": "demo.txt", "content": "Hello from FastMCP!"})
        print(f"  Result: {result}")

        print("\n=== Read it back ===")
        result = await client.call_tool("read_file", {"filepath": "demo.txt"})
        print(f"  Content: {result}")

        print("\n=== List workspace ===")
        result = await client.call_tool("list_files", {})
        print(f"  Files:\n{result}")

        print("\n=== Traversal rejection ===")
        result = await client.call_tool("write_file", {"filepath": "../evil.txt", "content": "should be blocked"})
        print(f"  Result (expected error): {result}")

        print("\n=== Resources ===")
        resources = await client.list_resources()
        for r in resources:
            print(f"  {r.uri}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validated FastMCP file service")
    parser.add_argument(
        "--explore", "--demo",
        action="store_true",
        help="Run in-process tool discovery, valid calls, and traversal rejection demo.",
    )
    args = parser.parse_args()
    mcp = create_server()
    if args.explore:
        asyncio.run(explore_server(mcp))
    else:
        mcp.run()


if __name__ == "__main__":
    main()
