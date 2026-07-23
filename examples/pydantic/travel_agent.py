"""A native Pydantic AI travel agent exposed over ACP.

The agent keeps a small "trip workspace" on disk and exposes two tools over it:
`read_trip_file` and `write_trip_file`. The `AdapterConfig` below is what makes
those tools render nicely in an ACP client:

* a `FileSystemProjectionMap` turns the tool calls into file-edit diffs, so the
  client shows what was read and what is about to be written;
* `write_trip_file` requires approval, so the client is asked for permission
  before anything is written to disk;
* a couple of hooks are registered so the client can observe the model request
  and the write tool as they happen.

Run it with `python examples/pydantic/travel_agent.py`, or point an ACP client
(such as Zed) at that command.

The workspace confinement below relies on descriptor-relative `openat` calls, so
this demo targets POSIX platforms (Linux/macOS); Windows lacks `dir_fd` support.
"""

from __future__ import annotations as _annotations

import os
from pathlib import Path

from pydantic_acp import AdapterConfig, FileSystemProjectionMap, HookProjectionMap, run_acp
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks, ValidatedToolArgs
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import ToolDefinition

__all__ = ("agent", "config", "main")

# The workspace the demo tools are allowed to touch. Kept as a module-level
# global (rather than captured in a closure) so it can be pointed at a temporary
# directory from tests.
_TRAVEL_ROOT = Path("native-demo")

_ITINERARY = """# Travel Brief

- Day 1: land, drop bags, walk the old town.
- Day 2: day trip to the coast.
- Day 3: museums, then the night train home.
"""


def _ensure_travel_workspace() -> Path:
    """Create the trip workspace (and its starter itinerary) if it is missing."""
    _TRAVEL_ROOT.mkdir(parents=True, exist_ok=True)
    itinerary = _TRAVEL_ROOT / "itinerary.md"
    if not itinerary.exists():
        itinerary.write_text(_ITINERARY, encoding="utf-8")
    return _TRAVEL_ROOT


def _trip_path_parts(path: str) -> tuple[str, ...]:
    """Split a tool-supplied `path` into workspace-relative components.

    Anything that could name a file outside the workspace without touching the
    filesystem -- an absolute path, a drive letter, a `..` segment -- is rejected
    here, before any syscall runs.
    """
    candidate = Path(path)
    if candidate.is_absolute() or candidate.drive:
        raise ValueError(f"{path!r} is outside the trip workspace")
    parts = tuple(part for part in candidate.parts if part != ".")
    if not parts or ".." in parts:
        raise ValueError(f"{path!r} is outside the trip workspace")
    return parts


def _open_containing_dir(parts: tuple[str, ...], *, create: bool) -> int:
    """Open the directory that holds `parts[-1]` and return its descriptor.

    Each component is opened relative to the previous descriptor with
    `O_NOFOLLOW`, so the kernel -- not an earlier `resolve()` snapshot -- decides
    what the name refers to at the moment it is used. That closes the
    check-then-write race: a symlink swapped into the workspace after validation
    makes the `openat` fail instead of silently redirecting the operation
    outside the workspace. The caller owns the returned descriptor.
    """
    dir_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    dir_fd = os.open(_ensure_travel_workspace(), os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in parts[:-1]:
            if create:
                try:
                    os.mkdir(component, dir_fd=dir_fd)
                except FileExistsError:
                    pass
            child_fd = os.open(component, dir_flags, dir_fd=dir_fd)
            os.close(dir_fd)
            dir_fd = child_fd
    except BaseException:
        os.close(dir_fd)
        raise
    return dir_fd


hooks = Hooks()


@hooks.on.before_model_request
async def observe_before_model_request(
    ctx: RunContext[None],
    request_context: ModelRequestContext,
) -> ModelRequestContext:
    """Surface every model request to the ACP client without changing it."""
    del ctx
    return request_context


@hooks.on.before_tool_execute(tools=["write_trip_file"])
async def observe_write_tool(
    ctx: RunContext[None],
    *,
    call: ToolCallPart,
    tool_def: ToolDefinition,
    args: ValidatedToolArgs,
) -> ValidatedToolArgs:
    """Surface writes to the ACP client without changing the arguments."""
    del ctx, call, tool_def
    return args


agent = Agent(
    "openai:gpt-5",
    # Resolve the model on first run rather than at import, so this module can be
    # imported (and its tools inspected) without provider credentials present.
    defer_model_check=True,
    capabilities=[hooks],
    instructions=(
        "You are a travel assistant working in a small trip workspace. "
        "Use read_trip_file to look at existing notes and write_trip_file to record new ones."
    ),
)


@agent.tool_plain
def read_trip_file(path: str, max_chars: int = 4000) -> str:
    """Read a file from the trip workspace, truncated to `max_chars` characters."""
    if max_chars < 0:
        # A negative limit would slice off the *tail* instead of bounding the
        # prefix, quietly returning far more than the caller asked for.
        raise ValueError("max_chars must be non-negative")
    parts = _trip_path_parts(path)
    dir_fd = _open_containing_dir(parts, create=False)
    try:
        file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dir_fd)
    finally:
        os.close(dir_fd)
    with os.fdopen(file_fd, encoding="utf-8") as handle:
        return handle.read(max_chars)


@agent.tool_plain(requires_approval=True)
def write_trip_file(path: str, content: str) -> str:
    """Write a file into the trip workspace. Requires the client's approval."""
    parts = _trip_path_parts(path)
    dir_fd = _open_containing_dir(parts, create=True)
    try:
        file_fd = os.open(
            parts[-1],
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW,
            0o644,
            dir_fd=dir_fd,
        )
    finally:
        os.close(dir_fd)
    with os.fdopen(file_fd, "w", encoding="utf-8") as handle:
        handle.write(content)
    return f"wrote {len(content)} characters to {path}"


config = AdapterConfig(
    agent_name="travel-agent",
    agent_title="Travel Agent",
    projection_maps=(
        FileSystemProjectionMap(
            read_tool_names=frozenset({"read_trip_file"}),
            write_tool_names=frozenset({"write_trip_file"}),
            path_arg="path",
            content_arg="content",
        ),
    ),
    hook_projection_map=HookProjectionMap(
        # "Before Execute" reads better than the default "Before Tool" next to the
        # tool name this demo always shows in hook titles.
        event_labels={
            **HookProjectionMap().event_labels,
            "before_tool_execute": "Before Execute",
        },
    ),
)


def main() -> None:
    _ensure_travel_workspace()
    run_acp(agent=agent, config=config)


if __name__ == "__main__":
    main()
