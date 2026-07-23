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
"""

from __future__ import annotations as _annotations

from pathlib import Path

from pydantic_acp import AdapterConfig, FileSystemProjectionMap, HookProjectionMap, run_acp
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks, ValidatedToolArgs
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import ToolDefinition

__all__ = ("agent", "config", "main")


# The workspace lives next to the process working directory so an ACP client can
# watch the diffs land somewhere obvious; tests point it at a tmp_path instead.
_TRAVEL_ROOT = Path("native-demo")

_ITINERARY = """# Travel Brief

- Day 1: land, drop bags, walk the old town.
- Day 2: day trip to the coast.
- Day 3: museums, then the night train home.
"""


def _ensure_travel_workspace() -> Path:
    """Create the trip workspace (with a seed itinerary) and return its root."""
    _TRAVEL_ROOT.mkdir(parents=True, exist_ok=True)
    itinerary = _TRAVEL_ROOT / "itinerary.md"
    if not itinerary.exists():
        itinerary.write_text(_ITINERARY, encoding="utf-8")
    return _TRAVEL_ROOT


def _resolve_trip_path(path: str) -> Path:
    """Resolve `path` inside the trip workspace, refusing anything that escapes it."""
    root = _ensure_travel_workspace().resolve()
    resolved = (root / path).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{path!r} is outside the trip workspace")
    return resolved


hooks = Hooks()


@hooks.on.before_model_request()
async def observe_before_model_request(
    ctx: RunContext[None],
    request_context: ModelRequestContext,
) -> ModelRequestContext:
    """Pass the request through untouched; the adapter reports the hook to the client."""
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
    """Pass the write arguments through untouched, so the client can see the call."""
    del ctx, call, tool_def
    return args


agent = Agent(
    "openai:gpt-5",
    # The ACP client supplies the credentials, so defer the model check to keep
    # this module importable (and testable) without any provider configured.
    defer_model_check=True,
    capabilities=[hooks],
    instructions=(
        "You are a travel assistant working in a small trip workspace. "
        "Use read_trip_file to look at existing notes and write_trip_file "
        "to record new ones."
    ),
)


@agent.tool_plain
def read_trip_file(path: str, max_chars: int = 4000) -> str:
    """Read a file from the trip workspace."""
    content = _resolve_trip_path(path).read_text(encoding="utf-8")
    return content[:max_chars]


@agent.tool_plain(requires_approval=True)
def write_trip_file(path: str, content: str) -> str:
    """Write a file into the trip workspace (requires client approval)."""
    target = _resolve_trip_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
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
        # Keep the default hook labels and only shorten the tool-execute one, so
        # the client shows "Before Execute [write_trip_file] (observe_write_tool)".
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
