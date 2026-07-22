# AGENTS.md

Guide for AI agents working in this repository.

## Project status: scaffold / work-in-progress

This repo is intended to become a **PR reviewer built on `pydantic-ai`**, exposed
via the **Agent Client Protocol (ACP)** through the `pydantic-acp` library.

As of the last read it is **not yet implemented**. Treat the layout as
intentional scaffolding, not broken code to "fix" without asking:

- `main.py` is a 6-line stub (`print("Hello from pydantic-ai-pr-reviewer!")`).
  There is no application logic, no agent, no PR-review code yet.
- `tests/test_acp_client_provider.py` and `tests/test_native_pydantic_agent.py`
  are large (~1,200 lines combined) and reference modules that **do not exist
  in this repo**:
  - `tests/support.py` (`HostRecordingClient`, `RecordingClient`)
  - `tests/pydantic/support.py` (`FileEditToolCallContent`,
    `ToolCallProgress`, `ToolCallStart`, `create_acp_agent`, `text_block`,
    `RecordingClient`)
  - `examples/pydantic/travel_agent.py` (loaded dynamically via
    `importlib.util.spec_from_file_location`)
- `tests/` has **no `__init__.py`**, but the test modules use relative imports
  (`from .support import ...`, `from .pydantic.support import ...`). Collection
  will fail with `ImportError: attempted relative import with no known parent
  package` until a package layout is added.
- `pytest` is **not declared in `pyproject.toml`** and is **not installed in
  `.venv`**. Running `uv run pytest` currently shells out to the **system
  Homebrew Python** (`/opt/homebrew/Cellar/python@3.14/...`), which does not see
  `pydantic_acp` — masking the real "pytest missing" error as
  `ModuleNotFoundError: No module named 'pydantic_acp'`.

The shape of the tests strongly suggests they were lifted from the
`pydantic-acp` library's own test suite to serve as a target spec for what this
project needs to provide. Do not assume they pass today.

## Toolchain

- **Python 3.14** required (`.python-version` pins `3.14`;
  `pyproject.toml` sets `requires-python = ">=3.14"`).
- **`uv`** is the package manager (`uv.lock` present, `.venv/` is the local
  venv). Use `uv` rather than pip/poetry/pipenv.
- Only one runtime dependency is declared: `pydantic-acp>=1.4.0`
  (`uv.lock` resolves this to `pydantic-acp==1.4.0`). Transitively this pulls
  in `pydantic-ai-slim`, `pydantic-graph`, `agent-client-protocol` (the `acp`
  namespace), `pydantic`, `httpx`, etc. No dev/test dependencies are declared
  yet — adding `pytest` (and likely `pytest-anyio`, since the tests are async)
  to a `[dependency-groups]` dev group is a prerequisite for running tests.

### Common commands

```bash
uv sync                          # install/refresh declared deps into .venv
uv run python main.py            # runs the stub
uv run python -c "import pydantic_acp; print(pydantic_acp.__version__)"
uv run pytest                    # WILL FAIL today (see "Project status")
```

If you need to inspect the installed `pydantic-acp` API (it is large and
non-obvious), prefer:

```bash
uv run python -c "import pydantic_acp; print([n for n in dir(pydantic_acp) if not n.startswith('_')])"
```

## Layout

```
.
├── main.py                 # stub entrypoint
├── pyproject.toml          # name + python pin + pydantic-acp dep (no dev deps)
├── uv.lock                 # resolved graph
├── .python-version         # 3.14
└── tests/
    ├── test_acp_client_provider.py   # exercises pydantic_acp.AcpProvider / AcpModel
    └── test_native_pydantic_agent.py # exercises create_acp_agent wrapping a PydanticAI Agent
```

Missing (referenced by tests, not present in repo):
- `tests/__init__.py`
- `tests/support.py`
- `tests/pydantic/__init__.py`, `tests/pydantic/support.py`
- `examples/__init__.py`, `examples/pydantic/__init__.py`,
  `examples/pydantic/travel_agent.py`

## Intended architecture (inferred from tests, not yet implemented)

Two integration surfaces from `pydantic-acp` are being targeted:

1. **Hosting an external ACP agent as a PydanticAI model** —
   `tests/test_acp_client_provider.py` constructs `AcpProvider` + `AcpModel`
   pairs and drives them with a hand-written `EchoACPAgent` test double
   implementing the ACP agent surface (`initialize`, `new_session`, `prompt`,
   `set_session_model`, `on_connect`, plus the lifecycle/extension hooks:
   `authenticate`, `cancel`, `fork_session`, `load_session`, `resume_session`,
   `list_sessions`, `ext_method`, `ext_notification`).

   **Gotcha**: the tests call `AcpProvider(agent=..., cwd=..., prompt_renderer=...)`
   with the `agent` keyword, but the **installed** `pydantic-acp==1.4.0`
   `AcpProvider.__init__` does **not** accept an `agent` kwarg (per pyrefly,
   pyright, and zuban diagnostics). Either the tests are pinned to a different
   version, or the constructor signature changed. Verify the signature against
   the installed version before "fixing" these call sites.

2. **Wrapping a native PydanticAI `Agent` as an ACP agent** —
   `tests/test_native_pydantic_agent.py` uses `create_acp_agent(agent=..., config=...)`
   and asserts that:
   - A `FunctionModel`-driven PydanticAI agent emits ACP
     `ToolCallStart` / `ToolCallProgress` updates.
   - A "before model" hook fires (titled
     `Hook Before Model (observe_before_model_request)`).
   - A "before execute" hook fires for write tools (titled
     `Hook Before Execute [write_trip_file] (observe_write_tool)`).
   - Tool calls are projected as `FileEditToolCallContent` diffs (old/new text)
     through a `RecordingClient`.
   - Permission selection via `client.queue_permission_selected("allow_once")`
     gates writes.

The test fixture `_TRAVEL_ROOT` / `_ensure_travel_workspace()` in
`examples/pydantic/travel_agent.py` is expected to create a sandboxed
`native-demo/` directory and persist files via `read_trip_file` /
`write_trip_file` tools.

## Conventions observed in tests (the only non-stub code today)

- `from __future__ import annotations as _annotations` as the first non-docstring
  line, with the alias named `_annotations` (not the conventional `annotations`).
- Keyword-only argument blocks signposted with bare `*` in `__init__`.
- Unused parameters explicitly discarded via `del <name>, <name>` at the top of
  the function body rather than leading-underscore renaming.
- Type hints use `Any | None`, `list[Any]`, `tuple[...]` throughout (Python 3.14
  union/PEP 585 syntax). `Literal` is used for closed enum-like strings (e.g.
  ACP `stop_reason`: `"end_turn" | "max_tokens" | "max_turn_requests" |
  "refusal" | "cancelled"`).
- `cast("Any", object())` is used to satisfy type checkers when invoking
  helper functions with fake `AgentInfo` arguments in tests.
- `# type: ignore[misc]` and `# pragma: no branch` are used sparingly and
  intentionally.
- Async test bodies are driven with explicit `asyncio.run(...)` calls rather
  than `pytest.mark.asyncio` / `pytest-asyncio` — suggesting the project should
  NOT add the `pytest-asyncio` plugin; instead, async tests are written
  synchronously and await via `asyncio.run`.

## Gotchas

- **`/Users/arthrod/T` is a symlink** to `/Users/arthrod/temp/T`. Tooling that
  resolves real paths (pytest's `rootdir`, some editors) will report paths under
  `/Users/arthrod/temp/T/...` — same files, different display path. Do not be
  fooled into thinking there are two copies.
- **No `[tool.pytest.ini_options]`** exists in `pyproject.toml`, yet `uv run
  pytest` reports `configfile: pyproject.toml`. The reported `rootdir` also
  points at the symlink-resolved path. If you add pytest config, add a
  `[tool.pytest.ini_options]` table and `pytest` to a dev dependency group.
- **LSP / type-checker noise**: pyrefly, pyright, and zuban all flag the test
  files (`AcpProvider.__init__` unexpected `agent` kwarg, unresolved
  `acp.helpers` / `acp.schema` / `tests.pydantic.support`). These are real
  signals about the scaffold-vs-installed-API mismatch, not lint to silence.
- The `.venv` was created by `uv` and is gitignored. `.python-version`,
  `uv.lock`, `pyproject.toml` are the source of truth — never edit `.venv`
  contents directly.
- `pydantic-acp` exposes a very large public API (`AcpHostBridge`,
  `AcpPromptRenderer`, `CapabilityBridge`, `ProjectionMap`, `ApprovalBridge`,
  `create_acp_model`, `run_acp`, `testing`, etc.). Browse
  `.venv/lib/python3.14/site-packages/pydantic_acp/` before assuming an import
  is missing — many symbols live in submodules (`pydantic_acp.client`,
  `pydantic_acp.host`, `pydantic_acp.testing`, `pydantic_acp.providers`).

## Before you start work

1. Run `uv sync` to ensure `.venv` reflects `uv.lock`.
2. Acknowledge that **`uv run pytest` cannot succeed** until: (a) `pytest` is
   added as a dev dependency, (b) `tests/__init__.py` and the missing
   `support.py` / `examples/pydantic/travel_agent.py` modules are created, and
   (c) the `AcpProvider(agent=...)` signature mismatch is reconciled with the
   installed `pydantic-acp` version. Confirm scope with the user before
   attempting all three.
3. Read the two test files end-to-end before writing any source — they are the
   closest thing to a spec this repo has.
