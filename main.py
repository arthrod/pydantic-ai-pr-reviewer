# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "pydantic-acp>=1.4.0",
#     "typer>=0.15.0",
#     "rich>=13.0.0",
#     "logfire>=3.0.0",
#     "pydantic>=2.0.0",
# ]
# ///

"""Tenancious PR Reviewer.

Autonomous PR reviewer that wraps an ACP agent (claude / cline / gemini /
opencode / goose) in a pydantic-ai ``Agent``, asks it to address every review
comment, then merges or closes the PR based on the structured
``TenanciousReviewerResult.final_decision``.

Designed for fully-autonomous, continuous operation: no human-in-the-loop
prompts. ACP permission requests are auto-allowed by :class:`LocalHostDelegate`.

Failure philosophy
------------------
Every failure path must say *why*. The previous revision printed a bare
``Fatal error:`` with an empty message because ``str(TimeoutError())`` is the
empty string. :func:`describe_exception` guarantees a non-empty description,
and :func:`_run_review` distinguishes "the ACP agent never produced text"
(a broken/unauthenticated agent) from ordinary model errors.
"""

from __future__ import annotations as _annotations

import asyncio
import inspect
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import logfire
import typer
from acp.interfaces import Agent as AcpAgent
from acp.schema import (
    AllowedOutcome,
    CreateTerminalResponse,
    EnvVariable,
    KillTerminalResponse,
    PermissionOption,
    ReadTextFileResponse,
    ReleaseTerminalResponse,
    RequestPermissionResponse,
    TerminalOutputResponse,
    ToolCallUpdate,
    WaitForTerminalExitResponse,
    WriteTextFileResponse,
)
from pydantic import BaseModel, Field
from pydantic_acp import AcpProvider
from pydantic_acp.command_agent import AcpCommandAgent, AcpCommandOptions
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities.hooks import Hooks
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.output import PromptedOutput
from rich.console import Console, Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

console = Console(highlight=False)
err_console = Console(stderr=True, highlight=False)


# ═══════════════════════════════════════════════════════════════════════
# Error description — never render an empty reason.
#
# ``str(TimeoutError())`` is ``""``. So is ``str(CancelledError())`` and a
# surprising number of asyncio/JSON-RPC errors. Printing ``f"{exc}"`` for
# those produces the infamous "failed with no indication of the reason".
# ═══════════════════════════════════════════════════════════════════════


def describe_exception(exc: BaseException) -> str:
    """Return a human-readable, guaranteed non-empty description of *exc*."""
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = str(current).strip()
        name = type(current).__name__
        parts.append(f"{name}: {text}" if text else name)
        current = current.__cause__ or current.__context__

    # Cap the chain so a deep context stack doesn't flood the terminal.
    return " ← caused by ".join(parts[:4])


def _diagnose_empty_turn(
    spec: ProviderSpec,
    delegate: LocalHostDelegate,
    exc: BaseException,
) -> str:
    """Explain a zero-text-chunk failure by its real underlying cause.

    An empty ACP turn is *usually* an unauthenticated or broken agent, but it
    is just as often a concrete provider error (rate limit, auth rejection,
    upstream API failure). Blindly printing "unauthenticated or ACP broken"
    when the chain clearly says "Rate limited" is actively misleading, so we
    classify the cause and lead the message with it.
    """
    detail = describe_exception(exc)
    lowered = detail.lower()
    tools = f"{delegate.tool_calls} tool call(s) seen"

    if any(s in lowered for s in ("rate limit", "rate_limit", "ratelimit", " 429", "too many requests")):
        head = (
            f"the {spec.key} provider's API rate-limited the request (transient). "
            f"Wait and retry, or use a different --provider."
        )
    elif any(
        s in lowered
        for s in ("unauthenticated", "authentication", "auth_required", "auth required",
                  "-32000", " 401", "unauthorized", "invalid api key", "invalid_api_key",
                  "not logged in", "no credentials", "requires re-authentication", "sign in")
    ):
        head = (
            f"the {spec.key} agent is not authenticated. Sign in or set its API key "
            f"(e.g. run `{spec.executable}` interactively once), then retry."
        )
    elif any(s in lowered for s in ("connection", "network", "all connection attempts failed", "econnrefused")):
        head = (
            f"the {spec.key} agent could not reach its model backend ({tools}). "
            f"Check that the backend/endpoint it is configured for is running."
        )
    elif any(s in lowered for s in ("exceeded maximum", "please return text", "without producing any text")):
        # No concrete error surfaced: the agent genuinely returned an empty
        # turn. This is the cline-style case — its ACP mode does not stream
        # agent output.
        head = (
            f"the {spec.key} ACP agent completed its turn without producing any text "
            f"output ({tools}) and reported no error — its ACP mode likely does not "
            f"stream agent output. Verify with `{shlex.join(spec.command)}` directly."
        )
    else:
        head = (
            f"the {spec.key} ACP agent produced no text output ({tools})."
        )
    return f"{head} Underlying error: {detail}"


# ═══════════════════════════════════════════════════════════════════════
# ACP provider registry
#
# These commands are the *only* thing that puts each CLI into ACP stdio
# mode. Launching the bare binary (e.g. ``cline``) starts its normal
# interactive/prompt mode, which never speaks JSON-RPC — the handshake
# then hangs until the review timeout with no diagnostic at all.
#
# ``crush`` is deliberately absent: it has no ACP mode whatsoever
# (``crush acp`` → 'Unknown command "acp"'), so it can never serve here.
# ═══════════════════════════════════════════════════════════════════════


ProviderName = Literal[
    "claude", "cline", "gemini", "opencode", "goose", "glm", "dirac", "vibe", "grok"
]


@dataclass(frozen=True, kw_only=True)
class ProviderSpec:
    """How to launch one ACP agent and drive it autonomously."""

    key: ProviderName
    command: tuple[str, ...]
    #: Environment overrides applied on top of ``os.environ`` for the child.
    env: Mapping[str, str] = field(default_factory=dict)
    #: Session permission modes to try, best-effort, in order. The first one
    #: the agent accepts wins. Picking a mode that routes permission requests
    #: to *us* is what makes autonomous operation possible.
    modes: tuple[str, ...] = ()
    description: str = ""

    @property
    def executable(self) -> str:
        return self.command[0]

    def available(self) -> str | None:
        """Return the resolved executable path, or ``None`` when missing."""
        return shutil.which(self.executable)


PROVIDERS: dict[ProviderName, ProviderSpec] = {
    # Claude Code via the official ACP adapter. Verified working end-to-end:
    # streams agent_message_chunk, emits tool_call updates, and routes tool
    # permissions to the ACP client.
    #
    # CLAUDECODE is blanked because the adapter refuses to start "inside
    # another Claude Code session". We are deliberately launching a separate
    # agent process, so the nested-session guard does not apply.
    "claude": ProviderSpec(
        key="claude",
        command=("npx", "-y", "@agentclientprotocol/claude-agent-acp"),
        env={"CLAUDECODE": "", "CLAUDE_CODE_ENTRYPOINT": "", "CLAUDE_CODE_SSE_PORT": ""},
        # "bypassPermissions" is preferred: it skips Claude Code's own policy
        # AND the Bash sandbox, so autonomous shell/gh/git never hit "blocked by
        # policy" (which fires *inside* the agent, before our delegate is asked
        # — especially when ~/.claude/settings.json sets defaultMode "dontAsk").
        # "acceptEdits"/"default" (ask us, we auto-allow) are fallbacks.
        modes=("bypassPermissions", "acceptEdits", "default"),
        description="Claude Code via @agentclientprotocol/claude-agent-acp",
    ),
    "cline": ProviderSpec(
        key="cline",
        command=("cline", "--acp"),
        modes=("act",),
        description="Cline CLI in ACP mode (--acp)",
    ),
    "gemini": ProviderSpec(
        key="gemini",
        # `--approval-mode yolo` auto-approves all tools (gemini has no ACP
        # session modes). NOTE: gemini currently rejects session/new for
        # individual accounts ("client no longer supported, migrate to
        # Antigravity"), so this is unusable until that changes.
        command=("gemini", "--acp", "--approval-mode", "yolo"),
        description="Gemini CLI in ACP mode (--acp, yolo approvals)",
    ),
    "opencode": ProviderSpec(
        key="opencode",
        command=("opencode", "acp"),
        description="opencode ACP server",
    ),
    "goose": ProviderSpec(
        key="goose",
        command=("goose", "acp"),
        # "auto" auto-approves; "smart_approve" is a safer fallback.
        modes=("auto", "smart_approve"),
        description="goose ACP agent over stdio",
    ),
    # The following three are dedicated `*-acp` stdio agents (not a `--acp`
    # flag on a TUI). Each was handshake-verified: initialize + session/new
    # succeed and they advertise real session modes. End-to-end output still
    # depends on the user having provider credentials (see auth notes).
    "glm": ProviderSpec(
        key="glm",
        # `glm-acp-agent` starts the ACP stdio loop directly (no subcommand).
        # Auth: needs Z_AI_API_KEY in the environment or creds stored via
        # `glm-acp-agent --setup`. Modes mirror Claude Code's.
        command=("glm-acp-agent",),
        # Prefer bypass_permissions for autonomous runs; fall back to ask-us.
        modes=("bypass_permissions", "accept_edits", "default"),
        description="Zhipu GLM via glm-acp-agent (ACP stdio)",
    ),
    "dirac": ProviderSpec(
        key="dirac",
        command=("dirac", "--acp"),
        # Auth: openai-codex-oauth (stored). "act" makes changes and routes
        # tool permissions to us; "yolo"/"auto" auto-approve as fallbacks.
        # "yolo" auto-approves every tool; "auto"/"act" are fallbacks.
        modes=("yolo", "auto", "act"),
        description="dirac in ACP mode (--acp)",
    ),
    "vibe": ProviderSpec(
        key="vibe",
        # `vibe-acp` (the ACP agent), NOT the Mistral `vibe` CLI, which has no
        # ACP mode. Advertised no auth methods in the handshake.
        command=("vibe-acp",),
        # "auto-approve" runs tools without prompting; others are fallbacks.
        modes=("auto-approve", "accept-edits", "default"),
        description="vibe via vibe-acp (ACP stdio)",
    ),
    "grok": ProviderSpec(
        key="grok",
        # The ACP stdio agent lives under `grok agent stdio` (NOT `grok --acp`,
        # which the TUI rejects). Auth: cached_token / grok.com (stored).
        # Advertises no session modes, so there is no permission mode to set;
        # our host delegate still auto-allows any permission requests.
        command=("grok", "agent", "--always-approve", "stdio"),
        modes=(),
        description="Grok via `grok agent stdio` (ACP stdio)",
    ),
}

#: Order tried when the requested provider is unavailable or fails to start.
#: cline is kept but is known to accept prompts yet emit no output in 3.0.46
#: (its --acp drops agent_message_chunk); the new *-acp agents come first.
FALLBACK_ORDER: tuple[ProviderName, ...] = (
    "claude", "glm", "dirac", "vibe", "grok", "cline", "opencode", "gemini", "goose"
)


# ═══════════════════════════════════════════════════════════════════════
# ACP delegate client — real filesystem + terminal access for the
# external ACP agent via the ACP host bridge.
#
# The ACP schema uses camelCase kwargs (e.g. ``optionId``, ``terminalId``).
# These are the authoritative names — do NOT attempt to pass snake_case
# variants; pydantic will silently discard them.
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class _ManagedTerminal:
    """Tracks a running terminal subprocess."""

    terminal_id: str
    process: asyncio.subprocess.Process
    command: str
    cwd: str | None = None
    stdout_chunks: list[str] = field(default_factory=list)
    stderr_chunks: list[str] = field(default_factory=list)
    exit_code: int | None = None
    done: asyncio.Event = field(default_factory=asyncio.Event)


class LocalHostDelegate:
    """An :class:`acp.interfaces.Client` that performs real filesystem I/O and
    terminal execution locally, auto-allows every permission request, and
    renders live agent progress to the console.

    The live rendering matters: without it an ACP run is a black box, and a
    silently-failing agent looks identical to a slow one.
    """

    def __init__(self, *, workspace_root: Path | None = None, verbose: bool = False) -> None:
        self.workspace_root = Path(workspace_root or Path.cwd()).resolve()
        self.verbose = verbose
        self._terminals: dict[str, _ManagedTerminal] = {}
        # Keep strong refs to background drain tasks so the asyncio GC
        # doesn't reap them mid-flight (ruff RUF006).
        self._drain_tasks: set[asyncio.Task[None]] = set()
        #: Counters used to explain an empty run after the fact.
        self.text_chunks = 0
        self.tool_calls = 0
        self.permissions_granted = 0
        self._text_buffer: list[str] = []
        #: tool_call_id → title, so completion updates can be labelled.
        self._tool_titles: dict[str, str] = {}

    # -- permission -------------------------------------------------

    async def request_permission(
        self,
        session_id: str,
        tool_call: ToolCallUpdate,
        options: list[PermissionOption],
        **kwargs: Any,
    ) -> RequestPermissionResponse:
        del session_id, kwargs
        # Prefer allow_always so repeated tool use doesn't round-trip.
        allow = (
            next((o for o in options if o.kind == "allow_always"), None)
            or next((o for o in options if o.kind == "allow_once"), None)
            or options[0]
        )
        self.permissions_granted += 1
        title = getattr(tool_call, "title", None) or getattr(tool_call, "tool_call_id", "?")
        console.print(f"  [dim green]✓ auto-allowed[/dim green] [dim]{title}[/dim]")
        # AllowedOutcome requires both ``option_id`` AND ``outcome="selected"``;
        # leaving ``outcome`` off raises a validation error at runtime.
        logfire.info("acp.permission_auto_allowed", option_id=allow.option_id)
        return RequestPermissionResponse(
            outcome=AllowedOutcome(option_id=allow.option_id, outcome="selected")
        )

    # -- session updates → live progress ----------------------------

    async def session_update(
        self,
        session_id: str,
        update: Any,
        **kwargs: Any,
    ) -> None:
        del session_id, kwargs
        kind = getattr(update, "session_update", None) or getattr(update, "sessionUpdate", None)

        if kind == "agent_message_chunk":
            content = getattr(update, "content", None)
            text = getattr(content, "text", None)
            if isinstance(text, str) and text:
                self.text_chunks += 1
                self._text_buffer.append(text)
                # Flush on sentence/newline boundaries to keep output readable.
                if "\n" in text or len("".join(self._text_buffer)) > 220:
                    self._flush_text()

        elif kind in ("tool_call", "tool_call_update"):
            status = getattr(update, "status", None)
            title = getattr(update, "title", None)
            call_id = getattr(update, "tool_call_id", None) or getattr(update, "toolCallId", None)

            # The title only rides along on the *first* update for a tool call;
            # the terminal completed/failed update carries the id and status but
            # no title. Remember titles by id or every tool line renders blank.
            if call_id and title:
                self._tool_titles[str(call_id)] = str(title)
            if kind == "tool_call":
                self.tool_calls += 1

            label = title or (self._tool_titles.get(str(call_id)) if call_id else None)
            if status in ("completed", "failed"):
                self._flush_text()
                colour = "green" if status == "completed" else "red"
                mark = "✓" if status == "completed" else "✗"
                console.print(
                    f"  [{colour}]{mark}[/{colour}] [dim]{str(label or 'tool')[:110]}[/dim]"
                )
                if call_id:
                    self._tool_titles.pop(str(call_id), None)
            elif label and self.verbose:
                console.print(f"  [dim]· {str(label)[:110]}[/dim]")

        elif kind == "agent_thought_chunk" and self.verbose:
            content = getattr(update, "content", None)
            text = getattr(content, "text", None)
            if isinstance(text, str) and text.strip():
                console.print(f"  [dim italic]{text.strip()[:160]}[/dim italic]")

    def _flush_text(self) -> None:
        buffered = "".join(self._text_buffer).strip()
        self._text_buffer.clear()
        if buffered:
            console.print(f"  [cyan]│[/cyan] {buffered}")

    def finish(self) -> None:
        """Flush any buffered agent text (call once the turn is over)."""
        self._flush_text()

    # -- filesystem -------------------------------------------------

    async def read_text_file(
        self,
        session_id: str,
        path: str,
        line: int | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> ReadTextFileResponse:
        del session_id, kwargs
        resolved = self._resolve(path)
        # ReadTextFileResponse has only one field: ``content: str``. There is
        # NO ``path`` or ``error`` field — errors must be encoded INTO the
        # content string, and ``content`` cannot be None.
        with logfire.span("acp.read_text_file", path=path) as span:
            if not resolved.is_file():
                span.set_attribute("found", False)
                return ReadTextFileResponse(content=f"ERROR: File not found: {path}")
            try:
                raw = resolved.read_text(encoding="utf-8")
            except Exception as exc:  # any read failure → content
                span.set_attribute("error", describe_exception(exc))
                return ReadTextFileResponse(content=f"ERROR: {describe_exception(exc)}")
            span.set_attribute("bytes", len(raw))
            content = raw
            if line is not None or limit is not None:
                lines = raw.splitlines()
                start = (line or 1) - 1
                end = None if limit is None else start + limit
                content = "\n".join(lines[start:end])
            return ReadTextFileResponse(content=content)

    async def write_text_file(
        self,
        session_id: str,
        path: str,
        content: str,
        **kwargs: Any,
    ) -> WriteTextFileResponse | None:
        del session_id, kwargs
        resolved = self._resolve(path)
        with logfire.span("acp.write_text_file", path=path, bytes=len(content)):
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
        # WriteTextFileResponse takes no fields (just _meta). Return None
        # rather than constructing an empty object.
        return None

    # -- terminal ---------------------------------------------------

    async def create_terminal(
        self,
        session_id: str,
        command: str,
        args: list[str] | None = None,
        env: list[EnvVariable] | None = None,
        cwd: str | None = None,
        output_byte_limit: int | None = None,
        **kwargs: Any,
    ) -> CreateTerminalResponse:
        del session_id, kwargs
        terminal_id = f"term-{len(self._terminals) + 1}"
        cmd_parts = [command, *(args or [])]
        run_cwd = self._resolve(cwd) if cwd else self.workspace_root

        merged_env = os.environ.copy()
        for ev in env or []:
            merged_env[ev.name] = ev.value

        with logfire.span(
            "acp.create_terminal",
            terminal_id=terminal_id,
            command=shlex.join(cmd_parts),
            cwd=str(run_cwd),
        ):
            proc = await asyncio.create_subprocess_exec(
                *cmd_parts,
                cwd=str(run_cwd),
                env=merged_env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=output_byte_limit or 10 * 1024 * 1024,
            )
        managed = _ManagedTerminal(
            terminal_id=terminal_id,
            process=proc,
            command=shlex.join(cmd_parts),
            cwd=str(run_cwd),
        )
        self._terminals[terminal_id] = managed
        task = asyncio.ensure_future(self._drain_terminal(managed))
        self._drain_tasks.add(task)
        task.add_done_callback(self._drain_tasks.discard)  # type: ignore[arg-type]
        console.print(f"  [dim]$ {shlex.join(cmd_parts)[:110]}[/dim]")
        logfire.info("acp.terminal_created", terminal_id=terminal_id)
        return CreateTerminalResponse(terminal_id=terminal_id)

    async def _drain_terminal(self, managed: _ManagedTerminal) -> None:
        proc = managed.process

        async def _read_stream(stream: asyncio.StreamReader | None, chunks: list[str]) -> None:
            if stream is None:
                return
            while True:
                try:
                    chunk = await stream.read(65536)
                except Exception:
                    break
                if not chunk:
                    break
                chunks.append(chunk.decode("utf-8", errors="replace"))

        stdout_task = asyncio.ensure_future(_read_stream(proc.stdout, managed.stdout_chunks))
        stderr_task = asyncio.ensure_future(_read_stream(proc.stderr, managed.stderr_chunks))
        self._drain_tasks.add(stdout_task)
        self._drain_tasks.add(stderr_task)
        stdout_task.add_done_callback(self._drain_tasks.discard)  # type: ignore[arg-type]
        stderr_task.add_done_callback(self._drain_tasks.discard)  # type: ignore[arg-type]
        await asyncio.wait([stdout_task, stderr_task], return_when=asyncio.ALL_COMPLETED)
        try:
            managed.exit_code = await proc.wait()
        except Exception:
            managed.exit_code = -1
        managed.done.set()
        logfire.info(
            "acp.terminal_exited",
            terminal_id=managed.terminal_id,
            exit_code=managed.exit_code,
            stdout_chars=sum(len(c) for c in managed.stdout_chunks),
            stderr_chars=sum(len(c) for c in managed.stderr_chunks),
        )

    async def terminal_output(
        self,
        session_id: str,
        terminal_id: str,
        **kwargs: Any,
    ) -> TerminalOutputResponse:
        del session_id, kwargs
        managed = self._terminals.get(terminal_id)
        # TerminalOutputResponse fields: ``output: str``, ``truncated: bool``.
        # There is NO ``terminal_id`` and NO ``error`` field — errors go INTO
        # ``output``, and ``truncated`` is required.
        if managed is None:
            return TerminalOutputResponse(
                output=f"ERROR: No terminal found with id {terminal_id}",
                truncated=False,
            )
        stderr_joined = "".join(managed.stderr_chunks)
        out = "".join(managed.stdout_chunks)
        if stderr_joined:
            out = f"{out}\n[stderr]\n{stderr_joined}" if out else stderr_joined
        return TerminalOutputResponse(output=out, truncated=False)

    async def wait_for_terminal_exit(
        self,
        session_id: str,
        terminal_id: str,
        **kwargs: Any,
    ) -> WaitForTerminalExitResponse:
        del session_id, kwargs
        managed = self._terminals.get(terminal_id)
        # WaitForTerminalExitResponse: ``exit_code: int | None``. No
        # ``terminal_id``, no ``error``. exit_code must be >= 0; use None when
        # the terminal is missing rather than a sentinel like -1.
        if managed is None:
            return WaitForTerminalExitResponse(exit_code=None)
        await managed.done.wait()
        return WaitForTerminalExitResponse(exit_code=managed.exit_code)

    async def release_terminal(
        self,
        session_id: str,
        terminal_id: str,
        **kwargs: Any,
    ) -> ReleaseTerminalResponse | None:
        del session_id, kwargs
        managed = self._terminals.pop(terminal_id, None)
        if managed is None:
            return None
        with suppress(Exception):
            managed.process.terminate()
        return None

    async def kill_terminal(
        self,
        session_id: str,
        terminal_id: str,
        **kwargs: Any,
    ) -> KillTerminalResponse | None:
        del session_id, kwargs
        managed = self._terminals.pop(terminal_id, None)
        if managed is None:
            return None
        with suppress(Exception):
            managed.process.kill()
        return None

    # -- session / misc ---------------------------------------------

    async def create_elicitation(self, message: str, mode: Any, **kwargs: Any) -> Any:
        del message, mode, kwargs
        raise RuntimeError("Elicitation not supported in autonomous mode")

    async def complete_elicitation(self, elicitation_id: str, **kwargs: Any) -> None:
        del elicitation_id, kwargs

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        del method, params
        return {}

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        del method, params

    def on_connect(self, conn: AcpAgent) -> None:
        del conn

    # -- helpers ----------------------------------------------------

    def _resolve(self, path: str | None) -> Path:
        if path is None:
            return self.workspace_root
        p = Path(path)
        if p.is_absolute():
            return p
        return self.workspace_root / p


# ═══════════════════════════════════════════════════════════════════════
# Structured output
# ═══════════════════════════════════════════════════════════════════════


class TenanciousReviewerResult(BaseModel):
    """Final structured result returned by the PR reviewer agent."""

    final_decision: Literal["merge", "close"] = Field(
        description="Whether to merge the PR (comments were addressed) or close it (unfixable).",
    )
    summary: str = Field(description="Concise description of what was done.")
    comments_addressed: list[str] = Field(
        default_factory=list,
        description="Each PR review comment that was addressed.",
    )
    commit_sha: str = Field(default="", description="SHA of the commit pushed to the PR branch.")
    commit_message: str = Field(default="", description="The commit message used.")
    reasoning: str = Field(description="Why this decision was reached.")


# ═══════════════════════════════════════════════════════════════════════
# Agent instructions
# ═══════════════════════════════════════════════════════════════════════

PR_REVIEWER_INSTRUCTIONS = """\
You are the Tenancious PR Reviewer. Your job: take a GitHub PR, read every \
review comment, implement the changes needed to address each comment, push \
the commit, and decide — merge or close?

## Workflow

1. **Read the PR and its review comments** with the `gh` CLI:
   - `gh pr view <N> --json title,body,headRefName,baseRefName,author,reviews,comments`
   - `gh pr diff <N>`
   - `gh pr checkout <N>` (puts you on the PR branch locally)

2. **Implement every review comment**. Treat each comment as a required change:
   - Edit the files referenced by each comment.
   - If a comment is a question, make the code self-documenting (rename, \
docstring, refactor) — do NOT just reply with text.
   - If the comments contradict each other or the PR intent, pick the most \
defensible interpretation and note it in `reasoning`.
   - Run tests or linters if any are present.

3. **Commit and push**:
   - `git add -A`
   - `git commit -m "<conventional message>"`
   - `git push` (NEVER force-push, NEVER amend someone else's commit)
   - If there is nothing to change, do not invent a commit — leave `commit_sha` empty.

4. **Decide**:
   - You addressed (or resolved) every comment → `final_decision = "merge"`.
   - The PR is fundamentally broken, comments are unaddressable, or the \
change should not land → `final_decision = "close"`.

## Output

Return a `TenanciousReviewerResult` as a JSON object:
- `final_decision`: `"merge"` or `"close"`
- `summary`: one paragraph of what you did
- `comments_addressed`: list each comment you addressed (by quote or file:line)
- `commit_sha`: run `git rev-parse HEAD` and put the SHA here
- `commit_message`: the commit message you used
- `reasoning`: why merge vs close

## Rules

- NEVER ask the user questions. Make reasonable autonomous decisions.
- NEVER force-push or rewrite history.
- Keep changes minimal and focused on the comments.
- The local filesystem and terminal are real — use them.
- You MUST finish by emitting the JSON object. Do not stop before that.
"""


def _build_review_prompt(pr_url: str, pr_number: int, repo_path: Path) -> str:
    return (
        f"Review and address PR {pr_url} (number #{pr_number}).\n"
        f"Work inside the local repository at {repo_path}.\n\n"
        f"Run `gh pr checkout {pr_number}` to get the branch, address every "
        f"review comment, commit, push, and return your final_decision as JSON."
    )


# ═══════════════════════════════════════════════════════════════════════
# ACP model construction + session bootstrap
# ═══════════════════════════════════════════════════════════════════════


class AcpStartupError(RuntimeError):
    """Raised when an ACP provider cannot be brought up to a usable session."""


async def _bootstrap_session(
    provider: AcpProvider,
    spec: ProviderSpec,
    *,
    startup_timeout: float,
) -> str:
    """Bring the ACP agent up to a live session and select a permission mode.

    This is the preflight the previous revision lacked. ``create_acp_model``
    only *describes* an agent — it spawns nothing — so wrapping it in
    ``try/except`` could never detect a broken provider, and the "fallback"
    was dead code. Here we actually perform ``initialize`` + ``session/new``,
    so an unusable provider fails fast and loudly instead of hanging until
    the review timeout.

    We prefer pydantic-acp's public bootstrap API — ``provider.ensure_session()``
    and ``provider.set_session_mode()`` — which perform ``initialize`` +
    ``session/new`` (authenticating if the agent demands it) and set the
    permission mode on the *same* session the prompt will reuse. Older library
    versions lack those methods, so we fall back to the private
    ``_ensure_session`` seam and ``provider.client.set_session_mode``.
    """
    ensure_public = getattr(provider, "ensure_session", None)
    ensure = ensure_public or getattr(provider, "_ensure_session", None)
    if ensure is None:  # pragma: no cover - library shape changed
        raise AcpStartupError(
            "pydantic-acp AcpProvider exposes neither ensure_session nor "
            "_ensure_session; cannot preflight the ACP session."
        )

    try:
        session_id = await asyncio.wait_for(
            ensure() if ensure_public else ensure(model_name=None),
            timeout=startup_timeout,
        )
    except TimeoutError as exc:
        raise AcpStartupError(
            f"timed out after {startup_timeout:.0f}s during ACP initialize/session-new. "
            f"`{shlex.join(spec.command)}` did not complete the handshake — it is "
            f"probably not an ACP stdio server or is waiting on interactive auth."
        ) from exc
    except Exception as exc:
        raise AcpStartupError(
            f"ACP handshake failed for `{shlex.join(spec.command)}`: {describe_exception(exc)}"
        ) from exc

    # Permission mode is best-effort: the right id differs per agent, and some
    # agents have no modes at all. Without it, Claude Code sessions inherit
    # `permissions.defaultMode` from settings.json — which for a "dontAsk"
    # setting denies every tool call and produces a silent no-op review.
    provider_set_mode = getattr(provider, "set_session_mode", None)
    for mode in spec.modes:
        try:
            if provider_set_mode is not None:
                await asyncio.wait_for(provider_set_mode(mode), timeout=30)
            else:
                client_set_mode = getattr(provider.client, "set_session_mode", None)
                if client_set_mode is None:
                    break
                await asyncio.wait_for(
                    client_set_mode(session_id=session_id, mode_id=mode), timeout=30
                )
        except Exception as exc:
            logfire.debug("acp.set_mode_failed", mode=mode, error=describe_exception(exc))
            continue
        logfire.info("acp.session_mode", mode=mode)
        console.print(f"[dim]  session mode: {mode}[/dim]")
        break

    return session_id


async def _start_provider(
    requested: ProviderName,
    repo_path: Path,
    delegate: LocalHostDelegate,
    *,
    startup_timeout: float,
) -> tuple[AcpProvider, ProviderSpec]:
    """Start the requested ACP provider, falling back through the registry.

    Unlike the previous revision, the fallback is real: each candidate is
    actually launched and handshaken before being accepted.
    """
    order: list[ProviderName] = [requested]
    order += [p for p in FALLBACK_ORDER if p != requested]

    failures: list[str] = []

    for key in order:
        spec = PROVIDERS[key]
        resolved = spec.available()
        if resolved is None:
            failures.append(f"{key}: `{spec.executable}` not on PATH")
            continue

        console.print(
            f"[dim]→ starting ACP provider [bold]{key}[/bold]: "
            f"{shlex.join(spec.command)}[/dim]"
        )

        agent_source = AcpCommandAgent(
            options=AcpCommandOptions(
                command=tuple(spec.command),
                cwd=repo_path,
                env=dict(spec.env) or None,
                # "discard": agent stderr would otherwise interleave with our
                # own Rich output and corrupt the display.
                stderr_mode="discard",
            ),
        )
        provider_kwargs: dict[str, Any] = {
            "acp_agent": agent_source,
            "host_client": delegate,
            "cwd": str(repo_path),
            "history_mode": "full",
        }
        # Turn an empty ACP turn into a legible ACP-specific error instead of
        # pydantic-ai's opaque "Exceeded maximum retries". Only available on
        # patched/newer pydantic-acp, so pass it conditionally.
        if "raise_on_empty_turn" in inspect.signature(AcpProvider).parameters:
            provider_kwargs["raise_on_empty_turn"] = True
        provider = AcpProvider(**provider_kwargs)

        try:
            await _bootstrap_session(provider, spec, startup_timeout=startup_timeout)
        except AcpStartupError as exc:
            failures.append(f"{key}: {exc}")
            console.print(f"[yellow]⚠ {key} unusable — {exc}[/yellow]")
            logfire.warning("provider_unavailable", provider=key, error=str(exc))
            with suppress(Exception):
                await provider.close()
            continue

        logfire.info("acp_provider_selected", provider=key, command=list(spec.command))
        console.print(f"[green]✓ ACP provider ready: [bold]{key}[/bold][/green]")
        return provider, spec

    detail = "\n  - ".join(failures) or "no providers configured"
    raise AcpStartupError(f"No usable ACP provider. Tried:\n  - {detail}")


# ═══════════════════════════════════════════════════════════════════════
# PR URL / number helpers
# ═══════════════════════════════════════════════════════════════════════


_PR_NUMBER_RE = re.compile(r"/pull/(\d+)(?:$|[/?#])")
_REPO_RE = re.compile(r"https?://github\.com/([^/]+/[^/]+)/pull/\d+")


def _extract_pr_number(pr_url: str) -> int:
    m = _PR_NUMBER_RE.search(pr_url)
    if not m:
        raise ValueError(f"Could not extract PR number from URL: {pr_url}")
    return int(m.group(1))


def _extract_repo_from_url(pr_url: str) -> str | None:
    m = _REPO_RE.match(pr_url)
    return m.group(1) if m else None


# ═══════════════════════════════════════════════════════════════════════
# git / gh helpers
#
# All PR state lives on the GitHub side, so `gh` is the only sane tool.
# (The previous revision had a `_merge_pr_gitpython` fast path that
# unconditionally raised ImportError to force the `gh` fallback — it did
# nothing but cost a network fetch, so it is gone.)
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class OpenPr:
    number: int
    url: str
    title: str
    head_sha: str
    is_draft: bool


def _run(cmd: list[str], cwd: Path | None = None, timeout: float = 120) -> tuple[int, str]:
    """Run a command, returning (rc, combined output). Never raises."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(cwd) if cwd else None,
            check=False,
            timeout=timeout,
        )
    except FileNotFoundError:
        return 127, f"command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout:.0f}s: {shlex.join(cmd)}"
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def list_open_prs(repo_path: Path) -> list[OpenPr]:
    """List open PRs with their head SHA (used to detect new pushes)."""
    rc, out = _run(
        [
            "gh", "pr", "list",
            "--state", "open",
            "--limit", "100",
            "--json", "number,url,title,headRefOid,isDraft",
        ],
        cwd=repo_path,
    )
    if rc != 0:
        err_console.print(f"[red]gh pr list failed (rc={rc}): {out}[/red]")
        return []
    try:
        raw = json.loads(out or "[]")
    except json.JSONDecodeError as exc:
        err_console.print(f"[red]gh pr list returned invalid JSON: {describe_exception(exc)}[/red]")
        return []
    return [
        OpenPr(
            number=item["number"],
            url=item["url"],
            title=item.get("title", ""),
            head_sha=item.get("headRefOid", ""),
            is_draft=bool(item.get("isDraft")),
        )
        for item in raw
    ]


def _gh_pr_command(args: list[str], repo_slug: str | None, cwd: Path) -> tuple[int, str]:
    cmd = ["gh", "pr", *args]
    if repo_slug:
        cmd += ["-R", repo_slug]
    return _run(cmd, cwd=cwd, timeout=180)


def merge_pr(pr_number: int, repo_slug: str | None, repo_path: Path) -> tuple[int, str]:
    """Merge a PR via `gh pr merge`, retrying with --admin on protection errors."""
    logfire.info("merge_via_gh", pr_number=pr_number)
    rc, out = _gh_pr_command(
        ["merge", str(pr_number), "--merge", "--delete-branch"], repo_slug, repo_path
    )
    if rc != 0 and ("not mergeable" in out.lower() or "protected" in out.lower()):
        console.print("[yellow]  merge blocked; retrying with --admin[/yellow]")
        rc, out = _gh_pr_command(
            ["merge", str(pr_number), "--merge", "--delete-branch", "--admin"],
            repo_slug,
            repo_path,
        )
    return rc, out


def close_pr(
    pr_number: int, comment: str, repo_slug: str | None, repo_path: Path
) -> tuple[int, str]:
    """Close a PR with an explanatory comment and delete its branch."""
    logfire.info("close_via_gh", pr_number=pr_number)
    return _gh_pr_command(
        ["close", str(pr_number), "--delete-branch", "--comment", comment],
        repo_slug,
        repo_path,
    )


def current_branch(repo_path: Path) -> str | None:
    rc, out = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_path, timeout=30)
    return out.strip() if rc == 0 else None


def restore_branch(repo_path: Path, branch: str | None) -> None:
    """Return the working tree to *branch*, discarding the agent's checkout.

    Continuous operation walks many PRs in one repo; each `gh pr checkout`
    leaves the tree on that PR's branch. Without this, PR #2 would be reviewed
    from PR #1's branch.
    """
    if not branch:
        return
    rc, out = _run(["git", "checkout", branch], cwd=repo_path, timeout=60)
    if rc != 0:
        err_console.print(f"[yellow]⚠ could not restore branch {branch}: {out}[/yellow]")


def working_tree_dirty(repo_path: Path) -> bool:
    rc, out = _run(["git", "status", "--porcelain"], cwd=repo_path, timeout=30)
    return rc == 0 and bool(out.strip())


# ═══════════════════════════════════════════════════════════════════════
# Processed-PR state (so `watch` doesn't re-review the same PR forever)
# ═══════════════════════════════════════════════════════════════════════


def detect_repo_slug(repo_path: Path) -> str | None:
    """Return ``owner/name`` for the repo, so state follows the repo not the path."""
    rc, out = _run(
        ["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
        cwd=repo_path,
        timeout=60,
    )
    return out.strip() if rc == 0 and out.strip() else None


def _state_path(repo_slug: str | None, repo_path: Path) -> Path:
    key = (repo_slug or str(repo_path.resolve())).replace("/", "_")
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    directory = base / "tenancious-pr-reviewer"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{key}.json"


def load_state(repo_slug: str | None, repo_path: Path) -> dict[str, str]:
    path = _state_path(repo_slug, repo_path)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_state(repo_slug: str | None, repo_path: Path, state: dict[str, str]) -> None:
    path = _state_path(repo_slug, repo_path)
    with suppress(OSError):
        path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════
# Rich message rendering
# ═══════════════════════════════════════════════════════════════════════


def _print_messages(messages: Sequence[ModelMessage]) -> None:
    """Render every ModelMessage emitted during the run."""
    console.rule(f"[bold cyan]Agent Messages ({len(messages)})[/bold cyan]")

    for i, msg in enumerate(messages, 1):
        if isinstance(msg, ModelRequest):
            rendered: list[str] = []
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    content = part.content
                    if not isinstance(content, str):
                        content = repr(content)
                    rendered.append(f"[blue]UserPromptPart[/blue]: {content}")
                else:
                    rendered.append(f"[dim]{type(part).__name__}[/dim]")
            body = "\n".join(rendered) or "[dim](empty)[/dim]"
            console.print(Panel(body, title=f"#{i} ModelRequest", border_style="blue"))

        elif isinstance(msg, ModelResponse):
            rendered = []
            for part in msg.parts:
                if isinstance(part, TextPart):
                    rendered.append(f"[green]TextPart[/green]: {part.content}")
                else:
                    rendered.append(f"[dim]{type(part).__name__}[/dim]")
            body = "\n".join(rendered) or "[dim](empty — agent produced no text)[/dim]"
            console.print(Panel(body, title=f"#{i} ModelResponse", border_style="green"))

        else:
            console.print(
                Panel(repr(msg), title=f"#{i} {type(msg).__name__}", border_style="dim")
            )

    console.rule()


# ═══════════════════════════════════════════════════════════════════════
# Hooks — observability during the run
# ═══════════════════════════════════════════════════════════════════════


def _make_hooks() -> Hooks:
    """Lifecycle hooks that trace the run to logfire.

    The signatures here are load-bearing and are NOT free-form: pydantic-ai
    dispatches them by keyword, and every hook must return the value it was
    given or the run dies at that point.

        before_model_request(ctx, request_context) -> ModelRequestContext
        before_tool_execute(ctx, *, call, tool_def, args) -> ValidatedToolArgs
        after_run(ctx, *, result) -> AgentRunResult

    Getting ``after_run`` wrong is especially cruel: the agent completes the
    entire PR review, then the run explodes on the very last callback with
    ``unexpected keyword argument 'result'`` and the work is thrown away.
    """
    hooks = Hooks()

    @hooks.on.before_model_request
    def _trace_request(ctx: RunContext[None], request_context: Any) -> Any:
        del ctx
        logfire.info("agent.model_request", messages=len(request_context.messages))
        return request_context

    @hooks.on.before_tool_execute
    def _trace_tool(
        ctx: RunContext[None],
        *,
        call: Any,
        tool_def: Any,
        args: Any,
    ) -> Any:
        del ctx, tool_def
        logfire.info("agent.tool_call", name=getattr(call, "tool_name", "?"))
        return args

    @hooks.on.after_run
    def _trace_run_end(ctx: RunContext[None], *, result: Any) -> Any:
        del ctx
        logfire.info("agent.run_complete")
        return result

    return hooks


# ═══════════════════════════════════════════════════════════════════════
# Core async review
# ═══════════════════════════════════════════════════════════════════════


class ReviewFailed(RuntimeError):
    """A review could not be completed — always carries a real reason."""


async def _run_review(
    pr_url: str,
    pr_number: int,
    repo_path: Path,
    provider_name: ProviderName,
    timeout_seconds: float,
    *,
    startup_timeout: float,
    verbose: bool,
) -> tuple[TenanciousReviewerResult, Sequence[ModelMessage]]:
    """Run the reviewer agent. Returns (result, all_messages)."""
    repo_path = Path(str(repo_path)).resolve()  # noqa: ASYNC240
    if not repo_path.is_dir():
        raise ReviewFailed(f"repository path does not exist: {repo_path}")

    delegate = LocalHostDelegate(workspace_root=repo_path, verbose=verbose)
    provider, spec = await _start_provider(
        provider_name, repo_path, delegate, startup_timeout=startup_timeout
    )

    try:
        model = provider.model(None, history_mode="full")

        agent = Agent(
            model,
            name="tenancious_pr_reviewer",
            # PromptedOutput (not the default ToolOutput) because the ACP
            # bridge has no result-tool mechanism. PromptedOutput still yields
            # a validated TenanciousReviewerResult on ``result.output``.
            output_type=PromptedOutput(TenanciousReviewerResult),
            instructions=PR_REVIEWER_INSTRUCTIONS,
            capabilities=[_make_hooks()],
            retries=2,
        )

        prompt = _build_review_prompt(pr_url, pr_number, repo_path)
        console.print(
            f"[cyan]Reviewing PR #{pr_number} via [bold]{spec.key}[/bold] "
            f"(timeout {timeout_seconds:.0f}s)[/cyan]"
        )
        started = time.monotonic()

        try:
            result = await asyncio.wait_for(agent.run(prompt), timeout=timeout_seconds)
        except TimeoutError as exc:
            delegate.finish()
            logfire.error("review_timeout", timeout=timeout_seconds)
            raise ReviewFailed(
                f"the review exceeded the {timeout_seconds:.0f}s timeout. "
                f"The {spec.key} agent produced {delegate.text_chunks} text chunk(s) and "
                f"{delegate.tool_calls} tool call(s) before the deadline. "
                f"Raise --timeout if the PR is genuinely large."
            ) from exc
        except Exception as exc:
            delegate.finish()
            # The single most confusing ACP failure: the agent completes its
            # turn but emits no agent_message_chunk at all. pydantic-ai then
            # reports "Exceeded maximum output retries", which says nothing
            # about the real cause. Translate it — but do NOT blindly blame
            # auth/ACP: the empty turn is often caused by a concrete error
            # (rate limit, auth, provider API error) that must lead the message.
            if delegate.text_chunks == 0:
                raise ReviewFailed(
                    _diagnose_empty_turn(spec, delegate, exc)
                ) from exc
            logfire.error("review_error", error=describe_exception(exc))
            raise ReviewFailed(describe_exception(exc)) from exc

        delegate.finish()
        elapsed = time.monotonic() - started
        console.print(
            f"[green]✓ Review complete in {elapsed:.0f}s[/green] "
            f"[dim]({delegate.text_chunks} text chunks, {delegate.tool_calls} tool calls, "
            f"{delegate.permissions_granted} permissions granted)[/dim]"
        )

        output = cast("TenanciousReviewerResult", result.output)
        logfire.info(
            "review_decision",
            final_decision=output.final_decision,
            comments=len(output.comments_addressed),
            sha=output.commit_sha,
        )
        return output, result.all_messages()
    finally:
        with suppress(Exception):
            await provider.close()


# ═══════════════════════════════════════════════════════════════════════
# Sync orchestrator: review → merge-or-close
# ═══════════════════════════════════════════════════════════════════════


def _do_review(
    pr_url: str,
    repo_path: Path,
    provider_name: ProviderName,
    timeout: float,
    *,
    startup_timeout: float = 120.0,
    dry_run: bool = False,
    verbose: bool = False,
) -> int:
    """Run the full pipeline for one PR. Returns 0 on success."""
    pr_number = _extract_pr_number(pr_url)
    repo_slug = _extract_repo_from_url(pr_url)
    original_branch = current_branch(repo_path)

    console.print(
        Panel.fit(
            f"[bold cyan]Tenancious PR Reviewer[/bold cyan]\n"
            f"PR: [bold]#{pr_number}[/bold]  •  Repo: [dim]{repo_path}[/dim]\n"
            f"Provider: [bold]{provider_name}[/bold]"
            + ("  •  [yellow]DRY RUN[/yellow]" if dry_run else ""),
            border_style="cyan",
        )
    )

    logfire.info("review_start", pr_url=pr_url, pr_number=pr_number, repo=str(repo_path))

    try:
        output, messages = asyncio.run(
            _run_review(
                pr_url,
                pr_number,
                repo_path,
                provider_name,
                timeout,
                startup_timeout=startup_timeout,
                verbose=verbose,
            )
        )
    except (ReviewFailed, AcpStartupError) as exc:
        # These already carry a full explanation.
        logfire.fatal("review_crashed", error=str(exc))
        err_console.print(f"[red]❌ PR #{pr_number} failed:[/red] {exc}")
        restore_branch(repo_path, original_branch)
        return 1
    except Exception as exc:
        logfire.fatal("review_crashed", error=describe_exception(exc))
        err_console.print(f"[red]❌ PR #{pr_number} failed:[/red] {describe_exception(exc)}")
        restore_branch(repo_path, original_branch)
        return 1

    if verbose:
        _print_messages(messages)

    _render_result(output)

    if dry_run:
        console.print(
            f"[yellow]DRY RUN: would {output.final_decision} PR #{pr_number}[/yellow]"
        )
        restore_branch(repo_path, original_branch)
        return 0

    if output.final_decision == "merge":
        console.print(f"[bold green]→ Merging PR #{pr_number}...[/bold green]")
        rc, out = merge_pr(pr_number, repo_slug, repo_path)
    else:
        comment = (
            f"Closing per tenancious reviewer: {output.summary}\n\n"
            f"Reasoning: {output.reasoning}"
        )
        console.print(f"[bold red]→ Closing PR #{pr_number}...[/bold red]")
        rc, out = close_pr(pr_number, comment, repo_slug, repo_path)

    restore_branch(repo_path, original_branch)

    if rc == 0:
        console.print(f"[green]✓ PR #{pr_number} {output.final_decision}d[/green]")
    else:
        err_console.print(
            f"[red]✗ gh {output.final_decision} failed for PR #{pr_number} (rc={rc}):[/red]\n{out}"
        )
    logfire.info("decision_executed", decision=output.final_decision, rc=rc, output=out)

    return 0 if rc == 0 else 1


def _render_result(result: TenanciousReviewerResult) -> None:
    """Pretty-print the structured result."""
    style = "green" if result.final_decision == "merge" else "red"
    icon = "✓" if result.final_decision == "merge" else "✗"

    console.print()
    console.rule(f"[bold {style}]TenanciousReviewerResult[/bold {style}]")

    console.print(
        Panel(
            f"[bold]{result.final_decision.upper()}[/bold]\n{result.summary}",
            title=f"[bold]{icon} final_decision: {result.final_decision}[/bold]",
            border_style=style,
        )
    )

    if result.comments_addressed:
        table = Table(title="Comments Addressed", border_style="dim blue")
        table.add_column("#", style="dim", width=4)
        table.add_column("Comment")
        for i, c in enumerate(result.comments_addressed, 1):
            table.add_row(str(i), c)
        console.print(table)

    body_parts: list[Any] = []
    if result.commit_message:
        body_parts.append(
            Panel(
                Syntax(result.commit_message, "markdown", theme="monokai"),
                title="[bold]Commit Message[/bold]",
                border_style="yellow",
            )
        )
    if result.commit_sha:
        body_parts.append(
            Panel(result.commit_sha, title="[bold]Commit SHA[/bold]", border_style="dim")
        )
    if result.reasoning:
        body_parts.append(
            Panel(result.reasoning, title="[bold]Reasoning[/bold]", border_style="dim")
        )
    if body_parts:
        console.print(Group(*body_parts))

    console.print()
    console.print(
        f"[{style}]●[/{style}] decision=[bold]{result.final_decision}[/bold]  "
        f"•  {len(result.comments_addressed)} comment(s) addressed"
    )


def _review_open_prs(
    repo_path: Path,
    provider_name: ProviderName,
    timeout: float,
    *,
    include_drafts: bool,
    dry_run: bool,
    verbose: bool,
    state: dict[str, str],
    repo_slug: str | None,
) -> tuple[int, int]:
    """Review every eligible open PR. Returns (processed, failures)."""
    prs = list_open_prs(repo_path)
    if not prs:
        console.print("[dim]No open PRs.[/dim]")
        return 0, 0

    eligible = [
        pr
        for pr in prs
        if (include_drafts or not pr.is_draft) and state.get(str(pr.number)) != pr.head_sha
    ]
    skipped = len(prs) - len(eligible)
    if skipped:
        console.print(f"[dim]Skipping {skipped} PR(s) (draft or already processed).[/dim]")
    if not eligible:
        return 0, 0

    console.print(f"[bold cyan]{len(eligible)} PR(s) to review.[/bold cyan]")
    failures = 0
    for pr in eligible:
        console.rule(f"[bold cyan]PR #{pr.number}: {pr.title}[/bold cyan]")

        if working_tree_dirty(repo_path):
            err_console.print(
                f"[red]✗ Skipping PR #{pr.number}: working tree is dirty. "
                f"Commit or stash your changes first.[/red]"
            )
            failures += 1
            continue

        code = _do_review(
            pr.url,
            repo_path,
            provider_name,
            timeout,
            dry_run=dry_run,
            verbose=verbose,
        )
        if code != 0:
            failures += 1
        else:
            # Record the head SHA so a re-run skips it until someone pushes.
            state[str(pr.number)] = pr.head_sha
            if not dry_run:
                save_state(repo_slug, repo_path, state)

    return len(eligible), failures


# ═══════════════════════════════════════════════════════════════════════
# Typer CLI
# ═══════════════════════════════════════════════════════════════════════

app = typer.Typer(
    name="pr-reviewer",
    help="Autonomous PR reviewer: implements review comments, then merges or closes.",
    add_completion=False,
)

_VERBOSE = {"on": False}

RepoOpt = Annotated[
    Path,
    typer.Option("--repo-path", "-r", help="Path to the local git repository"),
]
ProviderOpt = Annotated[
    ProviderName,
    typer.Option("--provider", "-p", help="ACP agent provider to use"),
]
TimeoutOpt = Annotated[
    float,
    typer.Option("--timeout", "-t", help="Maximum seconds for a single PR review"),
]
DryRunOpt = Annotated[
    bool,
    typer.Option("--dry-run", help="Review but never merge or close"),
]


@app.callback()
def _callback(
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable verbose / debug output")
    ] = False,
) -> None:
    _VERBOSE["on"] = verbose
    logfire.configure(
        send_to_logfire=False,
        console=logfire.ConsoleOptions(verbose=verbose) if verbose else False,
        min_level="debug" if verbose else "warn",
        # Logfire's f-string introspection cannot find call sites inside this
        # single-file script and warns noisily on every call. We pass explicit
        # kwargs everywhere, so introspection buys us nothing.
        inspect_arguments=False,
    )
    with suppress(Exception):
        logfire.instrument_pydantic_ai()


@app.command()
def review(
    pr_url: Annotated[
        str,
        typer.Argument(
            help="URL of the GitHub PR (e.g. https://github.com/org/repo/pull/42)",
            show_default=False,
        ),
    ],
    repo_path: RepoOpt = Path.cwd(),
    provider: ProviderOpt = "claude",
    timeout: TimeoutOpt = 900.0,
    dry_run: DryRunOpt = False,
) -> None:
    """Review one PR, implement its comments, then merge or close it."""
    code = _do_review(
        pr_url,
        repo_path.resolve(),
        provider,
        timeout,
        dry_run=dry_run,
        verbose=_VERBOSE["on"],
    )
    if code != 0:
        raise typer.Exit(code=code)


@app.command(name="review-all")
def review_all(
    repo_path: RepoOpt = Path.cwd(),
    provider: ProviderOpt = "claude",
    timeout: TimeoutOpt = 900.0,
    dry_run: DryRunOpt = False,
    include_drafts: Annotated[
        bool, typer.Option("--include-drafts", help="Also review draft PRs")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Re-review PRs even if already processed")
    ] = False,
) -> None:
    """Review every open PR in the repo, merging or closing each."""
    repo = repo_path.resolve()
    repo_slug = detect_repo_slug(repo)
    state = {} if force else load_state(repo_slug, repo)

    processed, failures = _review_open_prs(
        repo,
        provider,
        timeout,
        include_drafts=include_drafts,
        dry_run=dry_run,
        verbose=_VERBOSE["on"],
        state=state,
        repo_slug=repo_slug,
    )

    if failures:
        err_console.print(f"[red]✗ {failures}/{processed} PR(s) failed[/red]")
        raise typer.Exit(code=1)
    console.print(f"[green]✓ {processed} PR(s) processed[/green]")


@app.command()
def watch(
    repo_path: RepoOpt = Path.cwd(),
    provider: ProviderOpt = "claude",
    timeout: TimeoutOpt = 900.0,
    interval: Annotated[
        float, typer.Option("--interval", "-i", help="Seconds between polling cycles")
    ] = 300.0,
    dry_run: DryRunOpt = False,
    include_drafts: Annotated[
        bool, typer.Option("--include-drafts", help="Also review draft PRs")
    ] = False,
) -> None:
    """Continuously poll for open PRs and review each new one.

    A PR is reviewed once per head SHA: pushing new commits makes it eligible
    again. Failures never stop the loop.
    """
    repo = repo_path.resolve()
    repo_slug = detect_repo_slug(repo)
    state = load_state(repo_slug, repo)
    cycle = 0

    console.print(
        Panel.fit(
            f"[bold cyan]Watching for PRs[/bold cyan]\n"
            f"Repo: [dim]{repo}[/dim]\n"
            f"Provider: [bold]{provider}[/bold]  •  every {interval:.0f}s"
            + ("  •  [yellow]DRY RUN[/yellow]" if dry_run else ""),
            border_style="cyan",
        )
    )

    while True:
        cycle += 1
        console.rule(f"[dim]cycle {cycle}[/dim]")
        try:
            processed, failures = _review_open_prs(
                repo,
                provider,
                timeout,
                include_drafts=include_drafts,
                dry_run=dry_run,
                verbose=_VERBOSE["on"],
                state=state,
                repo_slug=repo_slug,
            )
            if processed:
                console.print(
                    f"[dim]cycle {cycle}: {processed} processed, {failures} failed[/dim]"
                )
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            # A cycle must never kill the watcher.
            err_console.print(f"[red]cycle {cycle} error:[/red] {describe_exception(exc)}")

        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            console.print("\n[dim]stopped[/dim]")
            return


@app.command()
def providers() -> None:
    """List ACP providers and whether their launcher is on PATH."""
    table = Table(title="ACP Providers", border_style="cyan")
    table.add_column("Provider", style="bold cyan")
    table.add_column("Launch command", style="dim")
    table.add_column("Status")
    table.add_column("Notes", style="dim")

    for key in FALLBACK_ORDER:
        spec = PROVIDERS[key]
        path = spec.available()
        status = "[green]✓ on PATH[/green]" if path else "[red]✗ not found[/red]"
        table.add_row(key, shlex.join(spec.command), status, spec.description)

    console.print(table)
    console.print(
        "\n[dim]`✓ on PATH` only means the launcher exists. Use "
        "`doctor` to actually handshake with an agent.[/dim]"
    )


@app.command()
def doctor(
    repo_path: RepoOpt = Path.cwd(),
    provider: ProviderOpt = "claude",
    startup_timeout: Annotated[
        float, typer.Option("--startup-timeout", help="Seconds allowed for the ACP handshake")
    ] = 120.0,
) -> None:
    """Diagnose the environment: gh auth, git state, and a real ACP handshake."""
    repo = repo_path.resolve()
    ok = True

    console.rule("[bold cyan]Environment[/bold cyan]")

    rc, out = _run(["gh", "auth", "status"], cwd=repo, timeout=60)
    if rc == 0:
        console.print("[green]✓ gh authenticated[/green]")
    else:
        ok = False
        err_console.print(f"[red]✗ gh not usable:[/red] {out}")

    rc, out = _run(["git", "rev-parse", "--is-inside-work-tree"], cwd=repo, timeout=30)
    if rc == 0 and out.strip() == "true":
        branch = current_branch(repo)
        dirty = working_tree_dirty(repo)
        console.print(f"[green]✓ git repo[/green] [dim](branch {branch})[/dim]")
        if dirty:
            console.print("[yellow]⚠ working tree is dirty — reviews will be skipped[/yellow]")
    else:
        ok = False
        err_console.print(f"[red]✗ not a git repository:[/red] {repo}")

    prs = list_open_prs(repo)
    console.print(f"[dim]open PRs: {len(prs)}[/dim]")
    for pr in prs:
        console.print(f"  [dim]#{pr.number} {pr.title} ({pr.head_sha[:8]})[/dim]")

    console.rule("[bold cyan]ACP handshake[/bold cyan]")

    async def _probe() -> None:
        delegate = LocalHostDelegate(workspace_root=repo)
        acp_provider, spec = await _start_provider(
            provider, repo, delegate, startup_timeout=startup_timeout
        )
        console.print(f"[green]✓ session established with {spec.key}[/green]")
        console.print(f"[dim]  session id: {acp_provider.session_id}[/dim]")
        with suppress(Exception):
            await acp_provider.close()

    try:
        asyncio.run(_probe())
    except Exception as exc:
        ok = False
        err_console.print(f"[red]✗ ACP handshake failed:[/red] {describe_exception(exc)}")

    console.print()
    if ok:
        console.print("[bold green]✓ ready[/bold green]")
    else:
        console.print("[bold red]✗ not ready — fix the items above[/bold red]")
        raise typer.Exit(code=1)


# ═══════════════════════════════════════════════════════════════════════
# Entrypoint
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app()
