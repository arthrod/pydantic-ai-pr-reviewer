# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "pydantic-acp>=1.4.0",
#     "typer>=0.15.0",
#     "rich>=13.0.0",
#     "logfire>=3.0.0",
#     "pydantic>=2.0.0",
#     "gitpython>=3.1.0",
# ]
# ///

"""Tenancious PR Reviewer.

Autonomous PR reviewer that wraps an ACP agent (cline/crush) in a pydantic-ai
``Agent``, asks it to address every review comment, then merges or closes the
PR based on the structured ``TenanciousReviewerResult.final_decision``.

Designed for fully-autonomous operation: no human-in-the-loop prompts, no
confirmation steps. The pydantic-ai structured output is trusted blindly —
``result.output.final_decision`` drives the merge/close decision directly.
"""

from __future__ import annotations as _annotations

import asyncio
import os
import re
import shlex
import shutil
import subprocess
from collections.abc import Sequence
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
from pydantic_acp import create_acp_model
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
from rich.progress import Progress, SpinnerColumn, TaskID, TextColumn
from rich.syntax import Syntax
from rich.table import Table

console = Console(highlight=False, force_terminal=True)
err_console = Console(stderr=True, highlight=False, force_terminal=True)


# ═══════════════════════════════════════════════════════════════════════
# ACP delegate client — real filesystem + terminal access for the
# external ACP agent (cline / crush) via the ACP host bridge.
#
# The ACP schema uses camelCase kwargs (e.g. ``optionId``, ``terminalId``).
# These are the authoritative names — do NOT attempt to pass snake_case
# variants; pydantic will silently discard them and ty/ruff will flag it.
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
    """An :class:`acp.interfaces.Client` implementation that performs
    real filesystem I/O and terminal execution on the local machine.

    Passed to :func:`pydantic_acp.create_acp_model` so the external ACP
    agent (cline / crush) can read, write, and execute commands through
    the standard ACP host bridge.
    """

    def __init__(self, *, workspace_root: Path | None = None) -> None:
        self.workspace_root = Path(workspace_root or Path.cwd()).resolve()
        self._terminals: dict[str, _ManagedTerminal] = {}
        # Keep strong refs to background drain tasks so the asyncio GC
        # doesn't reap them mid-flight (ruff RUF006).
        self._drain_tasks: set[asyncio.Task[None]] = set()

    # -- permission -------------------------------------------------

    async def request_permission(
        self,
        session_id: str,
        tool_call: ToolCallUpdate,
        options: list[PermissionOption],
        **kwargs: Any,
    ) -> RequestPermissionResponse:
        del session_id, tool_call, kwargs
        # Auto-allow for autonomous PR review — no human in the loop.
        allow = next((o for o in options if o.kind == "allow_once"), options[0])
        # AllowedOutcome requires both ``option_id`` AND ``outcome="selected"``;
        # leaving ``outcome`` off will raise a validation error at runtime.
        logfire.info("acp.permission_auto_allowed", option_id=allow.option_id)
        return RequestPermissionResponse(
            outcome=AllowedOutcome(option_id=allow.option_id, outcome="selected")
        )

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
        # ReadTextFileResponse has only one field: ``content: str``.
        # There is NO ``path`` or ``error`` field — errors must be
        # encoded INTO the content string. ``content`` cannot be None.
        with logfire.span("acp.read_text_file", path=path) as span:
            if not resolved.is_file():
                span.set_attribute("found", False)
                return ReadTextFileResponse(content=f"ERROR: File not found: {path}")
            try:
                raw = resolved.read_text(encoding="utf-8")
            except Exception as exc:  # any read failure → content
                span.set_attribute("error", str(exc))
                return ReadTextFileResponse(content=f"ERROR: {exc}")
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

        # Span the subprocess creation so we can see what the ACP agent
        # is executing in Logfire / on stdout — without this the delegate
        # is a black box per the logfire-instrumentation skill.
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
        # Hold a strong ref to the drain task or ruff/asyncio will complain.
        task = asyncio.ensure_future(self._drain_terminal(managed))
        self._drain_tasks.add(task)
        task.add_done_callback(self._drain_tasks.discard)  # type: ignore[arg-type]
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
        # Emit a structured log line so the exit is visible in stdout /
        # Logfire — per logfire-instrumentation skill.
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
        # There is NO ``terminal_id`` and NO ``error`` field — errors must
        # go INTO ``output``. ``truncated`` is required.
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
        # WaitForTerminalExitResponse: ``exit_code: int | None``.
        # No ``terminal_id``, no ``error``. exit_code must be ≥ 0; use
        # None when the terminal is missing instead of a sentinel like -1.
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

    async def session_update(
        self,
        session_id: str,
        update: Any,
        **kwargs: Any,
    ) -> None:
        del session_id, update, kwargs

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
# Structured output — the single thing the agent must return.
# Pydantic AI guarantees the object; trust `result.output.final_decision`.
# ═══════════════════════════════════════════════════════════════════════


class TenanciousReviewerResult(BaseModel):
    """Final structured result returned by the PR reviewer agent.

    Usage::

        result = await agent.run(prompt)
        output = result.output                 # TenanciousReviewerResult
        final_decision = output.final_decision  # "merge" or "close"
    """

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
# Agent instructions (modern API: `instructions=` not `system_prompt=`)
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

4. **Decide**:
   - You addressed (or resolved) every comment → `final_decision = "merge"`.
   - The PR is fundamentally broken, comments are unaddressable, or the \
change should not land → `final_decision = "close"`.

## Output

Return a `TenanciousReviewerResult`:
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
"""


def _build_review_prompt(pr_url: str, pr_number: int, repo_path: Path) -> str:
    return (
        f"Review and address PR {pr_url} (number #{pr_number}).\n"
        f"Work inside the local repository at {repo_path}.\n\n"
        f"Run `gh pr checkout {pr_number}` to get the branch, address every "
        f"review comment, commit, push, and return your final_decision."
    )


# ═══════════════════════════════════════════════════════════════════════
# ACP model factory with provider fallback
# ═══════════════════════════════════════════════════════════════════════


ProviderName = Literal["cline", "crush"]


def _provider_command(provider: ProviderName) -> list[str]:
    """Return the CLI command tuple for a given ACP provider name."""
    return [provider]  # bare CLI name — must be on PATH


def _resolve_model(
    provider: ProviderName,
    repo_path: Path,
    delegate: LocalHostDelegate,
) -> tuple[Any, ProviderName]:
    """Return an ACP model, trying *provider* first then the other.

    No smoke test — the real ``agent.run`` will surface any provider error,
    and we don't want to pay for a round-trip just to validate the bridge.
    """
    fallback: ProviderName = "crush" if provider == "cline" else "cline"
    last_error: Exception | None = None

    for prov in (provider, fallback):
        try:
            cmd = _provider_command(prov)
            model = create_acp_model(
                acp_command=cmd,
                cwd=str(repo_path),
                delegate_client=delegate,
                history_mode="full",
            )
            logfire.info("acp_provider_selected", provider=prov, command=cmd)
            return model, prov
        except Exception as exc:
            last_error = exc
            logfire.warning("provider_unavailable", provider=prov, error=str(exc))
            console.print(f"[yellow]⚠ {prov} unavailable ({exc}), trying fallback...[/yellow]")

    raise RuntimeError(
        f"Neither {provider} nor {fallback} is available as an ACP agent. "
        f"Last error: {last_error}"
    ) from last_error


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
# Merge / close: GitPython first, `gh` as fallback
# ═══════════════════════════════════════════════════════════════════════


def _list_open_pr_urls(repo_path: Path) -> list[str]:
    """List open PR URLs. PR state lives on the GitHub side, so `gh` is used."""
    proc = subprocess.run(
        [
            "gh",
            "pr",
            "list",
            "--state",
            "open",
            "--json",
            "number,url",
            "-q",
            ".[] | .url",
        ],
        capture_output=True,
        text=True,
        cwd=str(repo_path),
        check=False,
    )
    if proc.returncode != 0:
        err_console.print(f"[red]gh pr list failed: {proc.stderr.strip()}[/red]")
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _gh_pr_command(args: list[str], repo_slug: str | None) -> tuple[int, str]:
    """Run a `gh pr ...` command. Returns (rc, combined_output)."""
    cmd = ["gh", "pr", *args]
    if repo_slug:
        cmd += ["-R", repo_slug]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def _merge_pr_gitpython(repo_path: Path, pr_branch: str | None) -> tuple[bool, str]:
    """Attempt a local merge via GitPython. Returns (success, message).

    Used as a fast path for self-hosted / offline scenarios where pushing
    directly to the base branch is acceptable. For GitHub-hosted repos with
    branch protection this will raise, and the caller falls back to ``gh``.
    """
    try:
        import git

        repo = git.Repo(str(repo_path))
        if pr_branch:
            repo.remotes.origin.fetch(pr_branch)
        # Without GitHub API auth, we can't reliably trigger a merge that
        # satisfies branch protection — defer to the `gh` fallback path.
        msg = "gitpython merge requires self-hosted base branch; deferring to gh"
        raise ImportError(msg)
    except Exception as exc:
        return False, str(exc)


def _merge_pr(
    pr_number: int,
    repo_slug: str | None,
    repo_path: Path,
    pr_branch: str | None = None,
) -> tuple[int, str]:
    """Merge PR. Tries GitPython local merge first, falls back to ``gh pr merge``.

    In practice GitHub-side merges always go through ``gh pr merge`` because
    a local merge + push to main is rarely what you want (no PR audit trail,
    no merge-commit metadata, potential branch-protection failures). The
    GitPython path exists for offline / self-hosted scenarios.
    """
    ok, _ = _merge_pr_gitpython(repo_path, pr_branch)
    if ok:
        logfire.info("merge_via_gitpython", pr_number=pr_number)
        return 0, "merged via gitpython"
    logfire.info("merge_via_gh", pr_number=pr_number)
    return _gh_pr_command(
        ["merge", str(pr_number), "--merge", "--delete-branch"], repo_slug
    )


def _close_pr(
    pr_number: int,
    comment: str,
    repo_slug: str | None,
) -> tuple[int, str]:
    """Close PR with a comment and delete the branch (gh fallback only).

    Closing a PR is a GitHub-side state change — there is no meaningful
    local-only equivalent, so we always use ``gh pr close``.
    """
    logfire.info("close_via_gh", pr_number=pr_number)
    return _gh_pr_command(
        ["close", str(pr_number), "--delete-branch", "--comment", comment], repo_slug
    )


# ═══════════════════════════════════════════════════════════════════════
# Rich message rendering — spit out every ModelMessage subclass
# ═══════════════════════════════════════════════════════════════════════


def _print_messages(messages: Sequence[ModelMessage]) -> None:
    """Render every ModelMessage emitted during the run using rich."""
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
            body = "\n".join(rendered) or "[dim](empty)[/dim]"
            console.print(Panel(body, title=f"#{i} ModelResponse", border_style="green"))

        else:
            console.print(
                Panel(
                    repr(msg),
                    title=f"#{i} {type(msg).__name__}",
                    border_style="dim",
                )
            )

    console.rule()


# ═══════════════════════════════════════════════════════════════════════
# Hooks — observability during the run (best practice from skill)
#
# Hook callbacks CAN be sync — pydantic-ai accepts both sync and async
# callables. Use sync unless you actually need to await; ruff will flag
# ``async def`` that never awaits (RUF029).
# ═══════════════════════════════════════════════════════════════════════


def _make_hooks() -> Hooks:
    """Build lifecycle hooks that trace the run to the rich console + logfire.

    Per the logfire-instrumentation skill: emit structured log lines that
    travel alongside the spans from ``logfire.instrument_pydantic_ai()``.
    """
    hooks = Hooks()

    @hooks.on.before_model_request
    def _trace_request(
        ctx: RunContext[None],
        request_context: Any,
    ) -> Any:
        del ctx
        n = len(request_context.messages)
        console.log(f"[dim]→ model request ({n} messages so far)[/dim]")
        logfire.info("agent.model_request", messages=n)
        return request_context

    @hooks.on.before_tool_execute
    def _trace_tool(
        ctx: RunContext[None],
        tool_name: str,
        args: Any,
    ) -> Any:
        del ctx
        console.log(f"[magenta]→ tool call: {tool_name}[/magenta]")
        logfire.info("agent.tool_call", name=tool_name)
        return args

    @hooks.on.after_run
    def _trace_run_end(ctx: RunContext[None]) -> None:
        del ctx
        logfire.info("agent.run_complete")
        console.log("[dim]← run complete[/dim]")

    return hooks


# ═══════════════════════════════════════════════════════════════════════
# Core async review
# ═══════════════════════════════════════════════════════════════════════


async def _run_review(
    pr_url: str,
    pr_number: int,
    repo_path: Path,
    provider: ProviderName,
    timeout_seconds: float,
) -> tuple[TenanciousReviewerResult, Sequence[ModelMessage]]:
    """Run the reviewer agent. Returns (result, all_messages).

    Trusts blindly that pydantic-ai returns a ``TenanciousReviewerResult``:
    no JSON parsing, no fallback, direct attribute access on the output.
    """
    repo_path = Path(str(repo_path)).resolve()  # noqa: ASYNC240
    if not repo_path.exists():
        repo_path.mkdir(parents=True, exist_ok=True)

    delegate = LocalHostDelegate(workspace_root=repo_path)
    model, used_provider = _resolve_model(provider, repo_path, delegate)

    # NOTE: ty infers the agent's output type from the ``output_type=`` argument.
    # When using PromptedOutput, the inferred type is the wrapper — but the
    # runtime ``.output`` is still the pydantic model. Leave the local
    # variable unannotated so ty doesn't complain about the wrapper mismatch.
    agent = Agent(
        model,
        name="tenancious_pr_reviewer",
        # PromptedOutput (not the default ToolOutput) because the ACP bridge
        # does not support the result-tool mechanism pydantic-ai uses to
        # enforce structured output. PromptedOutput still returns a fully
        # validated ``TenanciousReviewerResult`` on ``result.output``.
        output_type=PromptedOutput(TenanciousReviewerResult),
        instructions=PR_REVIEWER_INSTRUCTIONS,
        capabilities=[_make_hooks()],
        retries=2,
    )

    prompt = _build_review_prompt(pr_url, pr_number, repo_path)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task: TaskID = progress.add_task(
            f"[cyan]Reviewing PR #{pr_number} via [bold]{used_provider}[/bold]...",
            total=None,
        )

        try:
            result = await asyncio.wait_for(
                agent.run(prompt),
                timeout=timeout_seconds,
            )
            progress.update(task, description="[green]✓ Review complete[/green]")
        except TimeoutError:
            progress.update(task, description="[red]✗ Review timed out[/red]")
            logfire.error("review_timeout", timeout=timeout_seconds)
            raise
        except Exception as exc:
            progress.update(task, description=f"[red]✗ Review failed: {exc}[/red]")
            logfire.error("review_error", error=str(exc))
            raise

    # ── Blind trust: pydantic-ai returned the pydantic object. ──
    # ty can't see through PromptedOutput's generic wrapper, so it infers
    # `str` here. The runtime value IS a TenanciousReviewerResult.
    output = cast("TenanciousReviewerResult", result.output)
    final_decision = output.final_decision  # direct attribute access
    logfire.info(
        "review_decision",
        final_decision=final_decision,
        comments=len(output.comments_addressed),
        sha=output.commit_sha,
    )
    return output, result.all_messages()


# ═══════════════════════════════════════════════════════════════════════
# Sync orchestrator: review → merge-or-close
# ═══════════════════════════════════════════════════════════════════════


def _do_review(
    pr_url: str,
    repo_path: Path,
    provider: ProviderName,
    timeout: float,
) -> int:
    """Run the full pipeline for one PR. Returns process exit code."""
    pr_number = _extract_pr_number(pr_url)
    repo_slug = _extract_repo_from_url(pr_url)

    console.print(
        Panel.fit(
            f"[bold cyan]Tenancious PR Reviewer[/bold cyan]\n"
            f"PR: [bold]#{pr_number}[/bold]  •  Repo: [dim]{repo_path}[/dim]\n"
            f"Provider: [bold]{provider}[/bold]",
            border_style="cyan",
        )
    )

    logfire.info("review_start", pr_url=pr_url, pr_number=pr_number, repo=str(repo_path))

    try:
        output, messages = asyncio.run(
            _run_review(pr_url, pr_number, repo_path, provider, timeout)
        )
    except Exception as exc:
        logfire.fatal("review_crashed", error=str(exc))
        err_console.print(f"[red]❌ Fatal error: {exc}[/red]")
        return 1

    # ── Spit out every ModelMessage / ModelRequest / UserPromptPart /
    #    TextPart / ModelResponse with rich. ──
    _print_messages(messages)

    # ── Render the result. ──
    _render_result(output)

    # ── Execute the decision. No asking the user. ──
    if output.final_decision == "merge":
        console.print(f"[bold green]→ Merging PR #{pr_number}...[/bold green]")
        rc, out = _merge_pr(pr_number, repo_slug, repo_path)
    else:  # "close"
        comment = (
            f"Closing per tenancious reviewer: {output.summary}\n\n"
            f"Reasoning: {output.reasoning}"
        )
        console.print(f"[bold red]→ Closing PR #{pr_number}...[/bold red]")
        rc, out = _close_pr(pr_number, comment, repo_slug)

    if rc == 0:
        console.print("[green]✓ gh command succeeded[/green]")
    else:
        console.print(f"[red]✗ gh command failed (rc={rc}):[/red]\n{out}")
    logfire.info("decision_executed", decision=output.final_decision, rc=rc, output=out)

    return 0 if rc == 0 else 1


def _render_result(result: TenanciousReviewerResult) -> None:
    """Pretty-print the structured result using Rich."""
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
            Panel(
                result.commit_sha,
                title="[bold]Commit SHA[/bold]",
                border_style="dim",
            )
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


# ═══════════════════════════════════════════════════════════════════════
# Typer CLI
# ═══════════════════════════════════════════════════════════════════════

app = typer.Typer(
    name="pr-reviewer",
    help="Autonomous PR reviewer: implements comments, then merges or closes.",
    add_completion=False,
)


@app.callback()
def _callback(
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose / debug output"),
    ] = False,
) -> None:
    logfire.configure(
        send_to_logfire=False,
        console=logfire.ConsoleOptions(verbose=verbose),
        min_level="debug" if verbose else "info",
    )
    # Best practice per building-pydantic-ai-agents skill:
    # auto-trace every agent run, tool call, and model request.
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
    repo_path: Annotated[
        Path,
        typer.Option(
            "--repo-path", "-r",
            help="Path to the local git repository",
            exists=False,
        ),
    ] = Path.cwd(),
    provider: Annotated[
        ProviderName,
        typer.Option(
            "--provider", "-p",
            help="ACP agent provider to use (cline or crush)",
        ),
    ] = "cline",
    timeout: Annotated[
        float,
        typer.Option(
            "--timeout", "-t",
            help="Maximum time in seconds for the review",
        ),
    ] = 600.0,
) -> None:
    """Review a PR, implement comments, then merge or close it."""
    code = _do_review(pr_url, repo_path.resolve(), provider, timeout)
    if code != 0:
        raise typer.Exit(code=code)


@app.command(name="review-all")
def review_all(
    repo_path: Annotated[
        Path,
        typer.Option(
            "--repo-path", "-r",
            help="Path to the local git repository",
            exists=False,
        ),
    ] = Path.cwd(),
    provider: Annotated[
        ProviderName,
        typer.Option(
            "--provider", "-p",
            help="ACP agent provider to use (cline or crush)",
        ),
    ] = "cline",
    timeout: Annotated[
        float,
        typer.Option(
            "--timeout", "-t",
            help="Maximum time in seconds per review",
        ),
    ] = 600.0,
) -> None:
    """Review every open PR in the current repo, merge or close each."""
    urls = _list_open_pr_urls(repo_path.resolve())
    if not urls:
        console.print("[yellow]No open PRs found.[/yellow]")
        return

    console.print(f"[bold cyan]Found {len(urls)} open PR(s).[/bold cyan]")
    failures = 0
    for url in urls:
        console.rule(f"[bold cyan]PR: {url}[/bold cyan]")
        code = _do_review(url, repo_path.resolve(), provider, timeout)
        if code != 0:
            failures += 1

    if failures:
        console.print(f"[red]✗ {failures}/{len(urls)} PR(s) failed[/red]")
        raise typer.Exit(code=1)
    console.print(f"[green]✓ All {len(urls)} PR(s) processed[/green]")


@app.command()
def providers() -> None:
    """List available ACP providers and their status."""
    table = Table(title="ACP Providers", border_style="cyan")
    table.add_column("Provider", style="bold cyan")
    table.add_column("Path", style="dim")
    table.add_column("Status")

    for name in ("cline", "crush"):
        path = shutil.which(name)
        if path:
            table.add_row(name, path, "[green]✓ Available[/green]")
        else:
            table.add_row(name, "—", "[red]✗ Not found[/red]")

    console.print(table)


# ═══════════════════════════════════════════════════════════════════════
# Entrypoint
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app()
