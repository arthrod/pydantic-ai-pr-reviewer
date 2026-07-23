"""Shared ACP host-client doubles used by the test suite.

``pydantic_acp.testing.RecordingACPClient`` already implements the full ACP client
surface (permissions, filesystem, terminals) that an agent or an ``AcpHostBridge``
delegate is expected to answer, so the doubles here reuse it instead of
re-implementing every method.

The one behavioural difference is how session updates are recorded: the upstream
fake stores ``UpdateRecord`` dataclasses, while these tests unpack the recorded
updates as ``(session_id, update)`` pairs, so ``session_update`` is overridden to
record plain tuples.
"""

from __future__ import annotations as _annotations

from typing import Any

from pydantic_acp.testing import RecordingACPClient

__all__ = ("HostRecordingClient", "RecordingClient")


class RecordingClient(RecordingACPClient):
    """An ACP host client double that records ``(session_id, update)`` pairs.

    Used wherever a test needs to observe the session updates an agent (or an
    ``AcpHostBridge``) emits, and to script permission answers via
    ``queue_permission_selected`` / ``queue_permission_cancelled``.
    """

    updates: list[tuple[str, Any]]

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        del kwargs
        self.updates.append((session_id, update))


class HostRecordingClient(RecordingClient):
    """A host client double used for the host-side bridge surface.

    Behaves exactly like `RecordingClient`; it exists as a separate name so tests
    that exercise the filesystem/terminal delegation of `AcpHostBridge` (rather
    than its session-update recording) read clearly at the call site.
    """
