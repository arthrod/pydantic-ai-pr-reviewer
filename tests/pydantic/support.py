"""Helpers for the native Pydantic AI adapter tests.

Re-exports the ACP schema types, the adapter entry point and the recording host
client those tests use, so a test module needs a single import to drive an
adapter and inspect what it sent to the client.
"""

from __future__ import annotations as _annotations

from acp.helpers import text_block
from acp.schema import FileEditToolCallContent, ToolCallProgress, ToolCallStart
from pydantic_acp import create_acp_agent

from ..support import HostRecordingClient, RecordingClient

__all__ = (
    "FileEditToolCallContent",
    "HostRecordingClient",
    "RecordingClient",
    "ToolCallProgress",
    "ToolCallStart",
    "create_acp_agent",
    "text_block",
)
