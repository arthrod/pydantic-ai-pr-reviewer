from __future__ import annotations as _annotations

import tomllib
from pathlib import Path
from typing import Any, Literal, cast

import pydantic_acp
import pytest
from acp import PROTOCOL_VERSION
from acp.helpers import text_block
from acp.interfaces import Agent as AcpAgent
from acp.schema import (
    AgentCapabilities,
    AgentMessageChunk,
    AllowedOutcome,
    ClientCapabilities,
    Implementation,
    InitializeResponse,
    NewSessionResponse,
    PermissionOption,
    PromptResponse,
    ToolCallUpdate,
    Usage,
    UsageUpdate,
)
from pydantic_acp import AcpHostBridge, AcpModel, AcpProvider, create_acp_agent
from pydantic_acp import client as client_module
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    BinaryContent,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolReturnPart,
    UploadedFile,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.native_tools import WebSearchTool
from pydantic_ai.providers import Provider
from pydantic_ai.tools import ToolDefinition

from .support import HostRecordingClient, RecordingClient


class EchoACPAgent:  # type: ignore[misc]
    def __init__(
        self,
        *,
        stop_reason: Literal[
            "end_turn", "max_tokens", "max_turn_requests", "refusal", "cancelled",
        ] = "end_turn",
        usage: Usage | None = None,
    ) -> None:
        self.client: Any | None = None
        self.initialized_protocols: list[int] = []
        self.session_cwds: list[str] = []
        self.session_models: list[tuple[str, str]] = []
        self.prompts: list[tuple[str, str]] = []
        self.stop_reason: Literal[
            "end_turn", "max_tokens", "max_turn_requests", "refusal", "cancelled",
        ] = stop_reason
        self.usage = usage
        self.client_capabilities: ClientCapabilities | None = None
        self.client_info: Implementation | None = None
        self.mcp_servers_seen: list[list[Any] | None] = []

    def on_connect(self, conn: Any) -> None:
        self.client = conn

    async def authenticate(self, method_id: str, **kwargs: Any) -> Any:
        del method_id, kwargs
        return None

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        del session_id, kwargs

    async def close_session(self, session_id: str, **kwargs: Any) -> Any:
        del session_id, kwargs
        return None

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        del method, params
        return {}

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        del method, params

    async def fork_session(
        self, cwd: str, session_id: str, mcp_servers: list[Any] | None = None, **kwargs: Any,
    ) -> Any:
        del cwd, session_id, mcp_servers, kwargs
        return None

    async def list_sessions(
        self, cursor: str | None = None, cwd: str | None = None, **kwargs: Any,
    ) -> Any:
        del cursor, cwd, kwargs
        return None

    async def load_session(
        self, cwd: str, session_id: str, mcp_servers: list[Any] | None = None, **kwargs: Any,
    ) -> Any:
        del cwd, session_id, mcp_servers, kwargs
        return None

    async def resume_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        del cwd, session_id, mcp_servers, kwargs
        return None

    async def set_config_option(
        self, config_id: str, session_id: str, value: Any, **kwargs: Any,
    ) -> None:
        del config_id, session_id, value, kwargs

    async def set_session_mode(self, mode_id: str, session_id: str, **kwargs: Any) -> None:
        del mode_id, session_id, kwargs

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        del kwargs
        self.initialized_protocols.append(protocol_version)
        self.client_capabilities = client_capabilities
        self.client_info = client_info
        return InitializeResponse(
            protocol_version=protocol_version,
            agent_info=Implementation(name="echo-acp-agent", version="test"),
            agent_capabilities=AgentCapabilities(),
        )

    async def new_session(
        self,
        cwd: str,
        mcp_servers: list[Any] | None = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        del kwargs
        self.session_cwds.append(cwd)
        self.mcp_servers_seen.append(mcp_servers)
        return NewSessionResponse(session_id=f"session-{len(self.session_cwds)}")

    async def set_session_model(
        self,
        model_id: str,
        session_id: str,
        **kwargs: Any,
    ) -> None:
        del kwargs
        self.session_models.append((session_id, model_id))

    async def prompt(
        self,
        prompt: list[Any],
        session_id: str,
        message_id: str | None = None,
        **kwargs: Any,
    ) -> PromptResponse:
        del message_id, kwargs
        if self.client is None:
            raise AssertionError("ACP agent was not connected to a host client")
        rendered_prompt = "".join(str(getattr(block, "text", "")) for block in prompt)
        self.prompts.append((session_id, rendered_prompt))
        await self.client.session_update(
            session_id=session_id,
            update=AgentMessageChunk(
                session_update="agent_message_chunk",
                content=text_block(f"acp echo: {rendered_prompt}"),
            ),
            source="echo-acp-agent",
        )
        return PromptResponse(stop_reason=self.stop_reason, usage=self.usage)


def _build_provider_and_model(
    agent: Any,
    *,
    model_name: str = "zed-agent",
    cwd: str = "/workspace",
    prompt_renderer: Any = None,
) -> tuple[AcpProvider, AcpModel]:
    """Construct an ``AcpProvider``/``AcpModel`` pair with this file's shared test defaults."""
    provider = AcpProvider(agent=agent, cwd=cwd, prompt_renderer=prompt_renderer)
    model = AcpModel(model_name=model_name, provider=provider)
    return provider, model


def test_acp_client_provider_is_plain_pydantic_ai_provider() -> None:
    acp_agent = EchoACPAgent()
    provider, model = _build_provider_and_model(acp_agent)

    assert isinstance(provider, Provider)
    assert provider.client is acp_agent
    assert provider.name == "acp"
    assert provider.base_url == "acp://local"
    assert model.provider is provider
    assert model.system == "acp"
    assert model.model_name == "zed-agent"
    assert model.base_url == "acp://local"


async def test_pydantic_ai_agent_can_use_acp_as_just_a_provider() -> None:
    acp_agent = EchoACPAgent()
    _provider, model = _build_provider_and_model(acp_agent)
    agent = Agent(model)

    result = await agent.run("Summarize the ACP bridge")

    assert "Summarize the ACP bridge" in result.output
    assert acp_agent.initialized_protocols == [PROTOCOL_VERSION]
    assert acp_agent.session_cwds == ["/workspace"]
    assert acp_agent.session_models == [("session-1", "zed-agent")]
    assert len(acp_agent.prompts) == 1
    assert "Summarize the ACP bridge" in acp_agent.prompts[0][1]


def test_pydantic_acp_requires_pydantic_ai_v2() -> None:
    package_pyproject = Path("packages/adapters/pydantic-acp/pyproject.toml")
    data: dict[str, Any] = tomllib.loads(package_pyproject.read_text())
    dependencies: list[str] = data["project"]["dependencies"]
    pydantic_ai_dependency: str = next(
        dependency for dependency in dependencies if dependency.startswith("pydantic-ai-slim")
    )

    assert ">=2.0.0" in pydantic_ai_dependency
    assert "==1." not in pydantic_ai_dependency


# --- AcpProvider / AcpModel behavior (changed code) ---------------------------------


async def test_acp_provider_reuses_session_and_model_across_multiple_requests() -> None:
    acp_agent = EchoACPAgent()
    _provider, model = _build_provider_and_model(acp_agent)
    agent = Agent(model)

    await agent.run("first turn")
    await agent.run("second turn")

    assert acp_agent.initialized_protocols == [PROTOCOL_VERSION]
    assert acp_agent.session_cwds == ["/workspace"]
    assert acp_agent.session_models == [("session-1", "zed-agent")]
    assert len(acp_agent.prompts) == 2


def test_acp_provider_model_factory_uses_default_model_name() -> None:
    acp_agent = EchoACPAgent()
    provider = AcpProvider(agent=acp_agent, cwd="/workspace")

    model = provider.model()

    assert model.model_name == "agent"
    assert model.provider is provider


async def test_acp_provider_custom_prompt_renderer_is_used_instead_of_default() -> None:
    acp_agent = EchoACPAgent()
    seen_message_counts: list[int] = []

    def render(messages: Any, params: Any) -> list[Any]:
        del params
        seen_message_counts.append(len(messages))
        return [text_block("rendered by custom renderer")]

    _provider, model = _build_provider_and_model(acp_agent, prompt_renderer=render)
    agent = Agent(model)

    result = await agent.run("ignored by custom renderer")

    assert seen_message_counts == [1]
    assert acp_agent.prompts == [("session-1", "rendered by custom renderer")]
    assert "acp echo: rendered by custom renderer" in result.output


async def test_acp_provider_custom_prompt_renderer_supports_async_callables() -> None:
    acp_agent = EchoACPAgent()

    async def render(messages: Any, params: Any) -> list[Any]:
        del messages, params
        return [text_block("rendered async")]

    _provider, model = _build_provider_and_model(acp_agent, prompt_renderer=render)
    agent = Agent(model)

    await agent.run("ignored")

    assert acp_agent.prompts == [("session-1", "rendered async")]


async def test_acp_provider_prefers_prompt_response_usage_over_host_updates() -> None:
    acp_agent = EchoACPAgent(
        usage=Usage(
            input_tokens=11,
            output_tokens=7,
            cached_read_tokens=2,
            cached_write_tokens=1,
            thought_tokens=3,
            total_tokens=24,
        ),
    )
    _provider, model = _build_provider_and_model(acp_agent)
    agent = Agent(model)

    result = await agent.run("count my tokens")

    usage = result.usage
    assert usage.input_tokens == 11
    assert usage.output_tokens == 7
    assert usage.cache_read_tokens == 2
    assert usage.cache_write_tokens == 1
    assert usage.details.get("reasoning_tokens") == 3


@pytest.mark.parametrize(
    ("stop_reason", "expected_finish_reason"),
    [
        ("end_turn", "stop"),
        ("max_tokens", "length"),
        ("cancelled", "error"),
        ("refusal", "stop"),
        ("max_turn_requests", "stop"),
    ],
)
async def test_acp_provider_maps_acp_stop_reasons_to_finish_reasons(
    stop_reason: str,
    expected_finish_reason: str,
) -> None:
    acp_agent = EchoACPAgent(
        stop_reason=cast(
            "Literal['end_turn', 'max_tokens', 'max_turn_requests', 'refusal', 'cancelled']",
            stop_reason,
        ),
    )
    _provider, model = _build_provider_and_model(acp_agent)

    response = await model.request(
        [ModelRequest(parts=[UserPromptPart("hello")])],
        None,
        ModelRequestParameters(),
    )

    assert response.finish_reason == expected_finish_reason
    assert response.provider_details == {"acp_session_id": "session-1"}


async def test_acp_model_rejects_function_tools() -> None:
    acp_agent = EchoACPAgent()
    _provider, model = _build_provider_and_model(acp_agent)

    with pytest.raises(UserError, match="function tools"):
        await model.request(
            [ModelRequest(parts=[UserPromptPart("hello")])],
            None,
            ModelRequestParameters(function_tools=[ToolDefinition(name="do_thing")]),
        )
    assert acp_agent.prompts == []


async def test_acp_model_rejects_native_tools() -> None:
    # AcpModel advertises no supported native tools, so pydantic-ai's own
    # `Model.prepare_request` rejects the native tool before AcpModel's own
    # `_ensure_supported_request` guard is ever reached.
    acp_agent = EchoACPAgent()
    _provider, model = _build_provider_and_model(acp_agent)

    with pytest.raises(UserError, match="(?i)native tool"):
        await model.request(
            [ModelRequest(parts=[UserPromptPart("hello")])],
            None,
            ModelRequestParameters(native_tools=[WebSearchTool()]),
        )
    assert acp_agent.prompts == []


async def test_acp_model_rejects_disallowed_text_output() -> None:
    acp_agent = EchoACPAgent()
    _provider, model = _build_provider_and_model(acp_agent)

    with pytest.raises(UserError, match="text-response provider bridge"):
        await model.request(
            [ModelRequest(parts=[UserPromptPart("hello")])],
            None,
            ModelRequestParameters(allow_text_output=False),
        )
    assert acp_agent.prompts == []


async def test_acp_model_request_stream_yields_the_buffered_response_text() -> None:
    acp_agent = EchoACPAgent()
    _provider, model = _build_provider_and_model(acp_agent)

    async with model.request_stream(
        [ModelRequest(parts=[UserPromptPart("stream this")])],
        None,
        ModelRequestParameters(),
    ) as streamed_response:
        async for _ in streamed_response:
            pass
        final_response = streamed_response.get()

    assert len(final_response.parts) == 1
    assert isinstance(final_response.parts[0], TextPart)
    assert final_response.parts[0].content == "acp echo: stream this"
    assert streamed_response.finish_reason == "stop"


async def test_acp_provider_switches_session_model_when_model_name_changes() -> None:
    acp_agent = EchoACPAgent()
    provider = AcpProvider(agent=acp_agent, cwd="/workspace")
    first_model = AcpModel(model_name="model-a", provider=provider)
    second_model = AcpModel(model_name="model-b", provider=provider)

    await Agent(first_model).run("first turn")
    await Agent(second_model).run("second turn")

    # The ACP session is created once and reused; only the model selection changes.
    assert acp_agent.session_cwds == ["/workspace"]
    assert acp_agent.session_models == [("session-1", "model-a"), ("session-1", "model-b")]


async def test_acp_provider_forwards_client_capabilities_info_and_mcp_servers() -> None:
    acp_agent = EchoACPAgent()
    capabilities = ClientCapabilities()
    client_info = Implementation(name="custom-client", version="9.9.9")
    mcp_servers = [{"name": "demo"}]

    provider = AcpProvider(
        agent=acp_agent,
        cwd="/workspace",
        client_capabilities=capabilities,
        client_info=client_info,
        mcp_servers=mcp_servers,
    )
    model = AcpModel(model_name="zed-agent", provider=provider)

    await Agent(model).run("hello")

    assert acp_agent.client_capabilities is capabilities
    assert acp_agent.client_info is client_info
    assert acp_agent.mcp_servers_seen == [mcp_servers]


def test_acp_provider_model_profile_returns_the_shared_acp_profile() -> None:
    provider = AcpProvider(agent=EchoACPAgent(), cwd="/workspace")
    assert provider.model_profile("anything") is client_module.ACP_MODEL_PROFILE


def test_acp_model_supported_native_tools_is_empty() -> None:
    assert AcpModel.supported_native_tools() == frozenset()


class NoHandshakeACPAgent:  # type: ignore[misc]
    """An ACP agent that never receives a reverse connection via ``on_connect``.

    This models an ACP agent implementation that does not implement the
    optional ``on_connect`` hook. ``AcpProvider`` must still be constructible
    and usable; the agent simply never observes host updates, so the resulting
    model response carries no text.
    """

    async def authenticate(self, method_id: str, **kwargs: Any) -> Any:
        del method_id, kwargs
        return None

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        del session_id, kwargs

    async def close_session(self, session_id: str, **kwargs: Any) -> Any:
        del session_id, kwargs
        return None

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        del method, params
        return {}

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        del method, params

    async def fork_session(
        self, cwd: str, session_id: str, mcp_servers: list[Any] | None = None, **kwargs: Any,
    ) -> Any:
        del cwd, session_id, mcp_servers, kwargs
        return None

    async def list_sessions(
        self, cursor: str | None = None, cwd: str | None = None, **kwargs: Any,
    ) -> Any:
        del cursor, cwd, kwargs
        return None

    async def load_session(
        self, cwd: str, session_id: str, mcp_servers: list[Any] | None = None, **kwargs: Any,
    ) -> Any:
        del cwd, session_id, mcp_servers, kwargs
        return None

    async def resume_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        del cwd, session_id, mcp_servers, kwargs
        return None

    async def set_config_option(
        self, config_id: str, session_id: str, value: Any, **kwargs: Any,
    ) -> None:
        del config_id, session_id, value, kwargs

    async def set_session_mode(self, mode_id: str, session_id: str, **kwargs: Any) -> None:
        del mode_id, session_id, kwargs

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        del client_capabilities, client_info, kwargs
        return InitializeResponse(
            protocol_version=protocol_version,
            agent_info=Implementation(name="no-handshake-agent", version="test"),
            agent_capabilities=AgentCapabilities(),
        )

    async def new_session(
        self,
        cwd: str,
        mcp_servers: list[Any] | None = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        del cwd, mcp_servers, kwargs
        return NewSessionResponse(session_id="session-1")

    async def prompt(
        self,
        prompt: list[Any],
        session_id: str,
        message_id: str | None = None,
        **kwargs: Any,
    ) -> PromptResponse:
        del prompt, session_id, message_id, kwargs
        return PromptResponse(stop_reason="end_turn")  # type: ignore[arg-type]


async def test_acp_provider_does_not_require_the_agent_to_support_on_connect() -> None:
    acp_agent = NoHandshakeACPAgent()
    provider = AcpProvider(agent=acp_agent, cwd="/workspace")
    model = AcpModel(model_name="agent", provider=provider)

    response = await model.request(
        [ModelRequest(parts=[UserPromptPart("hello")])],
        None,
        ModelRequestParameters(),
    )

    assert response.parts == []
    assert response.finish_reason == "stop"
    assert response.provider_details == {"acp_session_id": "session-1"}


# --- AcpHostBridge behavior (changed code) -------------------------------------------


async def test_host_bridge_records_updates_without_a_delegate() -> None:
    bridge = AcpHostBridge()

    await bridge.session_update(
        session_id="session-1",
        update=AgentMessageChunk(session_update="agent_message_chunk", content=text_block("hi")),
    )

    assert len(bridge.updates) == 1
    assert bridge.updates[0].session_id == "session-1"
    assert bridge.updates[0].source is None
    assert bridge.agent_message_text_since(0, session_id="session-1") == "hi"


async def test_host_bridge_forwards_session_update_to_delegate_when_present() -> None:
    delegate = RecordingClient()
    bridge = AcpHostBridge(delegate=delegate)
    update = AgentMessageChunk(session_update="agent_message_chunk", content=text_block("hi"))

    await bridge.session_update(session_id="session-1", update=update)

    assert bridge.updates[0].update is update
    assert delegate.updates == [("session-1", update)]


@pytest.mark.parametrize(
    "call",
    [
        lambda bridge: bridge.request_permission(
            options=[PermissionOption(kind="allow_once", name="Allow", option_id="allow")],
            session_id="session-1",
            tool_call=ToolCallUpdate(tool_call_id="call-1"),
        ),
        lambda bridge: bridge.write_text_file(
            content="data", path="/tmp/f", session_id="session-1",
        ),
        lambda bridge: bridge.read_text_file(path="/tmp/f", session_id="session-1"),
        lambda bridge: bridge.create_terminal(command="ls", session_id="session-1"),
        lambda bridge: bridge.ext_method(method="custom/thing", params={}),
    ],
)
async def test_host_bridge_raises_without_a_delegate_for_host_methods(call: Any) -> None:
    bridge = AcpHostBridge()

    with pytest.raises(RuntimeError, match="no host client delegate"):
        await call(bridge)


async def test_host_bridge_ext_notification_is_a_noop_without_a_delegate() -> None:
    bridge = AcpHostBridge()

    await bridge.ext_notification(method="custom/notify", params={})


async def test_host_bridge_delegates_filesystem_and_terminal_calls_to_host_client() -> None:
    delegate = HostRecordingClient()
    bridge = AcpHostBridge(delegate=delegate)

    write_response = await bridge.write_text_file(
        content="hello", path="/tmp/f", session_id="session-1",
    )
    read_response = await bridge.read_text_file(path="/tmp/f", session_id="session-1")
    terminal_response = await bridge.create_terminal(command="ls", session_id="session-1")

    assert write_response is delegate.write_response
    assert read_response.content == "file:/tmp/f:None:None"
    assert terminal_response.terminal_id == "terminal-1"
    assert delegate.write_calls == [("session-1", "/tmp/f", "hello")]
    assert delegate.read_calls == [("session-1", "/tmp/f", None, None)]


async def test_host_bridge_delegates_request_permission_to_host_client() -> None:
    delegate = RecordingClient()
    delegate.queue_permission_selected("allow")
    bridge = AcpHostBridge(delegate=delegate)

    response = await bridge.request_permission(
        options=[PermissionOption(kind="allow_once", name="Allow", option_id="allow")],
        session_id="session-1",
        tool_call=ToolCallUpdate(tool_call_id="call-1"),
    )

    assert isinstance(response.outcome, AllowedOutcome)
    assert response.outcome.option_id == "allow"


async def test_host_bridge_on_connect_forwards_to_delegate_when_supported() -> None:  # type: ignore[misc]
    delegate = RecordingClient()
    bridge = AcpHostBridge(delegate=delegate)
    connected: list[Any] = []

    def on_connect_handler(conn: Any) -> None:
        connected.append(conn)

    cast("Any", delegate).on_connect = on_connect_handler

    sentinel_agent = cast("AcpAgent", object())
    bridge.on_connect(sentinel_agent)

    assert connected == [sentinel_agent]


async def test_host_bridge_delegates_terminal_lifecycle_calls_to_host_client() -> None:
    delegate = HostRecordingClient()
    bridge = AcpHostBridge(delegate=delegate)

    output = await bridge.terminal_output(session_id="session-1", terminal_id="terminal-1")
    release = await bridge.release_terminal(session_id="session-1", terminal_id="terminal-1")
    wait = await bridge.wait_for_terminal_exit(session_id="session-1", terminal_id="terminal-1")
    kill = await bridge.kill_terminal(session_id="session-1", terminal_id="terminal-1")

    assert output.output == "terminal-output"
    assert release is delegate.release_response
    assert wait.exit_code == 0
    assert kill is delegate.kill_response
    assert delegate.output_calls == [("session-1", "terminal-1")]
    assert delegate.release_calls == [("session-1", "terminal-1")]
    assert delegate.wait_calls == [("session-1", "terminal-1")]
    assert delegate.kill_calls == [("session-1", "terminal-1")]


class ExtensionRecordingClient(RecordingClient):
    """A host client double that actually answers ACP extension calls."""

    def __init__(self) -> None:
        super().__init__()
        self.ext_method_calls: list[tuple[str, dict[str, Any]]] = []
        self.ext_notification_calls: list[tuple[str, dict[str, Any]]] = []

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.ext_method_calls.append((method, params))
        return {"ok": True}

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        self.ext_notification_calls.append((method, params))


async def test_host_bridge_delegates_extension_calls_to_host_client() -> None:
    delegate = ExtensionRecordingClient()
    bridge = AcpHostBridge(delegate=delegate)

    result = await bridge.ext_method(method="custom/thing", params={"x": 1})
    await bridge.ext_notification(method="custom/notify", params={"y": 2})

    assert result == {"ok": True}
    assert delegate.ext_method_calls == [("custom/thing", {"x": 1})]
    assert delegate.ext_notification_calls == [("custom/notify", {"y": 2})]


# --- Pure helper functions (changed code) --------------------------------------------


def test_usage_from_acp_extracts_token_counts_and_reasoning_tokens() -> None:
    usage = client_module._usage_from_acp(
        Usage(
            input_tokens=10,
            output_tokens=5,
            cached_read_tokens=2,
            cached_write_tokens=1,
            thought_tokens=4,
            total_tokens=18,
        ),
    )

    assert usage.input_tokens == 10
    assert usage.output_tokens == 5
    assert usage.cache_read_tokens == 2
    assert usage.cache_write_tokens == 1
    assert usage.details == {"reasoning_tokens": 4}


def test_usage_from_acp_returns_empty_usage_for_none() -> None:
    usage = client_module._usage_from_acp(None)

    assert not usage.has_values()


def test_default_render_prompt_blocks_covers_system_tool_and_retry_parts() -> None:
    request = ModelRequest(
        parts=[
            SystemPromptPart("Be terse."),
            UserPromptPart("What is the status?"),
            ToolReturnPart(tool_name="check_status", content="ok", tool_call_id="call-1"),
            RetryPromptPart(
                content="please retry", tool_name="check_status", tool_call_id="call-1",
            ),
        ],
    )

    blocks = client_module._default_render_prompt_blocks([request], ModelRequestParameters())

    assert len(blocks) == 1
    rendered = blocks[0].text
    assert "System:\nBe terse." in rendered
    assert "What is the status?" in rendered
    assert "Tool result: check_status" in rendered
    assert "please retry" in rendered


def test_default_render_prompt_blocks_covers_media_user_content() -> None:
    request = ModelRequest(
        parts=[
            UserPromptPart(
                content=[
                    "Look at this:",
                    ImageUrl(url="https://example.com/x.png"),
                    BinaryContent(data=b"abc", media_type="text/plain"),
                    UploadedFile(file_id="file-1", provider_name="openai"),
                ],
            ),
        ],
    )

    blocks = client_module._default_render_prompt_blocks([request], ModelRequestParameters())

    rendered = blocks[0].text
    assert "Look at this:" in rendered
    assert "[image-url:https://example.com/x.png]" in rendered
    assert "[binary:text/plain:3 bytes]" in rendered
    assert "[uploaded-file:openai:file-1]" in rendered


def test_default_render_prompt_blocks_falls_back_to_message_text_without_a_request() -> None:
    # No `ModelRequest` is present, so rendering must fall back to `message.text`
    # instead of the request-part renderer.
    response = ModelResponse(parts=[TextPart("previous turn output")])

    blocks = client_module._default_render_prompt_blocks([response], ModelRequestParameters())

    assert len(blocks) == 1
    assert blocks[0].text == "previous turn output"


def test_default_render_prompt_blocks_returns_nothing_for_empty_messages() -> None:
    assert client_module._default_render_prompt_blocks([], ModelRequestParameters()) == []


def test_is_agent_message_chunk_supports_duck_typed_updates() -> None:
    class FakeChunkLike:
        session_update = "agent_message_chunk"

    assert client_module._is_agent_message_chunk(FakeChunkLike()) is True
    assert client_module._is_agent_message_chunk(object()) is False


# --- Prior (pre-existing) server adapter behavior path --------------------------------


async def test_prior_server_adapter_direction_still_works_standalone() -> None:
    """Regression guard: the original ACP server adapter direction is unaffected.

    This PR adds the inverse client/provider bridge, but the original
    ``create_acp_agent`` direction (exposing a ``pydantic_ai.Agent`` over ACP)
    must keep working exactly as it did before this change.
    """
    adapter = create_acp_agent(agent=Agent(TestModel(custom_output_text="Hello from ACP")))
    client = RecordingClient()
    adapter.on_connect(client)

    await adapter.initialize(protocol_version=PROTOCOL_VERSION)
    new_session_response = await adapter.new_session(cwd="/workspace", mcp_servers=[])
    prompt_response = await adapter.prompt(
        prompt=[text_block("Summarize the change.")],
        session_id=new_session_response.session_id,
        message_id="user-message-1",
    )

    assert prompt_response.stop_reason == "end_turn"
    agent_updates = [
        update for _, update in client.updates if isinstance(update, AgentMessageChunk)
    ]
    assert "".join(chunk.content.text for chunk in agent_updates) == "Hello from ACP"


async def test_prior_server_adapter_can_be_consumed_through_new_client_provider() -> None:
    """The new AcpProvider/AcpModel bridge composes with the pre-existing server adapter.

    A Pydantic AI agent is exposed via the original ``create_acp_agent`` adapter
    (prior, unchanged behavior) and then that very adapter is wrapped by the new
    ``AcpProvider`` (this PR's change) so a *second* Pydantic AI agent can run
    against it as a plain provider/model. This proves both directions interoperate.
    """
    inner_model = TestModel(custom_output_text="Hello from ACP")
    server_adapter = create_acp_agent(agent=Agent(inner_model))

    _provider, model = _build_provider_and_model(server_adapter, model_name=inner_model.model_name)
    outer_agent = Agent(model)

    result = await outer_agent.run("Ping the bridge")

    assert result.output == "Hello from ACP"


# --- Additional coverage: AcpHostBridge pagination/session scoping ------------------


async def test_host_bridge_records_since_scopes_by_session_and_supports_snapshots() -> None:
    bridge = AcpHostBridge()
    first_update = AgentMessageChunk(session_update="agent_message_chunk", content=text_block("a"))
    second_update = AgentMessageChunk(session_update="agent_message_chunk", content=text_block("b"))

    await bridge.session_update(session_id="session-1", update=first_update)
    snapshot = bridge.snapshot_index()
    await bridge.session_update(session_id="session-2", update=second_update)

    all_records = bridge.records_since(0)
    assert [record.update for record in all_records] == [first_update, second_update]

    only_new = bridge.records_since(snapshot)
    assert [record.update for record in only_new] == [second_update]

    only_session_1 = bridge.records_since(0, session_id="session-1")
    assert [record.update for record in only_session_1] == [first_update]

    only_session_2_after_snapshot = bridge.records_since(snapshot, session_id="session-2")
    assert [record.update for record in only_session_2_after_snapshot] == [second_update]


async def test_host_bridge_usage_update_since_ignores_real_acp_usage_update_without_usage_field() -> (
    None
):
    # The real ACP `UsageUpdate` schema carries context-window `size`/`used` fields, not a
    # `usage` attribute. `usage_update_since` only reads `getattr(update, "usage", None)`, so
    # recording a genuine `UsageUpdate` must not populate any token counts.
    bridge = AcpHostBridge()
    start_index = bridge.snapshot_index()

    await bridge.session_update(
        session_id="session-1",
        update=UsageUpdate(session_update="usage_update", size=1000, used=10),
    )

    usage = bridge.usage_update_since(start_index, session_id="session-1")

    assert not usage.has_values()


# --- Additional coverage: AcpProvider host_client delegate wiring -------------------


async def test_acp_provider_forwards_host_client_delegate_updates_end_to_end() -> None:
    delegate = HostRecordingClient()
    acp_agent = EchoACPAgent()
    provider = AcpProvider(agent=acp_agent, cwd="/workspace", host_client=delegate)

    assert provider.host.delegate is delegate

    model = AcpModel(model_name="zed-agent", provider=provider)
    result = await Agent(model).run("hello via delegate")

    assert "acp echo: hello via delegate" in result.output
    forwarded_updates = [
        update for _, update in delegate.updates if isinstance(update, AgentMessageChunk)
    ]
    assert len(forwarded_updates) == 1
    assert forwarded_updates[0].content.text == "acp echo: hello via delegate"


# --- Additional coverage: pure helper edge cases -------------------------------------


def test_finish_reason_from_acp_returns_stop_when_stop_reason_is_none() -> None:
    assert client_module._finish_reason_from_acp(None) == "stop"


def test_render_tool_return_falls_back_to_tool_call_id_when_tool_name_is_empty() -> None:
    part = ToolReturnPart(tool_name="", content="ok", tool_call_id="call-42")

    rendered = client_module._render_tool_return(part)

    assert rendered == "Tool result: call-42:\nok"


# --- Additional coverage: root workspace pyproject.toml dependency -----------------


def test_root_pyproject_declares_pydantic_ai_v2_dependency() -> None:
    root_pyproject = Path("pyproject.toml")
    data: dict[str, Any] = tomllib.loads(root_pyproject.read_text())
    dependencies: list[str] = data["project"]["dependencies"]

    assert any(dependency.startswith("pydantic-ai>=2.4.0") for dependency in dependencies)


def test_pydantic_acp_pins_agent_client_protocol_version_used_by_client_module() -> None:
    package_pyproject = Path("packages/adapters/pydantic-acp/pyproject.toml")
    data: dict[str, Any] = tomllib.loads(package_pyproject.read_text())
    dependencies: list[str] = data["project"]["dependencies"]

    assert "agent-client-protocol==0.9.0" in dependencies


# --- Additional coverage: public package exports for the client bridge (__init__.py) -----


def test_acp_client_provider_symbols_are_exported_from_the_top_level_package() -> None:
    for name in (
        "ACP_MODEL_PROFILE",
        "AcpHostBridge",
        "AcpModel",
        "AcpPromptRenderer",
        "AcpProvider",
        "AcpUpdateRecord",
    ):
        assert name in pydantic_acp.__all__
        assert getattr(pydantic_acp, name) is getattr(client_module, name)


def test_acp_model_profile_disables_tool_and_structured_output_support() -> None:
    profile = pydantic_acp.ACP_MODEL_PROFILE

    assert profile["supports_tools"] is False
    assert profile["supports_json_schema_output"] is False
    assert profile["supports_json_object_output"] is False
    assert profile["supports_image_output"] is False
    assert profile["supported_native_tools"] == frozenset()
