from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from bauhinia_agent.agent.subagent import SubagentRequest, SubagentRunner
from bauhinia_agent.context.store import JsonlSessionStore
from bauhinia_agent.providers.base import ChatProvider
from bauhinia_agent.providers.types import (
    ChatRequest,
    ChatResponse,
    ProviderCapabilities,
    ToolCall,
    ToolDefinition,
)
from bauhinia_agent.runtime.cancellation import CancellationToken
from bauhinia_agent.tools.types import Tool, make_text_result


@dataclass
class _Provider(ChatProvider):
    responses: list[ChatResponse]
    capabilities: ProviderCapabilities = field(default_factory=ProviderCapabilities)
    requests: list[ChatRequest] = field(default_factory=list)

    @property
    def name(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return "fake-model"

    def complete(self, request: ChatRequest) -> ChatResponse:
        if request.tools == [] and request.tool_choice == "none" and request.max_tokens == 512:
            return ChatResponse(
                provider=self.name,
                model=self.model,
                content='{"decision":"uncertain","basis_message_id":"msg"}',
            )
        self.requests.append(request)
        return self.responses.pop(0)


def _tool(name: str) -> Tool:
    return Tool(
        definition=ToolDefinition(name=name, description=name),
        executor=lambda: make_text_result(name, name),
    )


def test_contract_intersects_profile_capabilities_and_effects(tmp_path) -> None:
    runner = SubagentRunner(
        store=JsonlSessionStore(tmp_path),
        provider=_Provider([]),
        tools=[
            _tool("view"),
            _tool("write"),
            _tool("shell"),
            _tool("delegate"),
        ],
    )

    read_request = SubagentRequest(
        role="coder",
        task="inspect only",
        parent_session_id="session_parent",
        allowed_tool_names=("view",),
        allowed_effects=("read",),
    )
    assert [tool.name for tool in runner.tools_for_request(read_request)] == ["view"]

    mismatched = SubagentRequest(
        role="coder",
        task="mismatched fields",
        parent_session_id="session_parent",
        allowed_tool_names=("write",),
        allowed_effects=("read",),
    )
    with pytest.raises(ValueError, match="requires Effects .*write"):
        runner.tools_for_request(mismatched)

    no_capabilities = SubagentRequest(
        role="coder",
        task="no tools",
        parent_session_id="session_parent",
        allowed_tool_names=(),
        allowed_effects=("read", "write", "execute"),
    )
    assert runner.tools_for_request(no_capabilities) == []

    legacy = SubagentRequest(
        role="coder",
        task="legacy defaults",
        parent_session_id="session_parent",
    )
    assert {tool.name for tool in runner.tools_for_request(legacy)} == {
        "view",
        "write",
        "shell",
    }


def test_actual_child_registry_exposes_only_contract_tools(tmp_path) -> None:
    provider = _Provider([ChatResponse(provider="fake", model="fake-model", content="scoped")])
    runner = SubagentRunner(
        store=JsonlSessionStore(tmp_path),
        provider=provider,
        tools=[_tool("view"), _tool("grep"), _tool("write")],
    )

    result = runner.run(
        SubagentRequest(
            role="reviewer",
            task="use grep only",
            parent_session_id="session_parent",
            allowed_tool_names=("grep",),
            allowed_effects=("read",),
        )
    )

    assert result.ok is True
    # This is the provider-facing registry after session helper injection.
    assert [definition.name for definition in provider.requests[0].tools] == ["grep"]


def test_unknown_effect_fails_closed_before_child_creation(tmp_path) -> None:
    store = JsonlSessionStore(tmp_path)
    provider = _Provider([])
    runner = SubagentRunner(
        store=store,
        provider=provider,
        tools=[_tool("view")],
    )

    result = runner.run(
        SubagentRequest(
            role="researcher",
            task="unsafe contract",
            parent_session_id="session_parent",
            allowed_tool_names=("view",),
            allowed_effects=("teleport",),
        )
    )

    assert result.ok is False
    assert result.error == "capability_scope_invalid"
    assert result.child_session_id == ""
    assert provider.requests == []
    assert list((tmp_path / "sessions").glob("*.jsonl")) == []


def test_pre_cancel_returns_fixed_result_without_starting_child(tmp_path) -> None:
    token = CancellationToken()
    token.cancel()
    provider = _Provider([])
    runner = SubagentRunner(
        store=JsonlSessionStore(tmp_path),
        provider=provider,
        tools=[_tool("view")],
    )

    result = runner.run(
        SubagentRequest(
            role="researcher",
            task="do not start",
            parent_session_id="session_parent",
        ),
        cancellation_token=token,
    )

    assert result.ok is False
    assert result.error == "cancelled"
    assert result.summary == "Subagent task was cancelled."
    assert result.confidence == 0.0
    assert result.confidence_source == "unknown"
    assert result.confidence_source_event_id is None
    assert provider.requests == []


def test_cancellation_token_reaches_child_loop(tmp_path) -> None:
    token = CancellationToken()

    class _CancellingProvider(_Provider):
        def complete(self, request: ChatRequest) -> ChatResponse:
            if request.tools == [] and request.tool_choice == "none" and request.max_tokens == 512:
                return super().complete(request)
            self.requests.append(request)
            token.cancel()
            return ChatResponse(
                provider=self.name,
                model=self.model,
                content="late response",
            )

    provider = _CancellingProvider([])
    runner = SubagentRunner(
        store=JsonlSessionStore(tmp_path),
        provider=provider,
        tools=[_tool("view")],
    )

    result = runner.run(
        SubagentRequest(
            role="researcher",
            task="cancel during provider call",
            parent_session_id="session_parent",
        ),
        cancellation_token=token,
    )

    assert result.ok is False
    assert result.error == "cancelled"
    assert result.summary == "Subagent task was cancelled."
    assert result.confidence == 0.0


def test_unclassified_tool_evidence_is_not_confidence(tmp_path) -> None:
    provider = _Provider(
        [
            ChatResponse(
                provider="fake",
                model="fake-model",
                content="done without deterministic verification",
            )
        ]
    )
    runner = SubagentRunner(
        store=JsonlSessionStore(tmp_path),
        provider=provider,
        tools=[_tool("view")],
    )

    result = runner.run(
        SubagentRequest(
            role="researcher",
            task="summarize",
            parent_session_id="session_parent",
            child_run_id="run_child_unknown",
            allowed_tool_names=("view",),
            allowed_effects=("read",),
        )
    )

    assert result.ok is True
    assert result.evidence == []
    assert result.confidence == 0.0
    assert result.confidence_source == "unknown"
    assert result.confidence_source_event_id is None


def test_negative_verified_outcome_is_not_reported_as_success(tmp_path) -> None:
    from bauhinia_agent.tools.shell import create_shell_tool

    tests_root = tmp_path / "tests"
    tests_root.mkdir()
    (tests_root / "__init__.py").write_text("", encoding="utf-8")
    (tests_root / "test_failure.py").write_text(
        "import unittest\n\nclass FailureTest(unittest.TestCase):\n" "    def test_failure(self):\n        self.fail('expected failure')\n",
        encoding="utf-8",
    )
    shell = create_shell_tool(tmp_path)
    provider = _Provider(
        [
            ChatResponse(
                provider="fake",
                model="fake-model",
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_failed_test",
                        name="shell",
                        arguments={"command": "python -m unittest -q tests.test_failure"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            ChatResponse(
                provider="fake",
                model="fake-model",
                content="verification failed",
            ),
        ]
    )
    runner = SubagentRunner(
        store=JsonlSessionStore(tmp_path),
        provider=provider,
        tools=[shell],
    )
    result = runner.run(
        SubagentRequest(
            role="tester",
            task="verify",
            parent_session_id="session_parent",
            child_run_id="run_child_failed_verification",
            allowed_tool_names=("shell",),
            allowed_effects=("execute", "write", "network"),
        )
    )

    assert result.ok is False
    assert result.error == "verification_failure"
    assert len(result.evidence) == 2
    assert result.confidence == 0.95
    assert result.confidence_source == "outcome_classified"
    assert result.confidence_source_event_id is not None
