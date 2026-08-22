from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from bauhinia_agent.agent.evo_observer import AgentEvoObserver, _verification_kind
from bauhinia_agent.agent.loop import AgentLoop
from bauhinia_agent.agent.session import AgentSession
from bauhinia_agent.agent.tool_execution import ToolExecutionEvent
from bauhinia_agent.app.runtime import AgentChatRunner, CurrentSessionState
from bauhinia_agent.context.store import JsonlSessionStore
from bauhinia_agent.evolution import CandidateReview, CandidateReviewService, EvoEventStore, ExperienceCandidateCreatedPayload, OutcomeClassifiedPayload
from bauhinia_agent.providers.base import ChatProvider
from bauhinia_agent.providers.types import ChatRequest, ChatResponse, ToolCall, ToolDefinition
from bauhinia_agent.tools.types import Tool, ToolResult

_VERIFICATION_COMMAND = "python -m unittest -q tests.test_example"


@dataclass
class _FakeProvider(ChatProvider):
    responses: list[ChatResponse]
    requests: list[ChatRequest] = field(default_factory=list)

    @property
    def name(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return "fake-model"

    def complete(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        return self.responses.pop(0)


def test_real_agent_run_records_verified_evidence_outcome_candidate_and_review(tmp_path) -> None:
    data_root = tmp_path / ".bauhinia-agent"
    session_store = JsonlSessionStore(data_root)
    tool = _verification_tool(tmp_path, ok=True)
    session = AgentSession.create(store=session_store, session_id="session_evo_gate", agents_md="", tools=[tool])
    provider = _FakeProvider(
        [
            ChatResponse(
                provider="fake",
                model="fake-model",
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_verify",
                        name="shell",
                        arguments={"command": _VERIFICATION_COMMAND},
                    )
                ],
                finish_reason="tool_calls",
            ),
            ChatResponse(provider="fake", model="fake-model", content="Verification finished."),
        ]
    )
    runner = AgentChatRunner(
        current_session=CurrentSessionState(session),
        provider=provider,
        tools=[tool],
        evolution_enabled=True,
    )

    response = runner.run_user_turn("Run the focused verification.")

    assert response.content == "Verification finished."
    events = EvoEventStore(data_root).list_events()
    evidence = [event for event in events if event.event_type == "EvidenceRecorded"]
    outcome = next(event for event in events if isinstance(event.payload, OutcomeClassifiedPayload))
    candidate = next(event for event in events if isinstance(event.payload, ExperienceCandidateCreatedPayload))
    assert len(evidence) == 2
    assert outcome.payload.outcome == "success"
    assert candidate.refs.run_id == outcome.refs.run_id
    assert set(candidate.payload.evidence_refs) == {event.refs.evidence_id for event in evidence}
    assert candidate.payload.lifecycle_state == "Candidate"
    observed = runner.loops[-1].evolution_observer
    assert observed is not None
    assert observed.last_result is not None
    assert set(observed.last_result.evidence_ids) == {event.refs.evidence_id for event in evidence}
    assert observed.last_result.outcome == "success"
    assert observed.last_result.outcome_category == "task_success"
    assert observed.last_result.outcome_confidence == 0.95

    review = CandidateReviewService(EvoEventStore(data_root)).review(
        candidate.refs.candidate_id or "",
        CandidateReview(decision="accept", reviewer="gate_tester", reason="Trace is complete."),
    )

    assert review.persisted is True
    assert CandidateReviewService(EvoEventStore(data_root)).list_for_retrieval() == []
    assert [message.role for message in session.rebuild_view().messages] == ["user", "assistant", "tool", "assistant"]


def test_real_agent_run_records_failure_candidate_without_changing_loop_result(tmp_path) -> None:
    data_root = tmp_path / ".bauhinia-agent"
    session_store = JsonlSessionStore(data_root)
    tool = _verification_tool(tmp_path, ok=False)
    session = AgentSession.create(store=session_store, session_id="session_evo_failure", agents_md="", tools=[tool])
    provider = _FakeProvider(
        [
            ChatResponse(
                provider="fake",
                model="fake-model",
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_verify",
                        name="shell",
                        arguments={"command": _VERIFICATION_COMMAND},
                    )
                ],
                finish_reason="tool_calls",
            ),
            ChatResponse(provider="fake", model="fake-model", content="Verification failed; investigate it."),
        ]
    )
    runner = AgentChatRunner(
        current_session=CurrentSessionState(session),
        provider=provider,
        tools=[tool],
        evolution_enabled=True,
    )

    response = runner.run_user_turn("Run the focused verification.")

    assert response.content == "Verification failed; investigate it."
    events = EvoEventStore(data_root).list_events()
    outcome = next(event for event in events if isinstance(event.payload, OutcomeClassifiedPayload))
    candidate = next(event for event in events if isinstance(event.payload, ExperienceCandidateCreatedPayload))
    assert outcome.payload.category == "verification_failure"
    assert candidate.payload.kind == "debug_hint"
    assert candidate.refs.run_id == outcome.refs.run_id


def test_fixed_child_run_observer_returns_evidence_outcome_without_compiling_candidate(
    tmp_path,
) -> None:
    data_root = tmp_path / ".bauhinia-agent"
    session_store = JsonlSessionStore(data_root)
    tool = _verification_tool(tmp_path, ok=True)
    session = AgentSession.create(
        store=session_store,
        session_id="session_evo_child",
        agents_md="",
        tools=[tool],
    )
    provider = _FakeProvider(
        [
            ChatResponse(
                provider="fake",
                model="fake-model",
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_child_verify",
                        name="shell",
                        arguments={"command": _VERIFICATION_COMMAND},
                    )
                ],
                finish_reason="tool_calls",
            ),
            ChatResponse(
                provider="fake",
                model="fake-model",
                content="Child verification finished.",
            ),
        ]
    )
    observer = AgentEvoObserver(
        session=session,
        provider=provider,
        run_id="run_child_fixed",
        compile_candidates=False,
    )
    loop = AgentLoop(
        session=session,
        provider=provider,
        tools=[tool],
        evolution_observer=observer,
        enable_delegate_tool=False,
    )

    response = loop.run_user_turn("Verify the child result.")

    assert response.content == "Child verification finished."
    assert observer.last_result is not None
    assert observer.last_result.run_id == "run_child_fixed"
    assert observer.last_result.evidence_count == 2
    assert len(observer.last_result.evidence_ids) == 2
    assert observer.last_result.outcome_category == "task_success"
    assert observer.last_result.outcome_confidence == 0.95
    assert observer.last_result.candidate_ids == ()
    events = EvoEventStore(data_root).list_events()
    assert not any(isinstance(event.payload, ExperienceCandidateCreatedPayload) for event in events)
    with pytest.raises(RuntimeError, match="only observe one turn"):
        observer.begin_turn()


@pytest.mark.parametrize(
    "command",
    [
        "pytest -q",
        "echo pytest -q",
        "pytest -q || exit 0",
        "pytest -q; true",
        "pytest -q | tee test.log",
    ],
)
def test_observer_does_not_treat_mentions_or_masked_composites_as_verified_tests(
    tmp_path,
    command: str,
) -> None:
    data_root = tmp_path / ".bauhinia-agent"
    session = AgentSession.create(
        store=JsonlSessionStore(data_root),
        session_id="session_evo_untrusted_command",
        agents_md="",
        tools=[_command_tool(command)],
    )
    provider = _FakeProvider(
        [
            ChatResponse(
                provider="fake",
                model="fake-model",
                content="",
                tool_calls=[ToolCall(id="call_untrusted", name="shell", arguments={})],
                finish_reason="tool_calls",
            ),
            ChatResponse(provider="fake", model="fake-model", content="Finished."),
        ]
    )
    loop = AgentLoop(
        session=session,
        provider=provider,
        tools=[_command_tool(command)],
        evolution_observer=AgentEvoObserver(
            session=session,
            provider=provider,
            compile_candidates=False,
        ),
        enable_delegate_tool=False,
    )

    loop.run_user_turn("Run the command.")

    evidence = [event.payload for event in EvoEventStore(data_root).list_events() if event.event_type == "EvidenceRecorded"]
    assert len(evidence) == 1
    assert evidence[0].evidence_type == "tool"
    assert evidence[0].verified is False


@pytest.mark.parametrize(
    ("command", "expected_type"),
    [
        (r".\.venv\Scripts\python.exe -m pytest -q", "test"),
        ("python -m unittest discover", "test"),
        ("ruff check .", "lint"),
        ("python -m mypy bauhinia_agent", "type_check"),
        ("npm run build", "build"),
    ],
)
def test_observer_recognizes_direct_verification_entrypoints(
    tmp_path,
    command: str,
    expected_type: str,
) -> None:
    del tmp_path
    event = ToolExecutionEvent(
        kind="finished",
        tool_call=ToolCall(
            id="call_direct",
            name="shell",
            arguments={"command": command},
        ),
        result=ToolResult(
            name="shell",
            ok=True,
            content="command completed",
            data={"command": command, "exit_code": 0, "cwd": "."},
        ),
    )
    assert _verification_kind(event) == expected_type


def _verification_tool(root, *, ok: bool) -> Tool:
    from bauhinia_agent.tools.shell import create_shell_tool

    tests_root = root / "tests"
    tests_root.mkdir(exist_ok=True)
    (tests_root / "__init__.py").write_text("", encoding="utf-8")
    assertion = "self.assertTrue(True)" if ok else "self.fail('expected failure')"
    (tests_root / "test_example.py").write_text(
        "import unittest\n\nclass ExampleTest(unittest.TestCase):\n" f"    def test_example(self):\n        {assertion}\n",
        encoding="utf-8",
    )
    return create_shell_tool(root)


def _command_tool(command: str) -> Tool:
    def execute() -> ToolResult:
        return ToolResult(
            name="shell",
            ok=True,
            content="command completed",
            data={"command": command, "exit_code": 0, "cwd": "."},
        )

    return Tool(
        definition=ToolDefinition(
            name="shell",
            description="command verifier",
            parameters={"type": "object", "properties": {}},
        ),
        executor=execute,
    )
