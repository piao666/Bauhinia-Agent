from __future__ import annotations

from dataclasses import dataclass, field

from bauhinia_agent.agent.session import AgentSession
from bauhinia_agent.app.runtime import AgentChatRunner, CurrentSessionState
from bauhinia_agent.context.store import JsonlSessionStore
from bauhinia_agent.evolution import CandidateReview, CandidateReviewService, EvoEventStore, ExperienceCandidateCreatedPayload, OutcomeClassifiedPayload
from bauhinia_agent.providers.base import ChatProvider
from bauhinia_agent.providers.types import ChatRequest, ChatResponse, ToolCall, ToolDefinition
from bauhinia_agent.tools.types import Tool, ToolResult


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
    tool = _verification_tool(ok=True)
    session = AgentSession.create(store=session_store, session_id="session_evo_gate", agents_md="", tools=[tool])
    provider = _FakeProvider(
        [
            ChatResponse(
                provider="fake",
                model="fake-model",
                content="",
                tool_calls=[ToolCall(id="call_verify", name="shell", arguments={})],
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
    tool = _verification_tool(ok=False)
    session = AgentSession.create(store=session_store, session_id="session_evo_failure", agents_md="", tools=[tool])
    provider = _FakeProvider(
        [
            ChatResponse(
                provider="fake",
                model="fake-model",
                content="",
                tool_calls=[ToolCall(id="call_verify", name="shell", arguments={})],
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


def _verification_tool(*, ok: bool) -> Tool:
    def execute() -> ToolResult:
        return ToolResult(
            name="shell",
            ok=ok,
            content="3 passed" if ok else "1 failed",
            data={"command": "pytest -q tests/test_example.py", "exit_code": 0 if ok else 1, "cwd": "."},
            error=None if ok else "command exited with 1",
        )

    return Tool(
        definition=ToolDefinition(name="shell", description="test verifier", parameters={"type": "object", "properties": {}}),
        executor=execute,
    )
