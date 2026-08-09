from __future__ import annotations

from bauhinia_agent.evolution.compiler import ExperienceCompiler
from bauhinia_agent.evolution.evidence import EvidenceAdapter, EvidenceInput
from bauhinia_agent.evolution.events import DecisionRecordedPayload, EvoEvent, EvoReferences, PlanCreatedPayload
from bauhinia_agent.evolution.identifiers import new_evo_id
from bauhinia_agent.evolution.outcomes import OutcomeClassifier
from bauhinia_agent.evolution.store import EvoEventStore, EvoStoreError


def test_compiler_creates_traceable_plan_template_from_verified_success(tmp_path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    run_id = new_evo_id("run")
    plan_event, decision_event = _record_plan_and_decision(store, run_id)
    evidence = EvidenceAdapter(store).record(
        EvidenceInput(
            run_id=run_id,
            evidence_type="test",
            source="pytest",
            summary="12 passed",
            exit_code=0,
            verified=True,
        )
    )
    outcome = OutcomeClassifier(store).classify(run_id)

    result = ExperienceCompiler(store).compile(run_id, environment_summary="Windows; TOKEN=private; Python 3.12")

    assert result.persisted is True
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.payload.kind == "plan_template"
    assert candidate.payload.lifecycle_state == "Candidate"
    assert candidate.payload.confidence <= 0.4
    assert candidate.payload.novelty is None
    assert candidate.payload.novelty_status == "unassessed"
    assert candidate.payload.environment_summary == "Windows; TOKEN=[REDACTED]; Python 3.12"
    assert evidence.evidence is not None
    assert outcome.outcome is not None
    assert candidate.payload.evidence_refs == (evidence.evidence.evidence_id,)
    assert candidate.payload.source_event_ids == (
        plan_event.event_id,
        decision_event.event_id,
        evidence.evidence.event_id,
        outcome.outcome.event_id,
    )
    assert candidate.payload.source_run_ids == (run_id,)
    assert candidate.payload.extensions["task_signature"] != "unknown"
    assert candidate.payload.extensions["pattern_key"]
    assert candidate.candidate_id.startswith("candidate_")
    assert ExperienceCompiler(store).list_for_run(run_id) == [candidate]
    assert all(event.event_type not in {"MemoryCreated", "PromotionChanged"} for event in store.list_events())
    persisted = (tmp_path / ".bauhinia-agent" / "evo" / "events.jsonl").read_text(encoding="utf-8")
    assert "private" not in persisted


def test_compiler_creates_evidence_linked_debug_hint_for_verification_failure(tmp_path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    run_id = new_evo_id("run")
    evidence = EvidenceAdapter(store).record(
        EvidenceInput(
            run_id=run_id,
            evidence_type="test",
            source="pytest",
            summary="test_login failed",
            exit_code=1,
            verified=True,
        )
    )
    OutcomeClassifier(store).classify(run_id)

    result = ExperienceCompiler(store).compile(run_id, environment_summary="Windows")

    assert result.persisted is True
    assert result.candidates[0].payload.kind == "debug_hint"
    assert "rerun" in result.candidates[0].payload.summary.lower()
    assert "hypothesis" in result.candidates[0].payload.summary.lower()
    assert evidence.evidence is not None
    assert result.candidates[0].payload.evidence_refs == (evidence.evidence.evidence_id,)


def test_compiler_refuses_to_create_candidate_without_evidence_or_outcome(tmp_path) -> None:
    compiler = ExperienceCompiler(EvoEventStore(tmp_path / ".bauhinia-agent"))

    result = compiler.compile(new_evo_id("run"), environment_summary="Windows")

    assert result.persisted is False
    assert result.candidates == ()
    assert result.diagnostic is not None
    assert result.diagnostic.code == "insufficient_evidence"


def test_compiler_reports_recorder_failure_without_changing_source_run(tmp_path) -> None:
    source_store = EvoEventStore(tmp_path / ".bauhinia-agent")
    run_id = new_evo_id("run")
    EvidenceAdapter(source_store).record(EvidenceInput(run_id=run_id, evidence_type="test", source="pytest", summary="1 passed", exit_code=0, verified=True))
    OutcomeClassifier(source_store).classify(run_id)
    result = ExperienceCompiler(_FailingStore(source_store.list_events())).compile(run_id, environment_summary="Windows")

    assert result.persisted is False
    assert result.candidates == ()
    assert result.diagnostic is not None
    assert result.diagnostic.code == "candidate_recording_failed"


def _record_plan_and_decision(store: EvoEventStore, run_id: str) -> tuple[EvoEvent, EvoEvent]:
    plan_id = new_evo_id("plan")
    plan_event = store.append(
        EvoEvent(
            event_id=new_evo_id("event"),
            event_type="PlanCreated",
            refs=EvoReferences(run_id=run_id, plan_id=plan_id),
            payload=PlanCreatedPayload(goal="implement a verified change"),
        )
    ).event
    decision_event = store.append(
        EvoEvent(
            event_id=new_evo_id("event"),
            event_type="DecisionRecorded",
            refs=EvoReferences(run_id=run_id, plan_id=plan_id),
            payload=DecisionRecordedPayload(
                subgoal="verify the change",
                evidence_refs=(),
                assumptions=(),
                options_considered=("run focused tests",),
                selected_action="run focused tests after the implementation",
                rationale_summary="deterministic verification is required",
                confidence=0.8,
                expected_observation="tests pass",
                verification_method="pytest",
            ),
        )
    ).event
    return plan_event, decision_event


class _FailingStore:
    def __init__(self, events: list[object]) -> None:
        self._events = events

    def append(self, event: object) -> object:
        raise EvoStoreError("store offline")

    def list_events(self) -> list[object]:
        return self._events
