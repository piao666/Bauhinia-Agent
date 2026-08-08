from __future__ import annotations

from bauhinia_agent.evolution.candidate_analysis import CandidateAnalysisService
from bauhinia_agent.evolution.events import EvoEvent, EvoReferences, ExperienceCandidateCreatedPayload
from bauhinia_agent.evolution.identifiers import new_evo_id
from bauhinia_agent.evolution.store import EvoEventStore, EvoStoreError


def test_duplicate_candidates_form_one_traceable_merge_proposal_without_mutation(tmp_path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    first = _candidate(store, summary="Run focused tests after editing the provider adapter.")
    second = _candidate(store, summary="Run focused tests after editing the provider adapter.")

    result = CandidateAnalysisService(store).analyze()

    assert result.persisted is True
    assert len(result.merge_proposals) == 1
    proposal = result.merge_proposals[0]
    assert proposal.payload.candidate_ids == tuple(sorted((first.refs.candidate_id, second.refs.candidate_id)))
    assert proposal.payload.scope == "project"
    assert proposal.payload.kind == "plan_template"
    assert proposal.payload.similarity == 1.0
    assert proposal.payload.evidence_refs == ("evidence_a", "evidence_b")
    assert result.conflict_groups == ()
    assert [event.event_type for event in store.list_events()] == [
        "ExperienceCandidateCreated",
        "ExperienceCandidateCreated",
        "CandidateMergeProposed",
    ]


def test_analysis_is_idempotent_and_does_not_linearly_create_near_duplicate_proposals(tmp_path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    _candidate(store, summary="Run focused tests after editing the provider adapter.")
    _candidate(store, summary="Run focused tests after editing the provider adapter.")
    service = CandidateAnalysisService(store)

    first = service.analyze()
    second = service.analyze()

    assert len(first.merge_proposals) == 1
    assert second.merge_proposals == ()
    assert [event.event_type for event in store.list_events()].count("CandidateMergeProposed") == 1


def test_opposite_conclusions_form_an_explicit_conflict_group(tmp_path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    first = _candidate(
        store,
        kind="plan_template",
        summary="Use the provider adapter fallback after a timeout.",
        applicability="Provider adapter timeout recovery.",
    )
    second = _candidate(
        store,
        kind="anti_pattern",
        summary="Do not use the provider adapter fallback after a timeout.",
        applicability="Provider adapter timeout recovery.",
    )

    result = CandidateAnalysisService(store).analyze()

    assert result.merge_proposals == ()
    assert len(result.conflict_groups) == 1
    conflict = result.conflict_groups[0]
    assert conflict.payload.candidate_ids == tuple(sorted((first.refs.candidate_id, second.refs.candidate_id)))
    assert conflict.payload.conclusions == ("caution", "support")
    assert conflict.payload.evidence_refs == ("evidence_a", "evidence_b")
    assert [event.event_type for event in store.list_events()][-1] == "CandidateConflictDetected"


def test_scope_boundary_prevents_cross_scope_merge_or_conflict(tmp_path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    _candidate(store, scope="project", summary="Run focused tests after editing the provider adapter.")
    _candidate(store, scope="session", summary="Run focused tests after editing the provider adapter.")

    result = CandidateAnalysisService(store).analyze()

    assert result.merge_proposals == ()
    assert result.conflict_groups == ()
    assert len(store.list_events()) == 2


def test_analysis_recorder_failure_returns_diagnostic_without_mutating_candidates(tmp_path) -> None:
    source = EvoEventStore(tmp_path / ".bauhinia-agent")
    _candidate(source, summary="Run focused tests after editing the provider adapter.")
    _candidate(source, summary="Run focused tests after editing the provider adapter.")
    source_events = source.list_events()

    result = CandidateAnalysisService(_FailingStore(source_events)).analyze()

    assert result.persisted is False
    assert result.merge_proposals == ()
    assert result.conflict_groups == ()
    assert result.diagnostic is not None
    assert result.diagnostic.code == "candidate_analysis_recording_failed"
    assert len(source_events) == 2


def _candidate(
    store: EvoEventStore,
    *,
    kind: str = "plan_template",
    scope: str = "project",
    summary: str,
    applicability: str = "Provider adapter verification.",
) -> EvoEvent:
    candidate_id = new_evo_id("candidate")
    run_id = new_evo_id("run")
    suffix = "a" if len(store.list_events()) == 0 else "b"
    return store.append(
        EvoEvent(
            event_id=new_evo_id("event"),
            event_type="ExperienceCandidateCreated",
            refs=EvoReferences(run_id=run_id, candidate_id=candidate_id),
            payload=ExperienceCandidateCreatedPayload(
                kind=kind,
                summary=summary,
                scope=scope,
                applicability=applicability,
                confidence=0.4,
                source_event_ids=(new_evo_id("event"),),
                evidence_refs=(f"evidence_{suffix}",),
                source_run_ids=(run_id,),
            ),
        )
    ).event


class _FailingStore:
    def __init__(self, events: list[object]) -> None:
        self._events = events

    def list_events(self) -> list[object]:
        return self._events

    def append(self, event: object) -> object:
        raise EvoStoreError("store offline")
