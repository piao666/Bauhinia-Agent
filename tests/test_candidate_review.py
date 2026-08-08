from __future__ import annotations

import pytest

from bauhinia_agent.evolution.candidate_review import CandidateReview, CandidateReviewError, CandidateReviewService
from bauhinia_agent.evolution.events import EvoEvent, EvoReferences, ExperienceCandidateCreatedPayload
from bauhinia_agent.evolution.identifiers import new_evo_id
from bauhinia_agent.evolution.store import EvoEventStore, EvoStoreError


def test_review_queue_records_edit_without_mutating_source_candidate(tmp_path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    source = _candidate(store)
    candidate_id = source.refs.candidate_id
    assert candidate_id is not None
    service = CandidateReviewService(store)

    result = service.review(
        candidate_id,
        CandidateReview(
            decision="edit",
            reviewer="curator_1",
            reason="Narrow the candidate before later evaluation.",
            scope="branch",
            ttl_seconds=3_600,
            sensitivity="internal",
        ),
    )

    assert result.persisted is True
    assert result.review is not None
    assert result.review.payload.decision == "edit"
    assert result.review.payload.scope == "branch"
    assert result.review.payload.ttl_seconds == 3_600
    assert result.review.payload.sensitivity == "internal"
    assert store.list_events()[0].payload.scope == "project"
    queue = service.list_queue()
    assert len(queue) == 1
    assert queue[0].candidate_id == candidate_id
    assert queue[0].status == "edited"
    assert queue[0].latest_review_id == result.review.event_id
    assert queue[0].effective_scope == "branch"
    assert queue[0].effective_ttl_seconds == 3_600
    assert queue[0].effective_sensitivity == "internal"


def test_accept_and_reject_are_auditable_but_never_enable_retrieval(tmp_path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    accepted = _candidate(store)
    rejected = _candidate(store)
    accepted_id = accepted.refs.candidate_id
    rejected_id = rejected.refs.candidate_id
    assert accepted_id is not None
    assert rejected_id is not None
    service = CandidateReviewService(store)

    accepted_result = service.review(accepted_id, CandidateReview(decision="accept", reviewer="curator_1", reason="Source evidence is complete."))
    rejected_result = service.review(rejected_id, CandidateReview(decision="reject", reviewer="curator_1", reason="Contradicts current architecture."))

    assert accepted_result.persisted is True
    assert rejected_result.persisted is True
    assert service.list_queue() == []
    assert service.list_for_retrieval() == []
    feedback = service.rejection_feedback()
    assert len(feedback) == 1
    assert feedback[0].candidate_id == rejected_id
    assert feedback[0].reason == "Contradicts current architecture."
    assert all(event.event_type not in {"MemoryCreated", "MemoryUsed", "PromotionChanged"} for event in store.list_events())


def test_deferred_candidate_remains_in_queue_and_review_replaces_derived_status(tmp_path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    source = _candidate(store)
    candidate_id = source.refs.candidate_id
    assert candidate_id is not None
    service = CandidateReviewService(store)

    deferred = service.review(candidate_id, CandidateReview(decision="defer", reviewer="curator_1", reason="Need more independent Runs."))
    edited = service.review(candidate_id, CandidateReview(decision="edit", reviewer="curator_1", reason="Document a short TTL.", ttl_seconds=600))

    assert deferred.persisted is True
    assert edited.persisted is True
    queue = service.list_queue()
    assert len(queue) == 1
    assert queue[0].status == "edited"
    assert queue[0].effective_ttl_seconds == 600


def test_review_rejects_missing_rejection_reason_and_empty_edit(tmp_path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    source = _candidate(store)
    candidate_id = source.refs.candidate_id
    assert candidate_id is not None
    service = CandidateReviewService(store)

    with pytest.raises(CandidateReviewError, match="reason"):
        service.review(candidate_id, CandidateReview(decision="reject", reviewer="curator_1", reason=""))
    with pytest.raises(CandidateReviewError, match="at least one"):
        service.review(candidate_id, CandidateReview(decision="edit", reviewer="curator_1", reason="No fields changed."))


def test_review_recorder_failure_returns_diagnostic_without_changing_candidate(tmp_path) -> None:
    source_store = EvoEventStore(tmp_path / ".bauhinia-agent")
    source = _candidate(source_store)
    candidate_id = source.refs.candidate_id
    assert candidate_id is not None
    source_events = source_store.list_events()

    result = CandidateReviewService(_FailingStore(source_events)).review(
        candidate_id,
        CandidateReview(decision="accept", reviewer="curator_1", reason="Source evidence is complete."),
    )

    assert result.persisted is False
    assert result.review is None
    assert result.diagnostic is not None
    assert result.diagnostic.code == "candidate_review_recording_failed"
    assert len(source_events) == 1


def _candidate(store: EvoEventStore) -> EvoEvent:
    candidate_id = new_evo_id("candidate")
    run_id = new_evo_id("run")
    return store.append(
        EvoEvent(
            event_id=new_evo_id("event"),
            event_type="ExperienceCandidateCreated",
            refs=EvoReferences(run_id=run_id, candidate_id=candidate_id),
            payload=ExperienceCandidateCreatedPayload(
                kind="plan_template",
                summary="Run focused tests after editing the provider adapter.",
                scope="project",
                applicability="Provider adapter verification.",
                confidence=0.4,
                source_event_ids=(new_evo_id("event"),),
                evidence_refs=(new_evo_id("evidence"),),
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
