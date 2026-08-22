"""Append-only human review queue for non-operative Experience Candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from bauhinia_agent.evolution.events import CandidateReviewRecordedPayload, EvoEvent, EvoReferences, ExperienceCandidateCreatedPayload
from bauhinia_agent.evolution.identifiers import new_evo_id, require_evo_id
from bauhinia_agent.evolution.store import EvoAppendResult, EvoEventStore, EvoStoreError

ReviewDecision = Literal["accept", "reject", "defer", "edit"]
ReviewStatus = Literal["pending", "edited", "deferred"]


class CandidateReviewError(ValueError):
    """Raised when a review cannot be recorded against a Candidate draft."""


class _CandidateReviewStore(Protocol):
    def append(self, event: EvoEvent[CandidateReviewRecordedPayload]) -> EvoAppendResult: ...

    def list_events(self) -> list[EvoEvent]: ...


@dataclass(frozen=True, slots=True)
class CandidateReview:
    decision: ReviewDecision
    reviewer: str
    reason: str
    scope: str | None = None
    ttl_seconds: int | None = None
    sensitivity: str | None = None


@dataclass(frozen=True, slots=True)
class CandidateReviewRecord:
    event_id: str
    candidate_id: str
    run_id: str
    occurred_at: str
    payload: CandidateReviewRecordedPayload


@dataclass(frozen=True, slots=True)
class CandidateQueueItem:
    candidate_id: str
    run_id: str
    candidate: ExperienceCandidateCreatedPayload
    status: ReviewStatus
    latest_review_id: str | None
    effective_scope: str
    effective_ttl_seconds: int | None
    effective_sensitivity: str | None


@dataclass(frozen=True, slots=True)
class CandidateRejectionFeedback:
    candidate_id: str
    review_event_id: str
    reason: str
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CandidateReviewDiagnostic:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class CandidateReviewResult:
    persisted: bool
    review: CandidateReviewRecord | None = None
    diagnostic: CandidateReviewDiagnostic | None = None


class CandidateReviewService:
    """Expose reviewable Candidate drafts without enabling their use in retrieval."""

    def __init__(self, store: EvoEventStore | _CandidateReviewStore) -> None:
        self._store = store

    def review(self, candidate_id: str, review: CandidateReview) -> CandidateReviewResult:
        """Append one human decision or edit without rewriting the Candidate event."""

        require_evo_id(candidate_id, field="candidate_id", kind="candidate")
        _validate_review(review)
        events = self._store.list_events()
        candidates = _candidates(events)
        source = candidates.get(candidate_id)
        if source is None:
            raise CandidateReviewError(f"unknown Candidate: {candidate_id}")
        history = _review_history(events, candidate_id)
        payload = CandidateReviewRecordedPayload(
            candidate_id=candidate_id,
            decision=review.decision,
            reviewer=review.reviewer,
            reason=review.reason,
            scope=review.scope,
            ttl_seconds=review.ttl_seconds,
            sensitivity=review.sensitivity,
        )
        event = EvoEvent(
            event_id=new_evo_id("event"),
            event_type="CandidateReviewRecorded",
            refs=EvoReferences(run_id=source.refs.run_id, candidate_id=candidate_id, parent_event_id=history[-1].event_id if history else source.event_id),
            payload=payload,
        )
        try:
            appended = self._store.append(event)
        except EvoStoreError as error:
            return CandidateReviewResult(False, diagnostic=CandidateReviewDiagnostic("candidate_review_recording_failed", str(error)))
        except Exception as error:  # noqa: BLE001 - review recording cannot change candidate state or source run
            return CandidateReviewResult(
                False,
                diagnostic=CandidateReviewDiagnostic("candidate_review_recording_failed", f"unexpected candidate review failure: {error}"),
            )
        diagnostic = None
        if appended.diagnostic is not None:
            diagnostic = CandidateReviewDiagnostic(appended.diagnostic.code, appended.diagnostic.message)
        return CandidateReviewResult(True, _review_from_event(appended.event), diagnostic)

    def list_queue(self) -> list[CandidateQueueItem]:
        """Return Candidate drafts still awaiting a later human disposition."""

        events = self._store.list_events()
        candidates = _candidates(events)
        return [item for candidate_id, source in sorted(candidates.items()) if (item := _queue_item(source, _review_history(events, candidate_id))) is not None]

    def list_for_retrieval(self) -> list[CandidateQueueItem]:
        """Candidates never enter ordinary retrieval during P6, including accepted ones."""

        return []

    def rejection_feedback(self) -> list[CandidateRejectionFeedback]:
        """Expose explicit rejection reasons for a later compiler-rule improvement loop."""

        events = self._store.list_events()
        candidates = _candidates(events)
        feedback: list[CandidateRejectionFeedback] = []
        for candidate_id, source in candidates.items():
            history = _review_history(events, candidate_id)
            for review in history:
                if review.payload.decision == "reject":
                    feedback.append(
                        CandidateRejectionFeedback(
                            candidate_id=candidate_id,
                            review_event_id=review.event_id,
                            reason=review.payload.reason,
                            evidence_refs=source.payload.evidence_refs,
                        )
                    )
        return feedback


def _validate_review(review: CandidateReview) -> None:
    if review.decision not in {"accept", "reject", "defer", "edit"}:
        raise CandidateReviewError("decision must be accept, reject, defer, or edit")
    for field, value in (("reviewer", review.reviewer), ("reason", review.reason)):
        if not isinstance(value, str) or not value.strip():
            raise CandidateReviewError(f"{field} must be a non-blank string")
    if review.scope is not None and (not isinstance(review.scope, str) or not review.scope.strip()):
        raise CandidateReviewError("scope must be a non-blank string or null")
    if isinstance(review.ttl_seconds, bool) or (review.ttl_seconds is not None and (not isinstance(review.ttl_seconds, int) or review.ttl_seconds < 1)):
        raise CandidateReviewError("ttl_seconds must be a positive integer or null")
    if review.sensitivity is not None and review.sensitivity not in {"public", "internal", "restricted"}:
        raise CandidateReviewError("sensitivity must be public, internal, restricted, or null")
    if review.decision == "edit" and review.scope is None and review.ttl_seconds is None and review.sensitivity is None:
        raise CandidateReviewError("edit requires at least one metadata field")


def _candidates(events: list[EvoEvent]) -> dict[str, EvoEvent[ExperienceCandidateCreatedPayload]]:
    result: dict[str, EvoEvent[ExperienceCandidateCreatedPayload]] = {}
    for event in events:
        if event.event_type != "ExperienceCandidateCreated" or not isinstance(event.payload, ExperienceCandidateCreatedPayload):
            continue
        candidate_id = event.refs.candidate_id
        if candidate_id is None or event.payload.lifecycle_state != "Candidate":
            continue
        if candidate_id in result:
            raise CandidateReviewError(f"duplicate Candidate ID: {candidate_id}")
        result[candidate_id] = event
    return result


def _review_history(events: list[EvoEvent], candidate_id: str) -> list[EvoEvent[CandidateReviewRecordedPayload]]:
    return [event for event in events if event.event_type == "CandidateReviewRecorded" and event.refs.candidate_id == candidate_id and isinstance(event.payload, CandidateReviewRecordedPayload)]


def _queue_item(source: EvoEvent[ExperienceCandidateCreatedPayload], history: list[EvoEvent[CandidateReviewRecordedPayload]]) -> CandidateQueueItem | None:
    latest = history[-1] if history else None
    if latest is not None and latest.payload.decision in {"accept", "reject"}:
        return None
    scope = source.payload.scope
    ttl_seconds = None
    sensitivity = None
    for review in history:
        if review.payload.scope is not None:
            scope = review.payload.scope
        if review.payload.ttl_seconds is not None:
            ttl_seconds = review.payload.ttl_seconds
        if review.payload.sensitivity is not None:
            sensitivity = review.payload.sensitivity
    status: ReviewStatus = "pending" if latest is None else "deferred" if latest.payload.decision == "defer" else "edited"
    candidate_id = source.refs.candidate_id
    if candidate_id is None:
        raise CandidateReviewError("Candidate event requires candidate_id")
    return CandidateQueueItem(
        candidate_id=candidate_id,
        run_id=source.refs.run_id,
        candidate=source.payload,
        status=status,
        latest_review_id=latest.event_id if latest else None,
        effective_scope=scope,
        effective_ttl_seconds=ttl_seconds,
        effective_sensitivity=sensitivity,
    )


def _review_from_event(event: EvoEvent[CandidateReviewRecordedPayload]) -> CandidateReviewRecord:
    candidate_id = event.refs.candidate_id
    if candidate_id is None:
        raise CandidateReviewError("CandidateReviewRecorded event requires candidate_id")
    return CandidateReviewRecord(event.event_id, candidate_id, event.refs.run_id, event.occurred_at, event.payload)
