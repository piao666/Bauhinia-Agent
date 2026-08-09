"""Append-only Candidate Artifact lifecycle and human Promotion Gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from bauhinia_agent.evaluation.comparison import (
    EvaluationComparisonRecord,
    EvaluationComparisonService,
    EvaluationComparisonSpec,
)
from bauhinia_agent.evolution.candidate_artifacts import CandidateArtifactLifecycle, CandidateArtifactRecord
from bauhinia_agent.evolution.evidence import redact_text
from bauhinia_agent.evolution.events import (
    CandidateArtifactCreatedPayload,
    CandidateShadowTrialRecordedPayload,
    EvaluationComparisonCompletedPayload,
    EvoEvent,
    EvoReferences,
    PromotionChangedPayload,
)
from bauhinia_agent.evolution.identifiers import new_evo_id, require_evo_id
from bauhinia_agent.evolution.store import EvoAppendResult, EvoEventStore, EvoStoreError

PROMOTION_APPROVER_ROLES = frozenset({"maintainer", "owner"})


class PromotionError(ValueError):
    """Raised when a lifecycle transition violates the Promotion Gate."""


class _PromotionStore(Protocol):
    def append(self, event: EvoEvent) -> EvoAppendResult: ...

    def list_events(self) -> list[EvoEvent]: ...


@dataclass(frozen=True, slots=True)
class PromotionRecord:
    event_id: str
    promotion_id: str
    artifact_id: str
    run_id: str
    occurred_at: str
    payload: PromotionChangedPayload


@dataclass(frozen=True, slots=True)
class PromotionDiagnostic:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class PromotionResult:
    persisted: bool
    promotion: PromotionRecord | None = None
    diagnostic: PromotionDiagnostic | None = None


@dataclass(frozen=True, slots=True)
class PromotionGateResult:
    validated: bool
    report: EvaluationComparisonRecord | None = None
    transition: PromotionRecord | None = None
    diagnostic: PromotionDiagnostic | None = None


class CandidateLifecycleService:
    """Project lifecycle state without runtime, permission, or materialization effects."""

    def __init__(self, store: EvoEventStore | _PromotionStore) -> None:
        self._store = store

    def state(self, artifact_id: str) -> CandidateArtifactLifecycle:
        require_evo_id(artifact_id, field="artifact_id", kind="artifact")
        events = self._store.list_events()
        _artifact(events, artifact_id)
        state = CandidateArtifactLifecycle.CANDIDATE
        for event in events:
            if event.event_type != "PromotionChanged" or not isinstance(event.payload, PromotionChangedPayload):
                continue
            if event.refs.artifact_id == artifact_id:
                state = CandidateArtifactLifecycle(event.payload.to_state)
        return state

    def start_shadow(self, artifact_id: str, *, reviewer: str, reason: str) -> PromotionResult:
        events = self._store.list_events()
        _artifact(events, artifact_id)
        if self.state(artifact_id) is not CandidateArtifactLifecycle.CANDIDATE:
            raise PromotionError("only a Candidate Artifact can enter Shadow")
        if not any(
            event.event_type == "CandidateShadowTrialRecorded" and isinstance(event.payload, CandidateShadowTrialRecordedPayload) and event.payload.artifact_id == artifact_id for event in events
        ):
            raise PromotionError("entering Shadow requires at least one recorded suggestion or Shadow Trial")
        return self._transition(
            artifact_id,
            from_state=CandidateArtifactLifecycle.CANDIDATE,
            to_state=CandidateArtifactLifecycle.SHADOW,
            reviewer=reviewer,
            reason=reason,
        )

    def validate_from_report(self, artifact_id: str, report_id: str) -> PromotionResult:
        require_evo_id(report_id, field="report_id", kind="evaluation")
        events = self._store.list_events()
        if self.state(artifact_id) is not CandidateArtifactLifecycle.SHADOW:
            raise PromotionError("only a Shadow Artifact can become Validated")
        artifact = _artifact(events, artifact_id)
        report = _report(events, report_id)
        _require_latest_report(events, artifact_id, report.event_id)
        if report.payload.artifact_id != artifact_id or report.payload.artifact_version != artifact.payload.artifact_version or not report.payload.eligible:
            raise PromotionError("Validated requires an eligible comparison report for this Artifact")
        return self._transition(
            artifact_id,
            from_state=CandidateArtifactLifecycle.SHADOW,
            to_state=CandidateArtifactLifecycle.VALIDATED,
            reviewer=None,
            reason="Deterministic held-out Promotion Gate passed.",
            evaluation_event_ids=(report.event_id,),
        )

    def approve(
        self,
        artifact_id: str,
        report_id: str,
        *,
        reviewer: str,
        reviewer_role: str,
        reason: str,
    ) -> PromotionResult:
        if reviewer_role not in PROMOTION_APPROVER_ROLES:
            raise PromotionError("Promoted requires a maintainer or owner approver")
        events = self._store.list_events()
        if self.state(artifact_id) is not CandidateArtifactLifecycle.VALIDATED:
            raise PromotionError("only a Validated Artifact can become Promoted")
        artifact = _artifact(events, artifact_id)
        report = _report(events, report_id)
        _require_latest_report(events, artifact_id, report.event_id)
        if report.payload.artifact_id != artifact_id or report.payload.artifact_version != artifact.payload.artifact_version or not report.payload.eligible:
            raise PromotionError("Promoted requires an eligible report for this Artifact")
        return self._transition(
            artifact_id,
            from_state=CandidateArtifactLifecycle.VALIDATED,
            to_state=CandidateArtifactLifecycle.PROMOTED,
            reviewer=reviewer,
            reason=reason,
            evaluation_event_ids=(report.event_id,),
            extensions={"reviewer_role": reviewer_role, "permissions_changed": False, "materialized": False},
        )

    def reject_from_report(self, artifact_id: str, report_id: str, *, reason: str) -> PromotionResult:
        events = self._store.list_events()
        current = self.state(artifact_id)
        if current not in {CandidateArtifactLifecycle.SHADOW, CandidateArtifactLifecycle.VALIDATED}:
            raise PromotionError("only a Shadow or Validated Artifact can be Rejected")
        artifact = _artifact(events, artifact_id)
        report = _report(events, report_id)
        _require_latest_report(events, artifact_id, report.event_id)
        if report.payload.artifact_id != artifact_id or report.payload.artifact_version != artifact.payload.artifact_version or report.payload.eligible:
            raise PromotionError("Rejected requires an ineligible report for this Artifact")
        return self._transition(
            artifact_id,
            from_state=current,
            to_state=CandidateArtifactLifecycle.REJECTED,
            reviewer=None,
            reason=reason,
            evaluation_event_ids=(report.event_id,),
            extensions={
                "automatic_rejection": True,
                "integrity_violations": list(report.payload.integrity_violations),
                "risk_event_count": report.payload.candidate_risk_event_count,
            },
        )

    def rollback_on_regression(
        self,
        artifact_id: str,
        report_id: str,
        *,
        impact_scope: str,
        reason: str,
    ) -> PromotionResult:
        if not isinstance(impact_scope, str) or not impact_scope.strip():
            raise PromotionError("impact_scope must be a non-blank string")
        events = self._store.list_events()
        if self.state(artifact_id) is not CandidateArtifactLifecycle.PROMOTED:
            raise PromotionError("rollback requires the currently Promoted Artifact")
        artifact = _artifact(events, artifact_id)
        report = _report(events, report_id)
        _require_latest_report(events, artifact_id, report.event_id)
        if report.payload.artifact_id != artifact_id or report.payload.artifact_version != artifact.payload.artifact_version or report.payload.eligible:
            raise PromotionError("rollback requires an ineligible regression report for this Artifact")
        target = _previous_promoted(events, artifact_id)
        critical = bool(report.payload.integrity_violations or report.payload.candidate_risk_event_count)
        return self._transition(
            artifact_id,
            from_state=CandidateArtifactLifecycle.PROMOTED,
            to_state=CandidateArtifactLifecycle.DEPRECATED,
            reviewer=None,
            reason=reason,
            evaluation_event_ids=(report.event_id,),
            rollback_target=None if target is None else target.artifact_id,
            extensions={
                "automatic_rollback": True,
                "impact_scope": redact_text(impact_scope.strip())[0],
                "rollback_sla": "immediate" if critical else "logical-disable-now-review-within-24h",
                "materialized_side_effects_reverted": False,
            },
        )

    def active_promoted(self) -> tuple[CandidateArtifactRecord, ...]:
        events = self._store.list_events()
        artifacts = {
            event.refs.artifact_id: CandidateArtifactRecord(
                event.event_id,
                event.refs.artifact_id,
                event.refs.run_id,
                event.occurred_at,
                event.payload,
            )
            for event in events
            if event.event_type == "CandidateArtifactCreated" and isinstance(event.payload, CandidateArtifactCreatedPayload) and event.refs.artifact_id is not None
        }
        selected: dict[str, str] = {}
        for event in events:
            if event.event_type != "PromotionChanged" or not isinstance(event.payload, PromotionChangedPayload):
                continue
            artifact_id = event.refs.artifact_id
            if artifact_id is None or artifact_id not in artifacts:
                continue
            lineage_id = artifacts[artifact_id].payload.lineage_id
            if event.payload.to_state == CandidateArtifactLifecycle.PROMOTED.value:
                selected[lineage_id] = artifact_id
            elif event.payload.to_state == CandidateArtifactLifecycle.DEPRECATED.value:
                if event.payload.rollback_target is not None:
                    selected[lineage_id] = event.payload.rollback_target
                elif selected.get(lineage_id) == artifact_id:
                    selected.pop(lineage_id, None)
        return tuple(artifacts[artifact_id] for _, artifact_id in sorted(selected.items()))

    def _transition(
        self,
        artifact_id: str,
        *,
        from_state: CandidateArtifactLifecycle,
        to_state: CandidateArtifactLifecycle,
        reviewer: str | None,
        reason: str,
        evaluation_event_ids: tuple[str, ...] = (),
        rollback_target: str | None = None,
        extensions: dict[str, object] | None = None,
    ) -> PromotionResult:
        events = self._store.list_events()
        artifact = _artifact(events, artifact_id)
        current = self.state(artifact_id)
        if current != from_state:
            raise PromotionError(f"transition expected {from_state.value}, found {current.value}")
        if not isinstance(reason, str) or not reason.strip():
            raise PromotionError("transition reason must be a non-blank string")
        if reviewer is not None and (not isinstance(reviewer, str) or not reviewer.strip()):
            raise PromotionError("reviewer must be a non-blank string")
        promotion_id = new_evo_id("promotion")
        payload = PromotionChangedPayload(
            from_state=from_state.value,
            to_state=to_state.value,
            reason=redact_text(reason.strip())[0],
            reviewer=None if reviewer is None else redact_text(reviewer.strip())[0],
            evaluation_event_ids=evaluation_event_ids,
            rollback_target=rollback_target,
            extensions={"service_version": "p8-003", "runtime_permissions_changed": False, **(extensions or {})},
        )
        event = EvoEvent(
            event_id=new_evo_id("event"),
            event_type="PromotionChanged",
            refs=EvoReferences(
                run_id=artifact.run_id,
                artifact_id=artifact_id,
                promotion_id=promotion_id,
                parent_event_id=_latest_artifact_event(events, artifact_id) or artifact.event_id,
            ),
            payload=payload,
        )
        try:
            appended = self._store.append(event)
        except EvoStoreError as error:
            return PromotionResult(False, diagnostic=PromotionDiagnostic("promotion_recording_failed", str(error)))
        except Exception as error:  # noqa: BLE001 - lifecycle recording cannot grant permissions or run the Artifact
            return PromotionResult(
                False,
                diagnostic=PromotionDiagnostic("promotion_recording_failed", f"unexpected lifecycle recorder failure: {error}"),
            )
        diagnostic = None
        if appended.diagnostic is not None:
            diagnostic = PromotionDiagnostic(appended.diagnostic.code, appended.diagnostic.message)
        return PromotionResult(True, _promotion_from_event(appended.event), diagnostic)


class PromotionGate:
    """Validate metrics automatically, but never approve Promotion automatically."""

    def __init__(self, store: EvoEventStore | _PromotionStore) -> None:
        self._store = store

    def evaluate_and_validate(self, spec: EvaluationComparisonSpec) -> PromotionGateResult:
        lifecycle = CandidateLifecycleService(self._store)
        if lifecycle.state(spec.artifact_id) is not CandidateArtifactLifecycle.SHADOW:
            raise PromotionError("Promotion Gate requires an Artifact in Shadow")
        comparison = EvaluationComparisonService(self._store).compare(spec)
        if comparison.report is None:
            diagnostic = comparison.diagnostic
            return PromotionGateResult(
                False,
                diagnostic=PromotionDiagnostic(
                    diagnostic.code if diagnostic else "comparison_failed",
                    diagnostic.message if diagnostic else "comparison failed without a diagnostic",
                ),
            )
        if not comparison.report.payload.eligible:
            if _requires_rejection(comparison.report.payload):
                rejected = lifecycle.reject_from_report(
                    spec.artifact_id,
                    comparison.report.payload.report_id,
                    reason="Candidate showed a material held-out regression or reward-integrity violation.",
                )
                return PromotionGateResult(
                    False,
                    comparison.report,
                    rejected.promotion,
                    rejected.diagnostic,
                )
            return PromotionGateResult(False, comparison.report)
        transition = lifecycle.validate_from_report(spec.artifact_id, comparison.report.payload.report_id)
        return PromotionGateResult(
            transition.persisted,
            comparison.report,
            transition.promotion,
            transition.diagnostic,
        )


def _artifact(events: list[EvoEvent], artifact_id: str) -> CandidateArtifactRecord:
    require_evo_id(artifact_id, field="artifact_id", kind="artifact")
    event = next(
        (event for event in events if event.event_type == "CandidateArtifactCreated" and isinstance(event.payload, CandidateArtifactCreatedPayload) and event.refs.artifact_id == artifact_id),
        None,
    )
    if event is None or event.refs.artifact_id is None:
        raise PromotionError(f"unknown Artifact: {artifact_id}")
    return CandidateArtifactRecord(event.event_id, event.refs.artifact_id, event.refs.run_id, event.occurred_at, event.payload)


def _report(events: list[EvoEvent], report_id: str) -> EvoEvent[EvaluationComparisonCompletedPayload]:
    event = next(
        (event for event in events if event.event_type == "EvaluationComparisonCompleted" and isinstance(event.payload, EvaluationComparisonCompletedPayload) and event.payload.report_id == report_id),
        None,
    )
    if event is None:
        raise PromotionError(f"unknown evaluation report: {report_id}")
    return event


def _latest_artifact_event(events: list[EvoEvent], artifact_id: str) -> str | None:
    return next((event.event_id for event in reversed(events) if event.refs.artifact_id == artifact_id), None)


def _require_latest_report(events: list[EvoEvent], artifact_id: str, event_id: str) -> None:
    latest = next(
        (
            event
            for event in reversed(events)
            if event.event_type == "EvaluationComparisonCompleted" and isinstance(event.payload, EvaluationComparisonCompletedPayload) and event.payload.artifact_id == artifact_id
        ),
        None,
    )
    if latest is None or latest.event_id != event_id:
        raise PromotionError("lifecycle decisions require the latest evaluation report for this Artifact")


def _previous_promoted(events: list[EvoEvent], artifact_id: str) -> CandidateArtifactRecord | None:
    current = _artifact(events, artifact_id)
    artifacts = {
        event.refs.artifact_id: CandidateArtifactRecord(
            event.event_id,
            event.refs.artifact_id,
            event.refs.run_id,
            event.occurred_at,
            event.payload,
        )
        for event in events
        if event.event_type == "CandidateArtifactCreated"
        and isinstance(event.payload, CandidateArtifactCreatedPayload)
        and event.refs.artifact_id is not None
        and event.payload.lineage_id == current.payload.lineage_id
        and event.payload.artifact_version < current.payload.artifact_version
    }
    states = {candidate_id: CandidateArtifactLifecycle.CANDIDATE for candidate_id in artifacts}
    for event in events:
        if event.event_type == "PromotionChanged" and isinstance(event.payload, PromotionChangedPayload) and event.refs.artifact_id in states:
            states[event.refs.artifact_id] = CandidateArtifactLifecycle(event.payload.to_state)
    candidates = [artifact for candidate_id, artifact in artifacts.items() if states[candidate_id] is CandidateArtifactLifecycle.PROMOTED]
    return max(candidates, key=lambda artifact: artifact.payload.artifact_version, default=None)


def _requires_rejection(report: EvaluationComparisonCompletedPayload) -> bool:
    if report.integrity_violations or report.candidate_risk_event_count:
        return True
    if report.candidate_success_rate < report.baseline_success_rate:
        return True
    if report.candidate_verification_quality < report.baseline_verification_quality:
        return True
    severe_markers = ("cost exceeds", "latency exceeds")
    return any(any(marker in reason.lower() for marker in severe_markers) for reason in report.blocking_reasons)


def _promotion_from_event(event: EvoEvent[PromotionChangedPayload]) -> PromotionRecord:
    if event.refs.promotion_id is None or event.refs.artifact_id is None:
        raise PromotionError("PromotionChanged requires promotion_id and artifact_id")
    return PromotionRecord(
        event.event_id,
        event.refs.promotion_id,
        event.refs.artifact_id,
        event.refs.run_id,
        event.occurred_at,
        event.payload,
    )
