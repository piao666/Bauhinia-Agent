"""Suggestion-only and Shadow controls for non-promoted Candidate Artifacts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Protocol

from bauhinia_agent.evolution.candidate_artifacts import CandidateArtifactError, CandidateArtifactRecord
from bauhinia_agent.evolution.evidence import redact_text
from bauhinia_agent.evolution.events import (
    CandidateArtifactControlChangedPayload,
    CandidateArtifactCreatedPayload,
    CandidateShadowTrialRecordedPayload,
    EvoEvent,
    EvoReferences,
)
from bauhinia_agent.evolution.identifiers import new_evo_id, require_evo_id
from bauhinia_agent.evolution.store import EvoAppendResult, EvoEventStore, EvoStoreError

ShadowMode = Literal["suggestion", "shadow"]
ArtifactControlAction = Literal["disable_shadow", "resume_shadow", "rollback_shadow"]
_HASH = re.compile(r"[0-9a-f]{16,128}\Z")


class ArtifactShadowError(ValueError):
    """Raised when a Shadow action violates isolation or version constraints."""


class _ArtifactShadowStore(Protocol):
    def append(self, event: EvoEvent) -> EvoAppendResult: ...

    def list_events(self) -> list[EvoEvent]: ...


@dataclass(frozen=True, slots=True)
class ShadowTrialSpec:
    artifact_id: str
    mode: ShadowMode
    task_input_hash: str
    workspace_baseline_hash: str
    environment_hash: str
    baseline_summary: str
    candidate_summary: str
    evidence_refs: tuple[str, ...]
    passed: bool
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ShadowTrialRecord:
    event_id: str
    run_id: str
    occurred_at: str
    payload: CandidateShadowTrialRecordedPayload


@dataclass(frozen=True, slots=True)
class ArtifactControlRequest:
    artifact_id: str
    action: ArtifactControlAction
    reviewer: str
    reason: str
    evidence_refs: tuple[str, ...]
    target_artifact_id: str | None = None


@dataclass(frozen=True, slots=True)
class ArtifactControlRecord:
    event_id: str
    run_id: str
    occurred_at: str
    payload: CandidateArtifactControlChangedPayload


@dataclass(frozen=True, slots=True)
class ShadowSuggestion:
    artifact_id: str
    lineage_id: str
    artifact_version: int
    kind: str
    name: str
    instructions: str
    effect_risk: str


@dataclass(frozen=True, slots=True)
class ArtifactShadowDiagnostic:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class ShadowTrialResult:
    persisted: bool
    trial: ShadowTrialRecord | None = None
    diagnostic: ArtifactShadowDiagnostic | None = None


@dataclass(frozen=True, slots=True)
class ArtifactControlResult:
    persisted: bool
    control: ArtifactControlRecord | None = None
    diagnostic: ArtifactShadowDiagnostic | None = None


@dataclass(slots=True)
class _ShadowState:
    artifacts: dict[str, CandidateArtifactRecord]
    selected_by_lineage: dict[str, str]
    disabled: set[str]


class ArtifactShadowService:
    """Record comparisons and control Shadow visibility without executing actions."""

    def __init__(self, store: EvoEventStore | _ArtifactShadowStore) -> None:
        self._store = store

    def list_suggestions(self) -> tuple[ShadowSuggestion, ...]:
        """Return explicitly requested suggestions; this is not runtime retrieval."""

        state = _shadow_state(self._store.list_events())
        suggestions: list[ShadowSuggestion] = []
        for lineage_id, artifact_id in sorted(state.selected_by_lineage.items()):
            if artifact_id in state.disabled:
                continue
            artifact = state.artifacts[artifact_id]
            suggestions.append(
                ShadowSuggestion(
                    artifact_id=artifact.artifact_id,
                    lineage_id=lineage_id,
                    artifact_version=artifact.payload.artifact_version,
                    kind=artifact.payload.kind,
                    name=artifact.payload.name,
                    instructions=artifact.payload.instructions,
                    effect_risk=artifact.effect_risk.value,
                )
            )
        return tuple(suggestions)

    def list_for_runtime(self) -> tuple[()]:
        """P7 Artifacts never enter ordinary runtime execution or retrieval."""

        return ()

    def record_trial(self, spec: ShadowTrialSpec) -> ShadowTrialResult:
        _validate_trial(spec)
        events = self._store.list_events()
        state = _shadow_state(events)
        artifact = state.artifacts.get(spec.artifact_id)
        if artifact is None:
            raise ArtifactShadowError(f"unknown Artifact: {spec.artifact_id}")
        selected = state.selected_by_lineage.get(artifact.payload.lineage_id)
        if selected != artifact.artifact_id or artifact.artifact_id in state.disabled:
            raise ArtifactShadowError("Artifact is not enabled for suggestion or Shadow comparison")
        payload = CandidateShadowTrialRecordedPayload(
            trial_id=new_evo_id("shadow_trial"),
            artifact_id=artifact.artifact_id,
            artifact_version=artifact.payload.artifact_version,
            mode=spec.mode,
            task_input_hash=spec.task_input_hash,
            workspace_baseline_hash=spec.workspace_baseline_hash,
            environment_hash=spec.environment_hash,
            baseline_summary=redact_text(spec.baseline_summary)[0],
            candidate_summary=redact_text(spec.candidate_summary)[0],
            evidence_refs=spec.evidence_refs,
            passed=spec.passed,
            real_effects_applied=False,
            failure_reason=None if spec.failure_reason is None else redact_text(spec.failure_reason)[0],
            extensions={"service_version": "p7-003", "execution_mode": "observe_only"},
        )
        event = EvoEvent(
            event_id=new_evo_id("event"),
            event_type="CandidateShadowTrialRecorded",
            refs=EvoReferences(
                run_id=artifact.run_id,
                artifact_id=artifact.artifact_id,
                parent_event_id=_latest_related_event_id(events, artifact.artifact_id) or artifact.event_id,
            ),
            payload=payload,
        )
        return self._append_trial(event)

    def control(self, request: ArtifactControlRequest) -> ArtifactControlResult:
        _validate_control(request)
        events = self._store.list_events()
        state = _shadow_state(events)
        artifact = state.artifacts.get(request.artifact_id)
        if artifact is None:
            raise ArtifactShadowError(f"unknown Artifact: {request.artifact_id}")
        selected = state.selected_by_lineage.get(artifact.payload.lineage_id)
        if request.action in {"disable_shadow", "rollback_shadow"} and selected != artifact.artifact_id:
            raise ArtifactShadowError("control action requires the selected Artifact version")
        if request.action == "resume_shadow" and artifact.artifact_id not in state.disabled:
            raise ArtifactShadowError("Artifact is not disabled")
        target = None
        if request.action == "rollback_shadow":
            if request.target_artifact_id is None:
                raise ArtifactShadowError("rollback_shadow requires target_artifact_id")
            target = state.artifacts.get(request.target_artifact_id)
            if target is None:
                raise ArtifactShadowError(f"unknown rollback target: {request.target_artifact_id}")
            if target.payload.lineage_id != artifact.payload.lineage_id:
                raise ArtifactShadowError("rollback target must belong to the same Artifact lineage")
            if target.payload.artifact_version >= artifact.payload.artifact_version:
                raise ArtifactShadowError("rollback target must be an earlier Artifact version")
        payload = CandidateArtifactControlChangedPayload(
            artifact_id=artifact.artifact_id,
            artifact_version=artifact.payload.artifact_version,
            action=request.action,
            reviewer=request.reviewer,
            reason=redact_text(request.reason)[0],
            evidence_refs=request.evidence_refs,
            target_artifact_id=target.artifact_id if target is not None else None,
            target_artifact_version=target.payload.artifact_version if target is not None else None,
            extensions={"service_version": "p7-003", "runtime_permissions_changed": False},
        )
        event = EvoEvent(
            event_id=new_evo_id("event"),
            event_type="CandidateArtifactControlChanged",
            refs=EvoReferences(
                run_id=artifact.run_id,
                artifact_id=artifact.artifact_id,
                parent_event_id=_latest_related_event_id(events, artifact.artifact_id) or artifact.event_id,
            ),
            payload=payload,
        )
        return self._append_control(event)

    def list_trials(self, artifact_id: str | None = None) -> tuple[ShadowTrialRecord, ...]:
        if artifact_id is not None:
            require_evo_id(artifact_id, field="artifact_id", kind="artifact")
        return tuple(
            _trial_from_event(event)
            for event in self._store.list_events()
            if event.event_type == "CandidateShadowTrialRecorded"
            and isinstance(event.payload, CandidateShadowTrialRecordedPayload)
            and (artifact_id is None or event.payload.artifact_id == artifact_id)
        )

    def _append_trial(self, event: EvoEvent[CandidateShadowTrialRecordedPayload]) -> ShadowTrialResult:
        try:
            appended = self._store.append(event)
        except EvoStoreError as error:
            return ShadowTrialResult(False, diagnostic=ArtifactShadowDiagnostic("shadow_trial_recording_failed", str(error)))
        except Exception as error:  # noqa: BLE001 - recorder failure cannot change execution
            return ShadowTrialResult(
                False,
                diagnostic=ArtifactShadowDiagnostic("shadow_trial_recording_failed", f"unexpected Shadow recorder failure: {error}"),
            )
        diagnostic = None
        if appended.diagnostic is not None:
            diagnostic = ArtifactShadowDiagnostic(appended.diagnostic.code, appended.diagnostic.message)
        return ShadowTrialResult(True, _trial_from_event(appended.event), diagnostic)

    def _append_control(self, event: EvoEvent[CandidateArtifactControlChangedPayload]) -> ArtifactControlResult:
        try:
            appended = self._store.append(event)
        except EvoStoreError as error:
            return ArtifactControlResult(False, diagnostic=ArtifactShadowDiagnostic("artifact_control_recording_failed", str(error)))
        except Exception as error:  # noqa: BLE001 - control recording cannot alter source facts or permissions
            return ArtifactControlResult(
                False,
                diagnostic=ArtifactShadowDiagnostic("artifact_control_recording_failed", f"unexpected Artifact control failure: {error}"),
            )
        diagnostic = None
        if appended.diagnostic is not None:
            diagnostic = ArtifactShadowDiagnostic(appended.diagnostic.code, appended.diagnostic.message)
        return ArtifactControlResult(True, _control_from_event(appended.event), diagnostic)


def _validate_trial(spec: ShadowTrialSpec) -> None:
    require_evo_id(spec.artifact_id, field="artifact_id", kind="artifact")
    if spec.mode not in {"suggestion", "shadow"}:
        raise ArtifactShadowError("mode must be suggestion or shadow")
    for field, value in (
        ("task_input_hash", spec.task_input_hash),
        ("workspace_baseline_hash", spec.workspace_baseline_hash),
        ("environment_hash", spec.environment_hash),
    ):
        if not isinstance(value, str) or not _HASH.fullmatch(value):
            raise ArtifactShadowError(f"{field} must be a lowercase hexadecimal digest")
    for field, value in (("baseline_summary", spec.baseline_summary), ("candidate_summary", spec.candidate_summary)):
        if not isinstance(value, str) or not value.strip():
            raise ArtifactShadowError(f"{field} must be a non-blank string")
    _evidence_refs(spec.evidence_refs)
    if not isinstance(spec.passed, bool):
        raise ArtifactShadowError("passed must be a boolean")
    if not spec.passed and (not isinstance(spec.failure_reason, str) or not spec.failure_reason.strip()):
        raise ArtifactShadowError("failed Shadow trials require failure_reason")
    if spec.passed and spec.failure_reason is not None:
        raise ArtifactShadowError("successful Shadow trials cannot include failure_reason")


def _validate_control(request: ArtifactControlRequest) -> None:
    require_evo_id(request.artifact_id, field="artifact_id", kind="artifact")
    if request.action not in {"disable_shadow", "resume_shadow", "rollback_shadow"}:
        raise ArtifactShadowError("unsupported Artifact control action")
    for field, value in (("reviewer", request.reviewer), ("reason", request.reason)):
        if not isinstance(value, str) or not value.strip():
            raise ArtifactShadowError(f"{field} must be a non-blank string")
    _evidence_refs(request.evidence_refs)


def _evidence_refs(values: tuple[str, ...]) -> None:
    if not isinstance(values, tuple) or not values:
        raise ArtifactShadowError("evidence_refs must be a non-empty tuple")
    for value in values:
        require_evo_id(value, field="evidence_refs[]", kind="evidence")
    if len(set(values)) != len(values):
        raise ArtifactShadowError("evidence_refs must be unique")


def _shadow_state(events: list[EvoEvent]) -> _ShadowState:
    state = _ShadowState({}, {}, set())
    for event in events:
        if event.event_type == "CandidateArtifactCreated" and isinstance(event.payload, CandidateArtifactCreatedPayload) and event.refs.artifact_id is not None:
            artifact = CandidateArtifactRecord(
                event.event_id,
                event.refs.artifact_id,
                event.refs.run_id,
                event.occurred_at,
                event.payload,
            )
            state.artifacts[artifact.artifact_id] = artifact
            state.selected_by_lineage[artifact.payload.lineage_id] = artifact.artifact_id
        elif event.event_type == "CandidateArtifactControlChanged" and isinstance(event.payload, CandidateArtifactControlChangedPayload):
            payload = event.payload
            artifact = state.artifacts.get(payload.artifact_id)
            if artifact is None:
                raise CandidateArtifactError(f"control references unknown Artifact: {payload.artifact_id}")
            if payload.action == "disable_shadow":
                state.disabled.add(payload.artifact_id)
            elif payload.action == "resume_shadow":
                state.disabled.discard(payload.artifact_id)
            elif payload.action == "rollback_shadow":
                if payload.target_artifact_id is None:
                    raise CandidateArtifactError("rollback control requires target_artifact_id")
                state.selected_by_lineage[artifact.payload.lineage_id] = payload.target_artifact_id
    return state


def _latest_related_event_id(events: list[EvoEvent], artifact_id: str) -> str | None:
    for event in reversed(events):
        if event.refs.artifact_id == artifact_id:
            return event.event_id
    return None


def _trial_from_event(event: EvoEvent[CandidateShadowTrialRecordedPayload]) -> ShadowTrialRecord:
    return ShadowTrialRecord(event.event_id, event.refs.run_id, event.occurred_at, event.payload)


def _control_from_event(event: EvoEvent[CandidateArtifactControlChangedPayload]) -> ArtifactControlRecord:
    return ArtifactControlRecord(event.event_id, event.refs.run_id, event.occurred_at, event.payload)
