"""Versioned, non-operative Candidate Artifact contracts and registry."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from bauhinia_agent.evolution.evidence import redact_text
from bauhinia_agent.evolution.events import (
    CandidateArtifactCreatedPayload,
    CandidateReviewRecordedPayload,
    EvoEvent,
    EvoReferences,
    ExperienceCandidateCreatedPayload,
)
from bauhinia_agent.evolution.identifiers import new_evo_id, require_evo_id
from bauhinia_agent.evolution.store import EvoAppendResult, EvoEventStore, EvoStoreError

CANDIDATE_ARTIFACT_SCHEMA_VERSION = "v1"
_SAFE_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")
_HIGH_RISK_EFFECTS = frozenset({"write", "execute", "network", "external", "unknown"})
_KNOWN_EFFECTS = frozenset({"none", "read", *_HIGH_RISK_EFFECTS})


class CandidateArtifactKind(StrEnum):
    PLAN_TEMPLATE = "plan_template"
    SKILL_DRAFT = "skill_draft"
    TOOL_INVOCATION_POLICY = "tool_invocation_policy"
    MEMORY_RULE = "memory_rule"


class CandidateArtifactLifecycle(StrEnum):
    CANDIDATE = "Candidate"
    SHADOW = "Shadow"
    VALIDATED = "Validated"
    PROMOTED = "Promoted"
    DEPRECATED = "Deprecated"
    REJECTED = "Rejected"


class CandidateEffectRisk(StrEnum):
    LOW = "low"
    HIGH = "high"


class CandidateArtifactError(ValueError):
    """Raised when an Artifact would violate traceability or isolation."""


class _CandidateArtifactStore(Protocol):
    def append(self, event: EvoEvent[CandidateArtifactCreatedPayload]) -> EvoAppendResult: ...

    def list_events(self) -> list[EvoEvent]: ...


@dataclass(frozen=True, slots=True)
class CandidateArtifactDraft:
    kind: CandidateArtifactKind
    name: str
    description: str
    instructions: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    dependencies: tuple[str, ...]
    effects: tuple[str, ...]
    scope: str
    applicability: str
    risks: tuple[str, ...]
    source_candidate_ids: tuple[str, ...]
    confidence: float
    triggers: tuple[str, ...] = ()
    support_candidate_ids: tuple[str, ...] = ()
    counterexample_candidate_ids: tuple[str, ...] = ()
    supersedes_artifact_id: str | None = None


@dataclass(frozen=True, slots=True)
class CandidateArtifactRecord:
    event_id: str
    artifact_id: str
    run_id: str
    occurred_at: str
    payload: CandidateArtifactCreatedPayload

    @property
    def effect_risk(self) -> CandidateEffectRisk:
        return effect_risk(self.payload.effects)

    def minimal_metadata(self) -> dict[str, object]:
        """Export review metadata without instructions, dependencies, or secrets."""

        return {
            "artifact_id": self.artifact_id,
            "artifact_schema_version": self.payload.artifact_schema_version,
            "lineage_id": self.payload.lineage_id,
            "artifact_version": self.payload.artifact_version,
            "kind": self.payload.kind,
            "name": self.payload.name,
            "scope": self.payload.scope,
            "effects": list(self.payload.effects),
            "effect_risk": self.effect_risk.value,
            "source_run_ids": list(self.payload.source_run_ids),
            "lifecycle_state": self.payload.lifecycle_state,
            "content_hash": self.payload.content_hash,
        }


@dataclass(frozen=True, slots=True)
class CandidateArtifactManifest:
    artifacts: tuple[CandidateArtifactRecord, ...]

    def versions(self, lineage_id: str) -> tuple[CandidateArtifactRecord, ...]:
        require_evo_id(lineage_id, field="lineage_id", kind="artifact")
        return tuple(item for item in self.artifacts if item.payload.lineage_id == lineage_id)

    def latest(self, lineage_id: str) -> CandidateArtifactRecord | None:
        versions = self.versions(lineage_id)
        return versions[-1] if versions else None


@dataclass(frozen=True, slots=True)
class CandidateArtifactDiagnostic:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class CandidateArtifactResult:
    persisted: bool
    artifact: CandidateArtifactRecord | None = None
    diagnostic: CandidateArtifactDiagnostic | None = None


class CandidateArtifactRegistry:
    """Append versioned Artifact facts without enabling runtime use."""

    def __init__(self, store: EvoEventStore | _CandidateArtifactStore) -> None:
        self._store = store

    def create(self, draft: CandidateArtifactDraft) -> CandidateArtifactResult:
        events = self._store.list_events()
        source_events = _source_candidates(events, draft.source_candidate_ids)
        accepted = _accepted_candidate_ids(events)
        missing_review = sorted(set(draft.source_candidate_ids) - accepted)
        if missing_review:
            raise CandidateArtifactError(f"source Candidates require human acceptance: {missing_review}")
        artifacts = _artifact_events(events)
        previous = None
        if draft.supersedes_artifact_id is not None:
            require_evo_id(draft.supersedes_artifact_id, field="supersedes_artifact_id", kind="artifact")
            previous = artifacts.get(draft.supersedes_artifact_id)
            if previous is None:
                raise CandidateArtifactError(f"unknown superseded Artifact: {draft.supersedes_artifact_id}")
            latest = max(
                (event for event in artifacts.values() if event.payload.lineage_id == previous.payload.lineage_id),
                key=lambda event: event.payload.artifact_version,
            )
            if latest.refs.artifact_id != draft.supersedes_artifact_id:
                raise CandidateArtifactError("a new Artifact version must supersede the latest lineage version")

        artifact_id = new_evo_id("artifact")
        lineage_id = previous.payload.lineage_id if previous is not None else artifact_id
        artifact_version = previous.payload.artifact_version + 1 if previous is not None else 1
        payload = _artifact_payload(draft, source_events, lineage_id=lineage_id, artifact_version=artifact_version)
        if previous is not None:
            _validate_successor(previous.payload, payload)
        event = EvoEvent(
            event_id=new_evo_id("event"),
            event_type="CandidateArtifactCreated",
            refs=EvoReferences(
                run_id=payload.source_run_ids[0],
                artifact_id=artifact_id,
                parent_event_id=previous.event_id if previous is not None else source_events[0].event_id,
            ),
            payload=payload,
        )
        try:
            appended = self._store.append(event)
        except EvoStoreError as error:
            return CandidateArtifactResult(False, diagnostic=CandidateArtifactDiagnostic("artifact_recording_failed", str(error)))
        except Exception as error:  # noqa: BLE001 - Artifact recording cannot affect source Candidates or Runs
            return CandidateArtifactResult(
                False,
                diagnostic=CandidateArtifactDiagnostic("artifact_recording_failed", f"unexpected Artifact recorder failure: {error}"),
            )
        diagnostic = None
        if appended.diagnostic is not None:
            diagnostic = CandidateArtifactDiagnostic(appended.diagnostic.code, appended.diagnostic.message)
        return CandidateArtifactResult(True, _artifact_from_event(appended.event), diagnostic)

    def manifest(self) -> CandidateArtifactManifest:
        records = tuple(
            _artifact_from_event(event)
            for event in self._store.list_events()
            if event.event_type == "CandidateArtifactCreated" and isinstance(event.payload, CandidateArtifactCreatedPayload) and event.refs.artifact_id is not None
        )
        return CandidateArtifactManifest(records)


def effect_risk(effects: tuple[str, ...]) -> CandidateEffectRisk:
    """Unknown Effects fail closed as high risk."""

    normalized = {item.strip().lower() for item in effects}
    if any(item not in _KNOWN_EFFECTS or item in _HIGH_RISK_EFFECTS for item in normalized):
        return CandidateEffectRisk.HIGH
    return CandidateEffectRisk.LOW


def render_skill_markdown(artifact: CandidateArtifactRecord) -> str:
    """Adapt a Skill Draft in memory; never write to a discoverable Skill root."""

    if artifact.payload.kind != CandidateArtifactKind.SKILL_DRAFT.value:
        raise CandidateArtifactError("only skill_draft Artifacts can render SKILL.md content")
    lines = [
        "---",
        f"name: {json.dumps(artifact.payload.name, ensure_ascii=False)}",
        f"description: {json.dumps(artifact.payload.description, ensure_ascii=False)}",
        "triggers:",
        *(f"  - {json.dumps(trigger, ensure_ascii=False)}" for trigger in artifact.payload.triggers),
        "---",
        "",
        artifact.payload.instructions,
    ]
    return "\n".join(lines).rstrip() + "\n"


def _artifact_payload(
    draft: CandidateArtifactDraft,
    source_events: tuple[EvoEvent[ExperienceCandidateCreatedPayload], ...],
    *,
    lineage_id: str,
    artifact_version: int,
) -> CandidateArtifactCreatedPayload:
    kind = _artifact_kind(draft.kind)
    name = _safe_name(draft.name)
    description = _required_redacted(draft.description, field="description")
    instructions = _required_redacted(draft.instructions, field="instructions")
    inputs = _redacted_items(draft.inputs, field="inputs", required=True)
    outputs = _redacted_items(draft.outputs, field="outputs", required=True)
    dependencies = _redacted_items(draft.dependencies, field="dependencies")
    effects = _redacted_items(draft.effects, field="effects", required=True, lowercase=True)
    triggers = _redacted_items(draft.triggers, field="triggers")
    scope = _required_redacted(draft.scope, field="scope")
    applicability = _required_redacted(draft.applicability, field="applicability")
    risks = _redacted_items(draft.risks, field="risks", required=True)
    confidence = _confidence(draft.confidence)
    source_candidate_ids = tuple(event.refs.candidate_id for event in source_events if event.refs.candidate_id is not None)
    support_candidate_ids = draft.support_candidate_ids or source_candidate_ids
    counterexample_candidate_ids = draft.counterexample_candidate_ids
    _validate_source_roles(source_candidate_ids, support_candidate_ids, counterexample_candidate_ids)
    source_run_ids = tuple(sorted({run_id for event in source_events for run_id in (*event.payload.source_run_ids, event.refs.run_id)}))
    evidence_refs = tuple(sorted({ref for event in source_events for ref in event.payload.evidence_refs}))
    counterexamples = tuple(sorted({item for event in source_events for item in event.payload.counterexamples}))
    content = {
        "kind": kind.value,
        "name": name,
        "description": description,
        "instructions": instructions,
        "inputs": inputs,
        "outputs": outputs,
        "dependencies": dependencies,
        "effects": effects,
        "triggers": triggers,
        "scope": scope,
        "applicability": applicability,
        "risks": risks,
    }
    content_hash = hashlib.sha256(json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return CandidateArtifactCreatedPayload(
        artifact_schema_version=CANDIDATE_ARTIFACT_SCHEMA_VERSION,
        lineage_id=lineage_id,
        artifact_version=artifact_version,
        kind=kind.value,
        name=name,
        description=description,
        instructions=instructions,
        inputs=inputs,
        outputs=outputs,
        dependencies=dependencies,
        effects=effects,
        triggers=triggers,
        scope=scope,
        applicability=applicability,
        risks=risks,
        source_candidate_ids=source_candidate_ids,
        support_candidate_ids=support_candidate_ids,
        counterexample_candidate_ids=counterexample_candidate_ids,
        source_run_ids=source_run_ids,
        evidence_refs=evidence_refs,
        counterexamples=counterexamples,
        confidence=confidence,
        content_hash=content_hash,
        lifecycle_state=CandidateArtifactLifecycle.CANDIDATE.value,
        supersedes_artifact_id=draft.supersedes_artifact_id,
        extensions={"registry_version": "p7-001", "runtime_enabled": False},
    )


def _source_candidates(events: list[EvoEvent], candidate_ids: tuple[str, ...]) -> tuple[EvoEvent[ExperienceCandidateCreatedPayload], ...]:
    if not candidate_ids:
        raise CandidateArtifactError("source_candidate_ids must not be empty")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise CandidateArtifactError("source_candidate_ids must be unique")
    candidates = {
        event.refs.candidate_id: event
        for event in events
        if event.event_type == "ExperienceCandidateCreated" and isinstance(event.payload, ExperienceCandidateCreatedPayload) and event.refs.candidate_id is not None
    }
    missing = [candidate_id for candidate_id in candidate_ids if candidate_id not in candidates]
    if missing:
        raise CandidateArtifactError(f"unknown source Candidates: {missing}")
    return tuple(candidates[candidate_id] for candidate_id in candidate_ids)


def _accepted_candidate_ids(events: list[EvoEvent]) -> set[str]:
    latest: dict[str, CandidateReviewRecordedPayload] = {}
    for event in events:
        if event.event_type != "CandidateReviewRecorded" or not isinstance(event.payload, CandidateReviewRecordedPayload):
            continue
        latest[event.payload.candidate_id] = event.payload
    return {candidate_id for candidate_id, review in latest.items() if review.decision == "accept"}


def _artifact_events(events: list[EvoEvent]) -> dict[str, EvoEvent[CandidateArtifactCreatedPayload]]:
    return {
        event.refs.artifact_id: event
        for event in events
        if event.event_type == "CandidateArtifactCreated" and isinstance(event.payload, CandidateArtifactCreatedPayload) and event.refs.artifact_id is not None
    }


def _validate_successor(previous: CandidateArtifactCreatedPayload, current: CandidateArtifactCreatedPayload) -> None:
    if current.kind != previous.kind or current.name != previous.name or current.scope != previous.scope:
        raise CandidateArtifactError("an Artifact version cannot change kind, name, or scope")
    if current.content_hash == previous.content_hash:
        raise CandidateArtifactError("a new Artifact version must change its content")


def _validate_source_roles(
    source_candidate_ids: tuple[str, ...],
    support_candidate_ids: tuple[str, ...],
    counterexample_candidate_ids: tuple[str, ...],
) -> None:
    source_set = set(source_candidate_ids)
    support_set = set(support_candidate_ids)
    counterexample_set = set(counterexample_candidate_ids)
    if not support_set:
        raise CandidateArtifactError("support_candidate_ids must not be empty")
    if len(support_set) != len(support_candidate_ids) or len(counterexample_set) != len(counterexample_candidate_ids):
        raise CandidateArtifactError("support and counterexample Candidate IDs must be unique")
    if support_set & counterexample_set:
        raise CandidateArtifactError("support and counterexample Candidate IDs must not overlap")
    if support_set | counterexample_set != source_set:
        raise CandidateArtifactError("source Candidate IDs must equal support plus counterexample IDs")


def _artifact_kind(value: object) -> CandidateArtifactKind:
    try:
        return value if isinstance(value, CandidateArtifactKind) else CandidateArtifactKind(str(value))
    except ValueError as error:
        raise CandidateArtifactError(f"unsupported Artifact kind: {value}") from error


def _safe_name(value: object) -> str:
    if not isinstance(value, str) or not _SAFE_NAME.fullmatch(value):
        raise CandidateArtifactError("name must be a lowercase safe identifier")
    return value


def _required_redacted(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CandidateArtifactError(f"{field} must be a non-blank string")
    return redact_text(value.strip())[0]


def _redacted_items(
    values: object,
    *,
    field: str,
    required: bool = False,
    lowercase: bool = False,
) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise CandidateArtifactError(f"{field} must be a tuple of strings")
    result = tuple(_required_redacted(value, field=f"{field}[]") for value in values)
    if lowercase:
        result = tuple(value.lower() for value in result)
    if required and not result:
        raise CandidateArtifactError(f"{field} must not be empty")
    if len(set(result)) != len(result):
        raise CandidateArtifactError(f"{field} must not contain duplicates")
    return result


def _confidence(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
        raise CandidateArtifactError("confidence must be a number between 0 and 1")
    return float(value)


def _artifact_from_event(event: EvoEvent[CandidateArtifactCreatedPayload]) -> CandidateArtifactRecord:
    artifact_id = event.refs.artifact_id
    if artifact_id is None:
        raise CandidateArtifactError("CandidateArtifactCreated event requires artifact_id")
    return CandidateArtifactRecord(event.event_id, artifact_id, event.refs.run_id, event.occurred_at, event.payload)
