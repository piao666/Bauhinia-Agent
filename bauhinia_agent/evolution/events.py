"""Versioned Evo event and payload contracts.

The event envelope is intentionally independent from the existing session JSONL
protocol. It provides the append-only facts that later Store and projection
implementations can persist without making UI or provider dictionaries the
source of truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from typing import Any, ClassVar, Generic, Mapping, TypeVar

from bauhinia_agent.evolution.identifiers import require_evo_id

EVO_EVENT_SCHEMA_VERSION = "v1"
LEGACY_EVENT_SCHEMA_VERSION = "v0"
EVO_EVENT_TYPES = frozenset(
    {
        "PlanCreated",
        "PlanNodeUpdated",
        "DecisionRecorded",
        "EvidenceRecorded",
        "OutcomeClassified",
        "MemoryCreated",
        "MemoryUsed",
        "ExperienceCandidateCreated",
        "CandidateMergeProposed",
        "CandidateConflictDetected",
        "CandidateReviewRecorded",
        "EvaluationCompleted",
        "PromotionChanged",
        "SelfModelUpdated",
    }
)


class EvoEventError(ValueError):
    """Raised when an Evo event or payload violates the contract."""


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _require_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvoEventError(f"{field} must be a non-blank string")
    return value


def _optional_text(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _require_text(value, field=field)


def _require_bool(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise EvoEventError(f"{field} must be a boolean")
    return value


def _require_non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvoEventError(f"{field} must be a non-negative integer")
    return value


def _optional_int(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvoEventError(f"{field} must be an integer or null")
    return value


def _require_sequence(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise EvoEventError(f"{field} must be a positive integer")
    return value


def _require_confidence(value: object, *, field: str = "confidence") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
        raise EvoEventError(f"{field} must be a number between 0 and 1")
    return float(value)


def _string_list(value: object, *, field: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    if value is None:
        return default
    if not isinstance(value, list):
        raise EvoEventError(f"{field} must be a list of strings")
    result: list[str] = []
    for index, item in enumerate(value):
        result.append(_require_text(item, field=f"{field}[{index}]"))
    return tuple(result)


def _json_value(value: object, *, field: str) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, tuple | list):
        return [_json_value(item, field=f"{field}[]") for item in value]
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise EvoEventError(f"{field} keys must be strings")
            result[key] = _json_value(item, field=f"{field}.{key}")
        return result
    raise EvoEventError(f"{field} must contain JSON-compatible values")


def _payload_parts(raw: object, *, known: set[str]) -> tuple[dict[str, object], dict[str, object]]:
    if not isinstance(raw, Mapping):
        raise EvoEventError("payload must be an object")
    values: dict[str, object] = {}
    extensions: dict[str, object] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise EvoEventError("payload keys must be strings")
        target = values if key in known else extensions
        target[key] = _json_value(value, field=f"payload.{key}")
    return values, extensions


class EvoPayload:
    """Base protocol for explicit event payload models."""

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {}
        for item in fields(self):
            if item.name == "extensions":
                continue
            result[item.name] = _json_value(getattr(self, item.name), field=f"payload.{item.name}")
        extensions = getattr(self, "extensions", {})
        if not isinstance(extensions, Mapping):
            raise EvoEventError("payload.extensions must be an object")
        overlap = set(result).intersection(extensions)
        if overlap:
            raise EvoEventError(f"payload extensions conflict with known fields: {sorted(overlap)}")
        for key, value in extensions.items():
            if not isinstance(key, str):
                raise EvoEventError("payload extension keys must be strings")
            result[key] = _json_value(value, field=f"payload.{key}")
        return result


@dataclass(frozen=True, slots=True)
class PlanCreatedPayload(EvoPayload):
    goal: str
    node_ids: tuple[str, ...] = ()
    extensions: dict[str, object] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, raw: object) -> "PlanCreatedPayload":
        values, extensions = _payload_parts(raw, known={"goal", "node_ids"})
        return cls(goal=_require_text(values.get("goal"), field="goal"), node_ids=_string_list(values.get("node_ids"), field="node_ids"), extensions=extensions)


@dataclass(frozen=True, slots=True)
class PlanNodeUpdatedPayload(EvoPayload):
    status: str
    change_summary: str
    attempt: int = 0
    verification_refs: tuple[str, ...] = ()
    extensions: dict[str, object] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, raw: object) -> "PlanNodeUpdatedPayload":
        values, extensions = _payload_parts(raw, known={"status", "change_summary", "attempt", "verification_refs"})
        return cls(
            status=_require_text(values.get("status"), field="status"),
            change_summary=_require_text(values.get("change_summary"), field="change_summary"),
            attempt=_require_non_negative_int(values.get("attempt", 0), field="attempt"),
            verification_refs=_string_list(values.get("verification_refs"), field="verification_refs"),
            extensions=extensions,
        )


@dataclass(frozen=True, slots=True)
class DecisionRecordedPayload(EvoPayload):
    subgoal: str
    evidence_refs: tuple[str, ...]
    assumptions: tuple[str, ...]
    options_considered: tuple[str, ...]
    selected_action: str
    rationale_summary: str
    confidence: float
    expected_observation: str
    verification_method: str
    outcome: str | None = None
    next_decision: str | None = None
    extensions: dict[str, object] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, raw: object) -> "DecisionRecordedPayload":
        values, extensions = _payload_parts(
            raw,
            known={
                "subgoal",
                "evidence_refs",
                "assumptions",
                "options_considered",
                "selected_action",
                "rationale_summary",
                "confidence",
                "expected_observation",
                "verification_method",
                "outcome",
                "next_decision",
            },
        )
        return cls(
            subgoal=_require_text(values.get("subgoal"), field="subgoal"),
            evidence_refs=_string_list(values.get("evidence_refs"), field="evidence_refs"),
            assumptions=_string_list(values.get("assumptions"), field="assumptions"),
            options_considered=_string_list(values.get("options_considered"), field="options_considered"),
            selected_action=_require_text(values.get("selected_action"), field="selected_action"),
            rationale_summary=_require_text(values.get("rationale_summary"), field="rationale_summary"),
            confidence=_require_confidence(values.get("confidence")),
            expected_observation=_require_text(values.get("expected_observation"), field="expected_observation"),
            verification_method=_require_text(values.get("verification_method"), field="verification_method"),
            outcome=_optional_text(values.get("outcome"), field="outcome"),
            next_decision=_optional_text(values.get("next_decision"), field="next_decision"),
            extensions=extensions,
        )


@dataclass(frozen=True, slots=True)
class EvidenceRecordedPayload(EvoPayload):
    evidence_type: str
    source: str
    summary: str
    locator: str | None = None
    verified: bool = False
    command: str | None = None
    input_summary: str | None = None
    cwd: str | None = None
    exit_code: int | None = None
    redacted: bool = False
    extensions: dict[str, object] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, raw: object) -> "EvidenceRecordedPayload":
        values, extensions = _payload_parts(
            raw,
            known={
                "evidence_type",
                "source",
                "summary",
                "locator",
                "verified",
                "command",
                "input_summary",
                "cwd",
                "exit_code",
                "redacted",
            },
        )
        return cls(
            evidence_type=_require_text(values.get("evidence_type"), field="evidence_type"),
            source=_require_text(values.get("source"), field="source"),
            summary=_require_text(values.get("summary"), field="summary"),
            locator=_optional_text(values.get("locator"), field="locator"),
            verified=_require_bool(values.get("verified", False), field="verified"),
            command=_optional_text(values.get("command"), field="command"),
            input_summary=_optional_text(values.get("input_summary"), field="input_summary"),
            cwd=_optional_text(values.get("cwd"), field="cwd"),
            exit_code=_optional_int(values.get("exit_code"), field="exit_code"),
            redacted=_require_bool(values.get("redacted", False), field="redacted"),
            extensions=extensions,
        )


@dataclass(frozen=True, slots=True)
class OutcomeClassifiedPayload(EvoPayload):
    outcome: str
    category: str
    summary: str
    evidence_refs: tuple[str, ...] = ()
    confidence: float = 0.0
    extensions: dict[str, object] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, raw: object) -> "OutcomeClassifiedPayload":
        values, extensions = _payload_parts(raw, known={"outcome", "category", "summary", "evidence_refs", "confidence"})
        return cls(
            outcome=_require_text(values.get("outcome"), field="outcome"),
            category=_require_text(values.get("category"), field="category"),
            summary=_require_text(values.get("summary"), field="summary"),
            evidence_refs=_string_list(values.get("evidence_refs"), field="evidence_refs"),
            confidence=_require_confidence(values.get("confidence", 0.0)),
            extensions=extensions,
        )


@dataclass(frozen=True, slots=True)
class MemoryCreatedPayload(EvoPayload):
    memory_type: str
    content: str
    scope: str
    confidence: float
    source_event_ids: tuple[str, ...]
    invalidation_rule: str | None = None
    extensions: dict[str, object] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, raw: object) -> "MemoryCreatedPayload":
        values, extensions = _payload_parts(raw, known={"memory_type", "content", "scope", "confidence", "source_event_ids", "invalidation_rule"})
        return cls(
            memory_type=_require_text(values.get("memory_type"), field="memory_type"),
            content=_require_text(values.get("content"), field="content"),
            scope=_require_text(values.get("scope"), field="scope"),
            confidence=_require_confidence(values.get("confidence")),
            source_event_ids=_string_list(values.get("source_event_ids"), field="source_event_ids"),
            invalidation_rule=_optional_text(values.get("invalidation_rule"), field="invalidation_rule"),
            extensions=extensions,
        )


@dataclass(frozen=True, slots=True)
class MemoryUsedPayload(EvoPayload):
    reason: str
    retrieval_rank: int | None = None
    helpfulness: str | None = None
    extensions: dict[str, object] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, raw: object) -> "MemoryUsedPayload":
        values, extensions = _payload_parts(raw, known={"reason", "retrieval_rank", "helpfulness"})
        rank = values.get("retrieval_rank")
        return cls(
            reason=_require_text(values.get("reason"), field="reason"),
            retrieval_rank=None if rank is None else _require_non_negative_int(rank, field="retrieval_rank"),
            helpfulness=_optional_text(values.get("helpfulness"), field="helpfulness"),
            extensions=extensions,
        )


@dataclass(frozen=True, slots=True)
class ExperienceCandidateCreatedPayload(EvoPayload):
    kind: str
    summary: str
    scope: str
    applicability: str
    confidence: float
    source_event_ids: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    counterexamples: tuple[str, ...] = ()
    novelty: float | None = None
    novelty_status: str = "unassessed"
    source_run_ids: tuple[str, ...] = ()
    environment_summary: str | None = None
    lifecycle_state: str = "Candidate"
    extensions: dict[str, object] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, raw: object) -> "ExperienceCandidateCreatedPayload":
        values, extensions = _payload_parts(
            raw,
            known={
                "kind",
                "summary",
                "scope",
                "applicability",
                "confidence",
                "source_event_ids",
                "evidence_refs",
                "counterexamples",
                "novelty",
                "novelty_status",
                "source_run_ids",
                "environment_summary",
                "lifecycle_state",
            },
        )
        return cls(
            kind=_require_text(values.get("kind"), field="kind"),
            summary=_require_text(values.get("summary", "Candidate summary unavailable."), field="summary"),
            scope=_require_text(values.get("scope"), field="scope"),
            applicability=_require_text(values.get("applicability"), field="applicability"),
            confidence=_require_confidence(values.get("confidence")),
            source_event_ids=_string_list(values.get("source_event_ids"), field="source_event_ids"),
            evidence_refs=_string_list(values.get("evidence_refs"), field="evidence_refs"),
            counterexamples=_string_list(values.get("counterexamples"), field="counterexamples"),
            novelty=None if values.get("novelty") is None else _require_confidence(values["novelty"], field="novelty"),
            novelty_status=_require_text(values.get("novelty_status", "unassessed"), field="novelty_status"),
            source_run_ids=_string_list(values.get("source_run_ids"), field="source_run_ids"),
            environment_summary=_optional_text(values.get("environment_summary"), field="environment_summary"),
            lifecycle_state=_require_text(values.get("lifecycle_state", "Candidate"), field="lifecycle_state"),
            extensions=extensions,
        )


@dataclass(frozen=True, slots=True)
class CandidateMergeProposedPayload(EvoPayload):
    """A reviewable deduplication proposal; source candidates remain unchanged."""

    cluster_id: str
    scope: str
    kind: str
    candidate_ids: tuple[str, ...]
    source_run_ids: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    task_features: tuple[str, ...]
    similarity: float
    proposal_summary: str
    extensions: dict[str, object] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, raw: object) -> "CandidateMergeProposedPayload":
        values, extensions = _payload_parts(
            raw,
            known={
                "cluster_id",
                "scope",
                "kind",
                "candidate_ids",
                "source_run_ids",
                "evidence_refs",
                "task_features",
                "similarity",
                "proposal_summary",
            },
        )
        return cls(
            cluster_id=_require_text(values.get("cluster_id"), field="cluster_id"),
            scope=_require_text(values.get("scope"), field="scope"),
            kind=_require_text(values.get("kind"), field="kind"),
            candidate_ids=_string_list(values.get("candidate_ids"), field="candidate_ids"),
            source_run_ids=_string_list(values.get("source_run_ids"), field="source_run_ids"),
            evidence_refs=_string_list(values.get("evidence_refs"), field="evidence_refs"),
            task_features=_string_list(values.get("task_features"), field="task_features"),
            similarity=_require_confidence(values.get("similarity"), field="similarity"),
            proposal_summary=_require_text(values.get("proposal_summary"), field="proposal_summary"),
            extensions=extensions,
        )


@dataclass(frozen=True, slots=True)
class CandidateConflictDetectedPayload(EvoPayload):
    """An explicit conflicting-candidate group that requires later review."""

    conflict_group_id: str
    scope: str
    candidate_ids: tuple[str, ...]
    conclusions: tuple[str, ...]
    source_run_ids: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    task_features: tuple[str, ...]
    similarity: float
    summary: str
    extensions: dict[str, object] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, raw: object) -> "CandidateConflictDetectedPayload":
        values, extensions = _payload_parts(
            raw,
            known={
                "conflict_group_id",
                "scope",
                "candidate_ids",
                "conclusions",
                "source_run_ids",
                "evidence_refs",
                "task_features",
                "similarity",
                "summary",
            },
        )
        return cls(
            conflict_group_id=_require_text(values.get("conflict_group_id"), field="conflict_group_id"),
            scope=_require_text(values.get("scope"), field="scope"),
            candidate_ids=_string_list(values.get("candidate_ids"), field="candidate_ids"),
            conclusions=_string_list(values.get("conclusions"), field="conclusions"),
            source_run_ids=_string_list(values.get("source_run_ids"), field="source_run_ids"),
            evidence_refs=_string_list(values.get("evidence_refs"), field="evidence_refs"),
            task_features=_string_list(values.get("task_features"), field="task_features"),
            similarity=_require_confidence(values.get("similarity"), field="similarity"),
            summary=_require_text(values.get("summary"), field="summary"),
            extensions=extensions,
        )


@dataclass(frozen=True, slots=True)
class CandidateReviewRecordedPayload(EvoPayload):
    """A human review decision or metadata edit for an immutable Candidate."""

    candidate_id: str
    decision: str
    reviewer: str
    reason: str
    scope: str | None = None
    ttl_seconds: int | None = None
    sensitivity: str | None = None
    extensions: dict[str, object] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, raw: object) -> "CandidateReviewRecordedPayload":
        values, extensions = _payload_parts(raw, known={"candidate_id", "decision", "reviewer", "reason", "scope", "ttl_seconds", "sensitivity"})
        decision = _require_text(values.get("decision"), field="decision")
        if decision not in {"accept", "reject", "defer", "edit"}:
            raise EvoEventError("decision must be accept, reject, defer, or edit")
        ttl_seconds = _optional_int(values.get("ttl_seconds"), field="ttl_seconds")
        if ttl_seconds is not None and ttl_seconds < 1:
            raise EvoEventError("ttl_seconds must be positive when supplied")
        sensitivity = _optional_text(values.get("sensitivity"), field="sensitivity")
        if sensitivity is not None and sensitivity not in {"public", "internal", "restricted"}:
            raise EvoEventError("sensitivity must be public, internal, or restricted")
        return cls(
            candidate_id=_require_text(values.get("candidate_id"), field="candidate_id"),
            decision=decision,
            reviewer=_require_text(values.get("reviewer"), field="reviewer"),
            reason=_require_text(values.get("reason"), field="reason"),
            scope=_optional_text(values.get("scope"), field="scope"),
            ttl_seconds=ttl_seconds,
            sensitivity=sensitivity,
            extensions=extensions,
        )


@dataclass(frozen=True, slots=True)
class EvaluationCompletedPayload(EvoPayload):
    dataset: str
    evaluator_version: str
    passed: bool
    sample_count: int
    success_rate: float | None = None
    verification_quality: float | None = None
    cost: float | None = None
    latency_ms: float | None = None
    risk_event_count: int = 0
    uncertainty: float | None = None
    extensions: dict[str, object] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, raw: object) -> "EvaluationCompletedPayload":
        values, extensions = _payload_parts(
            raw, known={"dataset", "evaluator_version", "passed", "sample_count", "success_rate", "verification_quality", "cost", "latency_ms", "risk_event_count", "uncertainty"}
        )

        def optional_rate(name: str) -> float | None:
            value = values.get(name)
            return None if value is None else _require_confidence(value, field=name)

        return cls(
            dataset=_require_text(values.get("dataset"), field="dataset"),
            evaluator_version=_require_text(values.get("evaluator_version"), field="evaluator_version"),
            passed=_require_bool(values.get("passed"), field="passed"),
            sample_count=_require_non_negative_int(values.get("sample_count"), field="sample_count"),
            success_rate=optional_rate("success_rate"),
            verification_quality=optional_rate("verification_quality"),
            cost=None if values.get("cost") is None else float(values["cost"]),
            latency_ms=None if values.get("latency_ms") is None else float(values["latency_ms"]),
            risk_event_count=_require_non_negative_int(values.get("risk_event_count", 0), field="risk_event_count"),
            uncertainty=optional_rate("uncertainty"),
            extensions=extensions,
        )


@dataclass(frozen=True, slots=True)
class PromotionChangedPayload(EvoPayload):
    from_state: str
    to_state: str
    reason: str
    reviewer: str | None = None
    evaluation_event_ids: tuple[str, ...] = ()
    rollback_target: str | None = None
    extensions: dict[str, object] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, raw: object) -> "PromotionChangedPayload":
        values, extensions = _payload_parts(raw, known={"from_state", "to_state", "reason", "reviewer", "evaluation_event_ids", "rollback_target"})
        return cls(
            from_state=_require_text(values.get("from_state"), field="from_state"),
            to_state=_require_text(values.get("to_state"), field="to_state"),
            reason=_require_text(values.get("reason"), field="reason"),
            reviewer=_optional_text(values.get("reviewer"), field="reviewer"),
            evaluation_event_ids=_string_list(values.get("evaluation_event_ids"), field="evaluation_event_ids"),
            rollback_target=_optional_text(values.get("rollback_target"), field="rollback_target"),
            extensions=extensions,
        )


@dataclass(frozen=True, slots=True)
class SelfModelUpdatedPayload(EvoPayload):
    dimension: str
    scope: str
    sample_count: int
    window_start: str
    window_end: str
    confidence: float
    metrics: dict[str, object]
    extensions: dict[str, object] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, raw: object) -> "SelfModelUpdatedPayload":
        values, extensions = _payload_parts(raw, known={"dimension", "scope", "sample_count", "window_start", "window_end", "confidence", "metrics"})
        metrics = values.get("metrics")
        if not isinstance(metrics, dict):
            raise EvoEventError("metrics must be an object")
        return cls(
            dimension=_require_text(values.get("dimension"), field="dimension"),
            scope=_require_text(values.get("scope"), field="scope"),
            sample_count=_require_non_negative_int(values.get("sample_count"), field="sample_count"),
            window_start=_require_text(values.get("window_start"), field="window_start"),
            window_end=_require_text(values.get("window_end"), field="window_end"),
            confidence=_require_confidence(values.get("confidence")),
            metrics=metrics,
            extensions=extensions,
        )


@dataclass(frozen=True, slots=True)
class UnknownEvoPayload(EvoPayload):
    """Payload for a future event type; raw fields remain inspectable."""

    raw: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {key: _json_value(value, field=f"payload.{key}") for key, value in self.raw.items()}

    @classmethod
    def from_dict(cls, raw: object) -> "UnknownEvoPayload":
        if not isinstance(raw, Mapping):
            raise EvoEventError("payload must be an object")
        return cls(raw={str(key): _json_value(value, field=f"payload.{key}") for key, value in raw.items()})


@dataclass(frozen=True, slots=True)
class EvoReferences:
    """Stable relationships shared by every Evo event."""

    run_id: str
    session_id: str | None = None
    plan_id: str | None = None
    node_id: str | None = None
    memory_id: str | None = None
    candidate_id: str | None = None
    evidence_id: str | None = None
    evaluation_id: str | None = None
    promotion_id: str | None = None
    self_model_id: str | None = None
    parent_event_id: str | None = None
    extensions: dict[str, object] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        require_evo_id(self.run_id, field="run_id", kind="run")
        for name in (
            "session_id",
            "plan_id",
            "node_id",
            "memory_id",
            "candidate_id",
            "evidence_id",
            "evaluation_id",
            "promotion_id",
            "self_model_id",
            "parent_event_id",
        ):
            value = getattr(self, name)
            if value is not None:
                require_evo_id(value, field=name)

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {}
        for item in fields(self):
            if item.name == "extensions":
                continue
            value = getattr(self, item.name)
            if value is not None:
                result[item.name] = value
        overlap = set(result).intersection(self.extensions)
        if overlap:
            raise EvoEventError(f"reference extensions conflict with known fields: {sorted(overlap)}")
        for key, value in self.extensions.items():
            if not isinstance(key, str):
                raise EvoEventError("reference extension keys must be strings")
            result[key] = _json_value(value, field=f"refs.{key}")
        return result

    @classmethod
    def from_dict(cls, raw: object, *, legacy_envelope: Mapping[str, object] | None = None) -> "EvoReferences":
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise EvoEventError("refs must be an object")
        known = {item.name for item in fields(cls) if item.name != "extensions"}
        values: dict[str, object] = dict(raw)
        if legacy_envelope is not None:
            for key in known:
                if key not in values and key in legacy_envelope:
                    values[key] = legacy_envelope[key]
        extensions = {str(key): _json_value(value, field=f"refs.{key}") for key, value in raw.items() if key not in known}
        return cls(
            run_id=require_evo_id(values.get("run_id"), field="run_id", kind="run"),
            session_id=None if values.get("session_id") is None else require_evo_id(values["session_id"], field="session_id", kind="session"),
            plan_id=None if values.get("plan_id") is None else require_evo_id(values["plan_id"], field="plan_id", kind="plan"),
            node_id=None if values.get("node_id") is None else require_evo_id(values["node_id"], field="node_id", kind="node"),
            memory_id=None if values.get("memory_id") is None else require_evo_id(values["memory_id"], field="memory_id", kind="memory"),
            candidate_id=None if values.get("candidate_id") is None else require_evo_id(values["candidate_id"], field="candidate_id", kind="candidate"),
            evidence_id=None if values.get("evidence_id") is None else require_evo_id(values["evidence_id"], field="evidence_id", kind="evidence"),
            evaluation_id=None if values.get("evaluation_id") is None else require_evo_id(values["evaluation_id"], field="evaluation_id", kind="evaluation"),
            promotion_id=None if values.get("promotion_id") is None else require_evo_id(values["promotion_id"], field="promotion_id", kind="promotion"),
            self_model_id=None if values.get("self_model_id") is None else require_evo_id(values["self_model_id"], field="self_model_id", kind="self_model"),
            parent_event_id=None if values.get("parent_event_id") is None else require_evo_id(values["parent_event_id"], field="parent_event_id", kind="event"),
            extensions=extensions,
        )


PayloadT = TypeVar("PayloadT", bound=EvoPayload)


@dataclass(frozen=True, slots=True)
class EvoEvent(Generic[PayloadT]):
    """Canonical versioned Evo event envelope."""

    event_id: str
    event_type: str
    refs: EvoReferences
    payload: PayloadT
    schema_version: str = EVO_EVENT_SCHEMA_VERSION
    occurred_at: str = field(default_factory=_utc_now_iso)
    sequence: int | None = None
    extensions: dict[str, object] = field(default_factory=dict, repr=False)
    _payload_types: ClassVar[dict[str, type[EvoPayload]]] = {}

    def __post_init__(self) -> None:
        require_evo_id(self.event_id, field="event_id", kind="event")
        _require_text(self.event_type, field="event_type")
        _require_text(self.schema_version, field="schema_version")
        _validate_utc(self.occurred_at, field="occurred_at")
        if self.sequence is not None:
            _require_sequence(self.sequence, field="sequence")

    @property
    def is_known_type(self) -> bool:
        return self.event_type in self._payload_types

    def validate_persisted(self) -> None:
        """Validate the additional invariant required once a Store persists it."""

        if self.sequence is None:
            raise EvoEventError("persisted Evo events require a positive sequence")

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "schema_version": self.schema_version,
            "occurred_at": self.occurred_at,
            "sequence": self.sequence,
            "refs": self.refs.to_dict(),
            "payload": self.payload.to_dict(),
        }
        overlap = set(result).intersection(self.extensions)
        if overlap:
            raise EvoEventError(f"event extensions conflict with known fields: {sorted(overlap)}")
        for key, value in self.extensions.items():
            if not isinstance(key, str):
                raise EvoEventError("event extension keys must be strings")
            result[key] = _json_value(value, field=f"event.{key}")
        return result

    def to_json(self) -> str:
        """Return deterministic JSON suitable for Golden tests and diagnostics."""

        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "EvoEvent[EvoPayload]":
        if not isinstance(raw, Mapping):
            raise EvoEventError("event must be an object")
        event_id = raw.get("event_id", raw.get("id"))
        event_type = raw.get("event_type", raw.get("type"))
        occurred_at = raw.get("occurred_at", raw.get("created_at"))
        if occurred_at is None:
            raise EvoEventError("event requires occurred_at or legacy created_at")
        schema_version = raw.get("schema_version", LEGACY_EVENT_SCHEMA_VERSION)
        sequence = raw.get("sequence")
        if sequence is not None:
            sequence = _require_sequence(sequence, field="sequence")
        refs_raw = raw.get("refs", raw.get("references"))
        refs = EvoReferences.from_dict(refs_raw, legacy_envelope=raw)
        payload_raw = raw.get("payload", raw.get("data", {}))
        parser = cls._payload_types.get(str(event_type))
        payload = parser.from_dict(payload_raw) if parser is not None else UnknownEvoPayload.from_dict(payload_raw)
        known = {"event_id", "event_type", "id", "type", "schema_version", "occurred_at", "created_at", "sequence", "refs", "references", "payload", "data"}
        extensions = {str(key): _json_value(value, field=f"event.{key}") for key, value in raw.items() if key not in known}
        return cls(
            event_id=require_evo_id(event_id, field="event_id", kind="event"),
            event_type=_require_text(event_type, field="event_type"),
            refs=refs,
            payload=payload,
            schema_version=_require_text(schema_version, field="schema_version"),
            occurred_at=_require_text(occurred_at, field="occurred_at"),
            sequence=sequence,
            extensions=extensions,
        )

    @classmethod
    def from_json(cls, value: str) -> "EvoEvent[EvoPayload]":
        try:
            raw = json.loads(value)
        except json.JSONDecodeError as error:
            raise EvoEventError(f"invalid event JSON: {error.msg}") from error
        return cls.from_dict(raw)


def _validate_utc(value: object, *, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise EvoEventError(f"{field} must be a UTC ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise EvoEventError(f"{field} must be a UTC ISO-8601 string") from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise EvoEventError(f"{field} must include UTC timezone")


EvoEvent._payload_types = {
    "PlanCreated": PlanCreatedPayload,
    "PlanNodeUpdated": PlanNodeUpdatedPayload,
    "DecisionRecorded": DecisionRecordedPayload,
    "EvidenceRecorded": EvidenceRecordedPayload,
    "OutcomeClassified": OutcomeClassifiedPayload,
    "MemoryCreated": MemoryCreatedPayload,
    "MemoryUsed": MemoryUsedPayload,
    "ExperienceCandidateCreated": ExperienceCandidateCreatedPayload,
    "CandidateMergeProposed": CandidateMergeProposedPayload,
    "CandidateConflictDetected": CandidateConflictDetectedPayload,
    "CandidateReviewRecorded": CandidateReviewRecordedPayload,
    "EvaluationCompleted": EvaluationCompletedPayload,
    "PromotionChanged": PromotionChangedPayload,
    "SelfModelUpdated": SelfModelUpdatedPayload,
}
