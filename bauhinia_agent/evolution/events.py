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
from hashlib import sha256
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
        "MemoryLifecycleChanged",
        "ContextPackRecorded",
        "MemoryUsed",
        "ExperienceCandidateCreated",
        "CandidateArtifactCreated",
        "CandidateShadowTrialRecorded",
        "CandidateArtifactControlChanged",
        "CandidateMergeProposed",
        "CandidateConflictDetected",
        "CandidateReviewRecorded",
        "EvaluationCorpusRegistered",
        "EvaluationTrialRecorded",
        "EvaluationComparisonCompleted",
        "EvaluationCompleted",
        "PromotionChanged",
        "SelfModelObservationRecorded",
        "SelfModelUpdated",
        "CollaborationTaskDelegated",
        "CollaborationTaskResultRecorded",
        "CollaborationConflictDetected",
        "CollaborationRunAggregated",
    }
)
_COLLABORATION_STATUSES = frozenset({"success", "failure", "cancelled", "timeout", "permission_denied"})
_COLLABORATION_CONFLICT_KINDS = frozenset({"resource", "conclusion"})
_COLLABORATION_ROLES = frozenset({"planner", "researcher", "executor", "verifier", "critic", "curator"})
_COLLABORATION_CLAIM_FORMATS = frozenset({"v2_full", "v1_fingerprint_only"})


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


def _optional_non_negative_number(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) < 0:
        raise EvoEventError(f"{field} must be a non-negative number or null")
    return float(value)


def _require_non_negative_number(value: object, *, field: str) -> float:
    result = _optional_non_negative_number(value, field=field)
    if result is None:
        raise EvoEventError(f"{field} must be a non-negative number")
    return result


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


def _non_negative_int_list(value: object, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise EvoEventError(f"{field} must be a list of non-negative integers")
    return tuple(_require_non_negative_int(item, field=f"{field}[{index}]") for index, item in enumerate(value))


def _bool_list(value: object, *, field: str) -> tuple[bool, ...]:
    if not isinstance(value, list):
        raise EvoEventError(f"{field} must be a list of booleans")
    return tuple(_require_bool(item, field=f"{field}[{index}]") for index, item in enumerate(value))


def _validated_identifier_list(
    value: object,
    *,
    field: str,
    allow_empty: bool,
    allow_duplicates: bool = False,
) -> tuple[str, ...]:
    result = _string_list(value, field=field)
    if not allow_empty and not result:
        raise EvoEventError(f"{field} must not be empty")
    for index, item in enumerate(result):
        require_evo_id(item, field=f"{field}[{index}]")
    if not allow_duplicates and len(result) != len(set(result)):
        raise EvoEventError(f"{field} must not contain duplicates")
    return result


def _optional_identifier(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return require_evo_id(value, field=field)


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
class MemoryLifecycleChangedPayload(EvoPayload):
    """One append-only memory lifecycle decision.

    Cross-record state and optimistic-concurrency semantics are intentionally
    left to the memory projection.  This payload validates only the stable
    event contract and the relationships intrinsic to each action.
    """

    lifecycle_schema_version: str
    change_id: str
    project_id: str
    action: str
    memory_ids: tuple[str, ...]
    reason: str
    evidence_refs: tuple[str, ...]
    actor_kind: str
    actor_id: str
    basis_event_ids: tuple[str, ...]
    replacement_memory_id: str | None = None
    proposal_memory_id: str | None = None
    confirmed_by_user_id: str | None = None
    extensions: dict[str, object] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, raw: object) -> "MemoryLifecycleChangedPayload":
        values, extensions = _payload_parts(
            raw,
            known={
                "lifecycle_schema_version",
                "change_id",
                "project_id",
                "action",
                "memory_ids",
                "reason",
                "evidence_refs",
                "actor_kind",
                "actor_id",
                "basis_event_ids",
                "replacement_memory_id",
                "proposal_memory_id",
                "confirmed_by_user_id",
            },
        )
        action = _require_text(values.get("action"), field="action")
        if action not in {"supersede", "invalidate", "propose_merge", "confirm"}:
            raise EvoEventError(f"unsupported memory lifecycle action: {action!r}")
        actor_kind = _require_text(values.get("actor_kind"), field="actor_kind")
        if actor_kind not in {"system", "user", "maintainer"}:
            raise EvoEventError(f"unsupported memory lifecycle actor_kind: {actor_kind!r}")

        memory_ids = _validated_identifier_list(values.get("memory_ids"), field="memory_ids", allow_empty=False)
        evidence_refs = _validated_identifier_list(values.get("evidence_refs"), field="evidence_refs", allow_empty=False)
        basis_event_ids = _validated_identifier_list(
            values.get("basis_event_ids"),
            field="basis_event_ids",
            allow_empty=False,
            allow_duplicates=True,
        )
        replacement_memory_id = _optional_identifier(values.get("replacement_memory_id"), field="replacement_memory_id")
        proposal_memory_id = _optional_identifier(values.get("proposal_memory_id"), field="proposal_memory_id")
        confirmed_by_user_id = _optional_text(values.get("confirmed_by_user_id"), field="confirmed_by_user_id")

        if action == "supersede":
            if len(memory_ids) != 1 or replacement_memory_id is None:
                raise EvoEventError("supersede requires exactly one memory_id and replacement_memory_id")
            if replacement_memory_id in memory_ids:
                raise EvoEventError("replacement_memory_id must differ from the superseded memory")
            if proposal_memory_id is not None or confirmed_by_user_id is not None:
                raise EvoEventError("supersede does not accept proposal_memory_id or confirmed_by_user_id")
        elif action == "invalidate":
            if len(memory_ids) != 1:
                raise EvoEventError("invalidate requires exactly one memory_id")
            if replacement_memory_id is not None or proposal_memory_id is not None or confirmed_by_user_id is not None:
                raise EvoEventError("invalidate does not accept replacement, proposal, or confirmation fields")
        elif action == "propose_merge":
            if len(memory_ids) < 2 or proposal_memory_id is None:
                raise EvoEventError("propose_merge requires at least two memory_ids and proposal_memory_id")
            if proposal_memory_id in memory_ids:
                raise EvoEventError("proposal_memory_id must differ from the source memories")
            if replacement_memory_id is not None or confirmed_by_user_id is not None:
                raise EvoEventError("propose_merge does not accept replacement_memory_id or confirmed_by_user_id")
        else:
            if len(memory_ids) != 1 or confirmed_by_user_id is None:
                raise EvoEventError("confirm requires exactly one memory_id and confirmed_by_user_id")
            if actor_kind == "system":
                raise EvoEventError("confirm requires a user or maintainer actor")
            if replacement_memory_id is not None or proposal_memory_id is not None:
                raise EvoEventError("confirm does not accept replacement_memory_id or proposal_memory_id")

        return cls(
            lifecycle_schema_version=_require_text(values.get("lifecycle_schema_version"), field="lifecycle_schema_version"),
            change_id=require_evo_id(values.get("change_id"), field="change_id"),
            project_id=_require_text(values.get("project_id"), field="project_id"),
            action=action,
            memory_ids=memory_ids,
            reason=_require_text(values.get("reason"), field="reason"),
            evidence_refs=evidence_refs,
            actor_kind=actor_kind,
            actor_id=_require_text(values.get("actor_id"), field="actor_id"),
            basis_event_ids=basis_event_ids,
            replacement_memory_id=replacement_memory_id,
            proposal_memory_id=proposal_memory_id,
            confirmed_by_user_id=confirmed_by_user_id,
            extensions=extensions,
        )


@dataclass(frozen=True, slots=True)
class ContextPackRecordedPayload(EvoPayload):
    """Deterministic record of one budgeted long-term-memory context pack."""

    context_pack_schema_version: str
    context_pack_id: str
    query_signature_hash: str
    token_budget: int
    used_tokens: int
    estimator_id: str
    selected_memory_ids: tuple[str, ...]
    selected_ranks: tuple[int, ...]
    selected_original_token_costs: tuple[int, ...]
    selected_packed_token_costs: tuple[int, ...]
    selected_truncated: tuple[bool, ...]
    selected_start_offsets: tuple[int, ...]
    selected_end_offsets: tuple[int, ...]
    omitted_memory_ids: tuple[str, ...]
    omitted_reasons: tuple[str, ...]
    extensions: dict[str, object] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, raw: object) -> "ContextPackRecordedPayload":
        values, extensions = _payload_parts(
            raw,
            known={
                "context_pack_schema_version",
                "context_pack_id",
                "query_signature_hash",
                "token_budget",
                "used_tokens",
                "estimator_id",
                "selected_memory_ids",
                "selected_ranks",
                "selected_original_token_costs",
                "selected_packed_token_costs",
                "selected_truncated",
                "selected_start_offsets",
                "selected_end_offsets",
                "omitted_memory_ids",
                "omitted_reasons",
            },
        )
        selected_memory_ids_raw = values.get("selected_memory_ids")
        omitted_memory_ids_raw = values.get("omitted_memory_ids")
        omitted_reasons_raw = values.get("omitted_reasons")
        if not isinstance(selected_memory_ids_raw, list):
            raise EvoEventError("selected_memory_ids must be a list of strings")
        if not isinstance(omitted_memory_ids_raw, list):
            raise EvoEventError("omitted_memory_ids must be a list of strings")
        if not isinstance(omitted_reasons_raw, list):
            raise EvoEventError("omitted_reasons must be a list of strings")

        selected_memory_ids = _validated_identifier_list(
            selected_memory_ids_raw,
            field="selected_memory_ids",
            allow_empty=True,
        )
        selected_ranks = _non_negative_int_list(values.get("selected_ranks"), field="selected_ranks")
        selected_original_token_costs = _non_negative_int_list(
            values.get("selected_original_token_costs"),
            field="selected_original_token_costs",
        )
        selected_packed_token_costs = _non_negative_int_list(
            values.get("selected_packed_token_costs"),
            field="selected_packed_token_costs",
        )
        selected_truncated = _bool_list(values.get("selected_truncated"), field="selected_truncated")
        selected_start_offsets = _non_negative_int_list(
            values.get("selected_start_offsets"),
            field="selected_start_offsets",
        )
        selected_end_offsets = _non_negative_int_list(
            values.get("selected_end_offsets"),
            field="selected_end_offsets",
        )
        selected_lengths = {
            len(selected_memory_ids),
            len(selected_ranks),
            len(selected_original_token_costs),
            len(selected_packed_token_costs),
            len(selected_truncated),
            len(selected_start_offsets),
            len(selected_end_offsets),
        }
        if len(selected_lengths) != 1:
            raise EvoEventError("selected context-pack arrays must have the same length")
        if any(rank <= 0 for rank in selected_ranks):
            raise EvoEventError("selected_ranks must contain positive integers")
        if tuple(sorted(set(selected_ranks))) != selected_ranks:
            raise EvoEventError("selected_ranks must be unique and strictly increasing")
        if any(
            end <= start
            for start, end in zip(
                selected_start_offsets,
                selected_end_offsets,
                strict=True,
            )
        ):
            raise EvoEventError("selected context-pack offsets must describe non-empty forward ranges")
        if any(
            not truncated and start != 0
            for truncated, start in zip(
                selected_truncated,
                selected_start_offsets,
                strict=True,
            )
        ):
            raise EvoEventError("an untruncated context-pack item must start at offset zero")
        if any(
            packed > original
            for packed, original in zip(
                selected_packed_token_costs,
                selected_original_token_costs,
                strict=True,
            )
        ):
            raise EvoEventError("selected packed token cost must not exceed original token cost")

        omitted_memory_ids = _validated_identifier_list(
            omitted_memory_ids_raw,
            field="omitted_memory_ids",
            allow_empty=True,
        )
        omitted_reasons = _string_list(omitted_reasons_raw, field="omitted_reasons")
        if len(omitted_memory_ids) != len(omitted_reasons):
            raise EvoEventError("omitted_memory_ids and omitted_reasons must have the same length")
        overlap = set(selected_memory_ids).intersection(omitted_memory_ids)
        if overlap:
            raise EvoEventError("a Memory cannot be both selected and omitted in one Context Pack")

        token_budget = _require_non_negative_int(values.get("token_budget"), field="token_budget")
        used_tokens = _require_non_negative_int(values.get("used_tokens"), field="used_tokens")
        if used_tokens > token_budget:
            raise EvoEventError("used_tokens must not exceed token_budget")

        query_signature_hash = _require_text(
            values.get("query_signature_hash"),
            field="query_signature_hash",
        )
        if len(query_signature_hash) != 64 or any(character not in "0123456789abcdef" for character in query_signature_hash):
            raise EvoEventError("query_signature_hash must be a lowercase SHA-256 digest")
        if not selected_memory_ids and used_tokens != 0:
            raise EvoEventError("an empty Context Pack must use zero tokens")

        return cls(
            context_pack_schema_version=_require_text(
                values.get("context_pack_schema_version"),
                field="context_pack_schema_version",
            ),
            context_pack_id=require_evo_id(
                values.get("context_pack_id"),
                field="context_pack_id",
                kind="context_pack",
            ),
            query_signature_hash=query_signature_hash,
            token_budget=token_budget,
            used_tokens=used_tokens,
            estimator_id=_require_text(values.get("estimator_id"), field="estimator_id"),
            selected_memory_ids=selected_memory_ids,
            selected_ranks=selected_ranks,
            selected_original_token_costs=selected_original_token_costs,
            selected_packed_token_costs=selected_packed_token_costs,
            selected_truncated=selected_truncated,
            selected_start_offsets=selected_start_offsets,
            selected_end_offsets=selected_end_offsets,
            omitted_memory_ids=omitted_memory_ids,
            omitted_reasons=omitted_reasons,
            extensions=extensions,
        )


@dataclass(frozen=True, slots=True)
class MemoryUsedPayload(EvoPayload):
    reason: str
    retrieval_rank: int | None = None
    helpfulness: str | None = None
    extensions: dict[str, object] = field(default_factory=dict, repr=False)
    context_pack_id: str | None = None
    usage_status: str = "legacy_unattributed"
    packed_token_cost: int | None = None
    truncated: bool = False
    outcome_event_id: str | None = None
    verification_evidence_refs: tuple[str, ...] = ()
    feedback_status: str = "legacy_unattributed"

    @classmethod
    def from_dict(cls, raw: object) -> "MemoryUsedPayload":
        values, extensions = _payload_parts(
            raw,
            known={
                "reason",
                "retrieval_rank",
                "helpfulness",
                "context_pack_id",
                "usage_status",
                "packed_token_cost",
                "truncated",
                "outcome_event_id",
                "verification_evidence_refs",
                "feedback_status",
            },
        )
        rank = values.get("retrieval_rank")
        usage_status = _require_text(
            values.get("usage_status", "legacy_unattributed"),
            field="usage_status",
        )
        if usage_status not in {"used", "not_used", "legacy_unattributed"}:
            raise EvoEventError(f"unsupported memory usage_status: {usage_status!r}")
        feedback_status = _require_text(
            values.get("feedback_status", "legacy_unattributed"),
            field="feedback_status",
        )
        if feedback_status not in {"helpful", "harmful", "neutral", "unknown", "legacy_unattributed"}:
            raise EvoEventError(f"unsupported memory feedback_status: {feedback_status!r}")
        context_pack_id = _optional_identifier(values.get("context_pack_id"), field="context_pack_id")
        if usage_status in {"used", "not_used"} and context_pack_id is None:
            raise EvoEventError(f"{usage_status} memory usage requires context_pack_id")
        outcome_event_id = _optional_identifier(values.get("outcome_event_id"), field="outcome_event_id")
        verification_evidence_refs = _validated_identifier_list(
            values.get("verification_evidence_refs"),
            field="verification_evidence_refs",
            allow_empty=True,
        )
        if feedback_status in {"helpful", "harmful", "neutral"} and (outcome_event_id is None or not verification_evidence_refs):
            raise EvoEventError(f"{feedback_status} feedback requires outcome_event_id and verification_evidence_refs")
        packed_token_cost = values.get("packed_token_cost")
        return cls(
            reason=_require_text(values.get("reason"), field="reason"),
            retrieval_rank=None if rank is None else _require_non_negative_int(rank, field="retrieval_rank"),
            helpfulness=_optional_text(values.get("helpfulness"), field="helpfulness"),
            extensions=extensions,
            context_pack_id=context_pack_id,
            usage_status=usage_status,
            packed_token_cost=(None if packed_token_cost is None else _require_non_negative_int(packed_token_cost, field="packed_token_cost")),
            truncated=_require_bool(values.get("truncated", False), field="truncated"),
            outcome_event_id=outcome_event_id,
            verification_evidence_refs=verification_evidence_refs,
            feedback_status=feedback_status,
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
class CandidateArtifactCreatedPayload(EvoPayload):
    """A versioned, non-operative artifact derived from reviewed experience."""

    artifact_schema_version: str
    lineage_id: str
    artifact_version: int
    kind: str
    name: str
    description: str
    instructions: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    dependencies: tuple[str, ...]
    effects: tuple[str, ...]
    triggers: tuple[str, ...]
    scope: str
    applicability: str
    risks: tuple[str, ...]
    source_candidate_ids: tuple[str, ...]
    support_candidate_ids: tuple[str, ...]
    counterexample_candidate_ids: tuple[str, ...]
    source_run_ids: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    counterexamples: tuple[str, ...]
    confidence: float
    content_hash: str
    lifecycle_state: str = "Candidate"
    supersedes_artifact_id: str | None = None
    extensions: dict[str, object] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, raw: object) -> "CandidateArtifactCreatedPayload":
        values, extensions = _payload_parts(
            raw,
            known={
                "artifact_schema_version",
                "lineage_id",
                "artifact_version",
                "kind",
                "name",
                "description",
                "instructions",
                "inputs",
                "outputs",
                "dependencies",
                "effects",
                "triggers",
                "scope",
                "applicability",
                "risks",
                "source_candidate_ids",
                "support_candidate_ids",
                "counterexample_candidate_ids",
                "source_run_ids",
                "evidence_refs",
                "counterexamples",
                "confidence",
                "content_hash",
                "lifecycle_state",
                "supersedes_artifact_id",
            },
        )
        version = _require_non_negative_int(values.get("artifact_version"), field="artifact_version")
        if version < 1:
            raise EvoEventError("artifact_version must be positive")
        source_candidate_ids = _string_list(values.get("source_candidate_ids"), field="source_candidate_ids")
        return cls(
            artifact_schema_version=_require_text(values.get("artifact_schema_version"), field="artifact_schema_version"),
            lineage_id=_require_text(values.get("lineage_id"), field="lineage_id"),
            artifact_version=version,
            kind=_require_text(values.get("kind"), field="kind"),
            name=_require_text(values.get("name"), field="name"),
            description=_require_text(values.get("description"), field="description"),
            instructions=_require_text(values.get("instructions"), field="instructions"),
            inputs=_string_list(values.get("inputs"), field="inputs"),
            outputs=_string_list(values.get("outputs"), field="outputs"),
            dependencies=_string_list(values.get("dependencies"), field="dependencies"),
            effects=_string_list(values.get("effects"), field="effects"),
            triggers=_string_list(values.get("triggers"), field="triggers"),
            scope=_require_text(values.get("scope"), field="scope"),
            applicability=_require_text(values.get("applicability"), field="applicability"),
            risks=_string_list(values.get("risks"), field="risks"),
            source_candidate_ids=source_candidate_ids,
            support_candidate_ids=_string_list(
                values.get("support_candidate_ids"),
                field="support_candidate_ids",
                default=source_candidate_ids,
            ),
            counterexample_candidate_ids=_string_list(values.get("counterexample_candidate_ids"), field="counterexample_candidate_ids"),
            source_run_ids=_string_list(values.get("source_run_ids"), field="source_run_ids"),
            evidence_refs=_string_list(values.get("evidence_refs"), field="evidence_refs"),
            counterexamples=_string_list(values.get("counterexamples"), field="counterexamples"),
            confidence=_require_confidence(values.get("confidence")),
            content_hash=_require_text(values.get("content_hash"), field="content_hash"),
            lifecycle_state=_require_text(values.get("lifecycle_state", "Candidate"), field="lifecycle_state"),
            supersedes_artifact_id=_optional_text(values.get("supersedes_artifact_id"), field="supersedes_artifact_id"),
            extensions=extensions,
        )


@dataclass(frozen=True, slots=True)
class CandidateShadowTrialRecordedPayload(EvoPayload):
    """An offline suggestion or Shadow comparison with no real effects."""

    trial_id: str
    artifact_id: str
    artifact_version: int
    mode: str
    task_input_hash: str
    workspace_baseline_hash: str
    environment_hash: str
    baseline_summary: str
    candidate_summary: str
    evidence_refs: tuple[str, ...]
    passed: bool
    real_effects_applied: bool = False
    failure_reason: str | None = None
    extensions: dict[str, object] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, raw: object) -> "CandidateShadowTrialRecordedPayload":
        values, extensions = _payload_parts(
            raw,
            known={
                "trial_id",
                "artifact_id",
                "artifact_version",
                "mode",
                "task_input_hash",
                "workspace_baseline_hash",
                "environment_hash",
                "baseline_summary",
                "candidate_summary",
                "evidence_refs",
                "passed",
                "real_effects_applied",
                "failure_reason",
            },
        )
        version = _require_non_negative_int(values.get("artifact_version"), field="artifact_version")
        if version < 1:
            raise EvoEventError("artifact_version must be positive")
        mode = _require_text(values.get("mode"), field="mode")
        if mode not in {"suggestion", "shadow"}:
            raise EvoEventError("mode must be suggestion or shadow")
        real_effects_applied = _require_bool(values.get("real_effects_applied", False), field="real_effects_applied")
        if real_effects_applied:
            raise EvoEventError("Shadow trials cannot apply real effects")
        passed = _require_bool(values.get("passed"), field="passed")
        failure_reason = _optional_text(values.get("failure_reason"), field="failure_reason")
        if not passed and failure_reason is None:
            raise EvoEventError("failed Shadow trials require failure_reason")
        return cls(
            trial_id=_require_text(values.get("trial_id"), field="trial_id"),
            artifact_id=_require_text(values.get("artifact_id"), field="artifact_id"),
            artifact_version=version,
            mode=mode,
            task_input_hash=_require_text(values.get("task_input_hash"), field="task_input_hash"),
            workspace_baseline_hash=_require_text(values.get("workspace_baseline_hash"), field="workspace_baseline_hash"),
            environment_hash=_require_text(values.get("environment_hash"), field="environment_hash"),
            baseline_summary=_require_text(values.get("baseline_summary"), field="baseline_summary"),
            candidate_summary=_require_text(values.get("candidate_summary"), field="candidate_summary"),
            evidence_refs=_string_list(values.get("evidence_refs"), field="evidence_refs"),
            passed=passed,
            real_effects_applied=False,
            failure_reason=failure_reason,
            extensions=extensions,
        )


@dataclass(frozen=True, slots=True)
class CandidateArtifactControlChangedPayload(EvoPayload):
    """Append-only control of suggestion/Shadow availability, not promotion."""

    artifact_id: str
    artifact_version: int
    action: str
    reviewer: str
    reason: str
    evidence_refs: tuple[str, ...]
    target_artifact_id: str | None = None
    target_artifact_version: int | None = None
    extensions: dict[str, object] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, raw: object) -> "CandidateArtifactControlChangedPayload":
        values, extensions = _payload_parts(
            raw,
            known={
                "artifact_id",
                "artifact_version",
                "action",
                "reviewer",
                "reason",
                "evidence_refs",
                "target_artifact_id",
                "target_artifact_version",
            },
        )
        version = _require_non_negative_int(values.get("artifact_version"), field="artifact_version")
        if version < 1:
            raise EvoEventError("artifact_version must be positive")
        action = _require_text(values.get("action"), field="action")
        if action not in {"disable_shadow", "resume_shadow", "rollback_shadow"}:
            raise EvoEventError("unsupported Artifact control action")
        target_version = _optional_int(values.get("target_artifact_version"), field="target_artifact_version")
        if target_version is not None and target_version < 1:
            raise EvoEventError("target_artifact_version must be positive")
        target_id = _optional_text(values.get("target_artifact_id"), field="target_artifact_id")
        if action == "rollback_shadow" and (target_id is None or target_version is None):
            raise EvoEventError("rollback_shadow requires a target Artifact and version")
        return cls(
            artifact_id=_require_text(values.get("artifact_id"), field="artifact_id"),
            artifact_version=version,
            action=action,
            reviewer=_require_text(values.get("reviewer"), field="reviewer"),
            reason=_require_text(values.get("reason"), field="reason"),
            evidence_refs=_string_list(values.get("evidence_refs"), field="evidence_refs"),
            target_artifact_id=target_id,
            target_artifact_version=target_version,
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
class EvaluationCorpusRegisteredPayload(EvoPayload):
    """Immutable, licensed Corpus manifest metadata without private answers."""

    corpus_schema_version: str
    corpus_id: str
    corpus_version: str
    license_spdx: str
    provenance: str
    case_ids: tuple[str, ...]
    case_splits: tuple[str, ...]
    task_input_hashes: tuple[str, ...]
    workspace_baseline_hashes: tuple[str, ...]
    environment_hashes: tuple[str, ...]
    private_reference_hashes: tuple[str, ...]
    case_manifest_hashes: tuple[str, ...]
    manifest_hash: str
    extensions: dict[str, object] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, raw: object) -> "EvaluationCorpusRegisteredPayload":
        known = {item.name for item in fields(cls) if item.name != "extensions"}
        values, extensions = _payload_parts(raw, known=known)
        case_ids = _string_list(values.get("case_ids"), field="case_ids")
        case_splits = _string_list(values.get("case_splits"), field="case_splits")
        task_hashes = _string_list(values.get("task_input_hashes"), field="task_input_hashes")
        workspace_hashes = _string_list(values.get("workspace_baseline_hashes"), field="workspace_baseline_hashes")
        environment_hashes = _string_list(values.get("environment_hashes"), field="environment_hashes")
        reference_hashes = _string_list(values.get("private_reference_hashes"), field="private_reference_hashes")
        case_manifest_hashes = _string_list(values.get("case_manifest_hashes"), field="case_manifest_hashes")
        lengths = {
            len(case_ids),
            len(case_splits),
            len(task_hashes),
            len(workspace_hashes),
            len(environment_hashes),
            len(reference_hashes),
            len(case_manifest_hashes),
        }
        if lengths != {len(case_ids)} or not case_ids:
            raise EvoEventError("Corpus case manifest fields must be non-empty parallel lists")
        if any(split not in {"source", "development", "held_out"} for split in case_splits):
            raise EvoEventError("Corpus case_splits contain an unsupported split")
        return cls(
            corpus_schema_version=_require_text(values.get("corpus_schema_version"), field="corpus_schema_version"),
            corpus_id=_require_text(values.get("corpus_id"), field="corpus_id"),
            corpus_version=_require_text(values.get("corpus_version"), field="corpus_version"),
            license_spdx=_require_text(values.get("license_spdx"), field="license_spdx"),
            provenance=_require_text(values.get("provenance"), field="provenance"),
            case_ids=case_ids,
            case_splits=case_splits,
            task_input_hashes=task_hashes,
            workspace_baseline_hashes=workspace_hashes,
            environment_hashes=environment_hashes,
            private_reference_hashes=reference_hashes,
            case_manifest_hashes=case_manifest_hashes,
            manifest_hash=_require_text(values.get("manifest_hash"), field="manifest_hash"),
            extensions=extensions,
        )


@dataclass(frozen=True, slots=True)
class EvaluationTrialRecordedPayload(EvoPayload):
    """One fixed, repeatable evaluator attempt represented as a standard Run."""

    evaluation_schema_version: str
    trial_id: str
    trial_key: str
    attempt: int
    case_id: str
    corpus_id: str
    corpus_version: str
    split: str
    variant_id: str
    variant_kind: str
    artifact_id: str | None
    artifact_version: int | None
    evaluator_version: str
    seed: int
    task_input_hash: str
    workspace_baseline_hash: str
    environment_hash: str
    model_config_hash: str
    variant_hash: str
    task_outcome: str
    evaluation_status: str
    success: bool | None
    verification_quality: float | None
    cost: float | None
    latency_ms: float | None
    risk_events: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    verification_commands: tuple[str, ...]
    verification_skipped: bool
    verification_coverage: float
    claimed_success: bool | None
    evidence_success: bool | None
    output_truncated: bool
    accessed_resource_hashes: tuple[str, ...]
    invalid_reasons: tuple[str, ...] = ()
    extensions: dict[str, object] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, raw: object) -> "EvaluationTrialRecordedPayload":
        known = {item.name for item in fields(cls) if item.name != "extensions"}
        values, extensions = _payload_parts(raw, known=known)
        attempt = _require_non_negative_int(values.get("attempt"), field="attempt")
        if attempt < 1:
            raise EvoEventError("attempt must be positive")
        artifact_version = _optional_int(values.get("artifact_version"), field="artifact_version")
        if artifact_version is not None and artifact_version < 1:
            raise EvoEventError("artifact_version must be positive")
        artifact_id = _optional_text(values.get("artifact_id"), field="artifact_id")
        split = _require_text(values.get("split"), field="split")
        if split not in {"source", "development", "held_out"}:
            raise EvoEventError("split must be source, development, or held_out")
        variant_kind = _require_text(values.get("variant_kind"), field="variant_kind")
        if variant_kind not in {"baseline", "candidate"}:
            raise EvoEventError("variant_kind must be baseline or candidate")
        if variant_kind == "candidate" and (artifact_id is None or artifact_version is None):
            raise EvoEventError("candidate Variants require artifact_id and artifact_version")
        if variant_kind == "baseline" and (artifact_id is not None or artifact_version is not None):
            raise EvoEventError("baseline Variants cannot reference an Artifact")
        task_outcome = _require_text(values.get("task_outcome"), field="task_outcome")
        if task_outcome not in {"task_success", "task_failure", "cancelled", "not_run"}:
            raise EvoEventError("unsupported evaluation task_outcome")
        evaluation_status = _require_text(values.get("evaluation_status"), field="evaluation_status")
        if evaluation_status not in {"completed", "evaluator_failure", "invalid", "cancelled"}:
            raise EvoEventError("unsupported evaluation_status")
        success = values.get("success")
        if success is not None:
            success = _require_bool(success, field="success")
        return cls(
            evaluation_schema_version=_require_text(values.get("evaluation_schema_version"), field="evaluation_schema_version"),
            trial_id=_require_text(values.get("trial_id"), field="trial_id"),
            trial_key=_require_text(values.get("trial_key"), field="trial_key"),
            attempt=attempt,
            case_id=_require_text(values.get("case_id"), field="case_id"),
            corpus_id=_require_text(values.get("corpus_id"), field="corpus_id"),
            corpus_version=_require_text(values.get("corpus_version"), field="corpus_version"),
            split=split,
            variant_id=_require_text(values.get("variant_id"), field="variant_id"),
            variant_kind=variant_kind,
            artifact_id=artifact_id,
            artifact_version=artifact_version,
            evaluator_version=_require_text(values.get("evaluator_version"), field="evaluator_version"),
            seed=_require_non_negative_int(values.get("seed"), field="seed"),
            task_input_hash=_require_text(values.get("task_input_hash"), field="task_input_hash"),
            workspace_baseline_hash=_require_text(values.get("workspace_baseline_hash"), field="workspace_baseline_hash"),
            environment_hash=_require_text(values.get("environment_hash"), field="environment_hash"),
            model_config_hash=_require_text(values.get("model_config_hash"), field="model_config_hash"),
            variant_hash=_require_text(values.get("variant_hash"), field="variant_hash"),
            task_outcome=task_outcome,
            evaluation_status=evaluation_status,
            success=success,
            verification_quality=None if values.get("verification_quality") is None else _require_confidence(values["verification_quality"], field="verification_quality"),
            cost=_optional_non_negative_number(values.get("cost"), field="cost"),
            latency_ms=_optional_non_negative_number(values.get("latency_ms"), field="latency_ms"),
            risk_events=_string_list(values.get("risk_events"), field="risk_events"),
            evidence_refs=_string_list(values.get("evidence_refs"), field="evidence_refs"),
            verification_commands=_string_list(values.get("verification_commands"), field="verification_commands"),
            verification_skipped=_require_bool(values.get("verification_skipped"), field="verification_skipped"),
            verification_coverage=_require_confidence(values.get("verification_coverage"), field="verification_coverage"),
            claimed_success=None if values.get("claimed_success") is None else _require_bool(values["claimed_success"], field="claimed_success"),
            evidence_success=None if values.get("evidence_success") is None else _require_bool(values["evidence_success"], field="evidence_success"),
            output_truncated=_require_bool(values.get("output_truncated"), field="output_truncated"),
            accessed_resource_hashes=_string_list(values.get("accessed_resource_hashes"), field="accessed_resource_hashes"),
            invalid_reasons=_string_list(values.get("invalid_reasons"), field="invalid_reasons"),
            extensions=extensions,
        )


@dataclass(frozen=True, slots=True)
class EvaluationComparisonCompletedPayload(EvoPayload):
    """Separated baseline/candidate metrics used by the Promotion Gate."""

    report_id: str
    artifact_id: str
    artifact_version: int
    corpus_id: str
    corpus_version: str
    evaluator_version: str
    baseline_variant_id: str
    candidate_variant_id: str
    case_ids: tuple[str, ...]
    trial_event_ids: tuple[str, ...]
    baseline_sample_count: int
    candidate_sample_count: int
    invalid_trial_count: int
    minimum_repeats: int
    baseline_success_rate: float
    candidate_success_rate: float
    baseline_verification_quality: float
    candidate_verification_quality: float
    baseline_cost: float
    candidate_cost: float
    baseline_latency_ms: float
    candidate_latency_ms: float
    baseline_risk_event_count: int
    candidate_risk_event_count: int
    uncertainty: float
    eligible: bool
    blocking_reasons: tuple[str, ...]
    integrity_violations: tuple[str, ...]
    extensions: dict[str, object] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, raw: object) -> "EvaluationComparisonCompletedPayload":
        known = {item.name for item in fields(cls) if item.name != "extensions"}
        values, extensions = _payload_parts(raw, known=known)
        artifact_version = _require_non_negative_int(values.get("artifact_version"), field="artifact_version")
        if artifact_version < 1:
            raise EvoEventError("artifact_version must be positive")
        minimum_repeats = _require_non_negative_int(values.get("minimum_repeats"), field="minimum_repeats")
        if minimum_repeats < 1:
            raise EvoEventError("minimum_repeats must be positive")
        return cls(
            report_id=_require_text(values.get("report_id"), field="report_id"),
            artifact_id=_require_text(values.get("artifact_id"), field="artifact_id"),
            artifact_version=artifact_version,
            corpus_id=_require_text(values.get("corpus_id"), field="corpus_id"),
            corpus_version=_require_text(values.get("corpus_version"), field="corpus_version"),
            evaluator_version=_require_text(values.get("evaluator_version"), field="evaluator_version"),
            baseline_variant_id=_require_text(values.get("baseline_variant_id"), field="baseline_variant_id"),
            candidate_variant_id=_require_text(values.get("candidate_variant_id"), field="candidate_variant_id"),
            case_ids=_string_list(values.get("case_ids"), field="case_ids"),
            trial_event_ids=_string_list(values.get("trial_event_ids"), field="trial_event_ids"),
            baseline_sample_count=_require_non_negative_int(values.get("baseline_sample_count"), field="baseline_sample_count"),
            candidate_sample_count=_require_non_negative_int(values.get("candidate_sample_count"), field="candidate_sample_count"),
            invalid_trial_count=_require_non_negative_int(values.get("invalid_trial_count"), field="invalid_trial_count"),
            minimum_repeats=minimum_repeats,
            baseline_success_rate=_require_confidence(values.get("baseline_success_rate"), field="baseline_success_rate"),
            candidate_success_rate=_require_confidence(values.get("candidate_success_rate"), field="candidate_success_rate"),
            baseline_verification_quality=_require_confidence(values.get("baseline_verification_quality"), field="baseline_verification_quality"),
            candidate_verification_quality=_require_confidence(values.get("candidate_verification_quality"), field="candidate_verification_quality"),
            baseline_cost=_require_non_negative_number(values.get("baseline_cost"), field="baseline_cost"),
            candidate_cost=_require_non_negative_number(values.get("candidate_cost"), field="candidate_cost"),
            baseline_latency_ms=_require_non_negative_number(values.get("baseline_latency_ms"), field="baseline_latency_ms"),
            candidate_latency_ms=_require_non_negative_number(values.get("candidate_latency_ms"), field="candidate_latency_ms"),
            baseline_risk_event_count=_require_non_negative_int(values.get("baseline_risk_event_count"), field="baseline_risk_event_count"),
            candidate_risk_event_count=_require_non_negative_int(values.get("candidate_risk_event_count"), field="candidate_risk_event_count"),
            uncertainty=_require_confidence(values.get("uncertainty"), field="uncertainty"),
            eligible=_require_bool(values.get("eligible"), field="eligible"),
            blocking_reasons=_string_list(values.get("blocking_reasons"), field="blocking_reasons"),
            integrity_violations=_string_list(values.get("integrity_violations"), field="integrity_violations"),
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
class SelfModelObservationRecordedPayload(EvoPayload):
    """Evidence-linked task observation used to rebuild Self Model profiles."""

    project_id: str
    model_config_hash: str
    evaluator_version: str
    environment_hash: str
    language: str
    repository_scale: str
    task_type: str
    tool_category: str
    risk_level: str
    verification_level: str
    source_event_id: str
    success: bool
    outcome_category: str
    verification_quality: float
    cost: float | None
    latency_ms: float | None
    risk_event_count: int
    evidence_refs: tuple[str, ...]
    extensions: dict[str, object] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, raw: object) -> "SelfModelObservationRecordedPayload":
        known = {item.name for item in fields(cls) if item.name != "extensions"}
        values, extensions = _payload_parts(raw, known=known)
        repository_scale = _require_text(values.get("repository_scale"), field="repository_scale")
        if repository_scale not in {"small", "medium", "large", "unknown"}:
            raise EvoEventError("repository_scale must be small, medium, large, or unknown")
        risk_level = _require_text(values.get("risk_level"), field="risk_level")
        if risk_level not in {"low", "medium", "high", "unknown"}:
            raise EvoEventError("risk_level must be low, medium, high, or unknown")
        verification_level = _require_text(values.get("verification_level"), field="verification_level")
        if verification_level not in {"none", "partial", "strong"}:
            raise EvoEventError("verification_level must be none, partial, or strong")
        return cls(
            project_id=_require_text(values.get("project_id"), field="project_id"),
            model_config_hash=_require_text(values.get("model_config_hash"), field="model_config_hash"),
            evaluator_version=_require_text(values.get("evaluator_version"), field="evaluator_version"),
            environment_hash=_require_text(values.get("environment_hash"), field="environment_hash"),
            language=_require_text(values.get("language"), field="language"),
            repository_scale=repository_scale,
            task_type=_require_text(values.get("task_type"), field="task_type"),
            tool_category=_require_text(values.get("tool_category"), field="tool_category"),
            risk_level=risk_level,
            verification_level=verification_level,
            source_event_id=_require_text(values.get("source_event_id"), field="source_event_id"),
            success=_require_bool(values.get("success"), field="success"),
            outcome_category=_require_text(values.get("outcome_category"), field="outcome_category"),
            verification_quality=_require_confidence(values.get("verification_quality"), field="verification_quality"),
            cost=_optional_non_negative_number(values.get("cost"), field="cost"),
            latency_ms=_optional_non_negative_number(values.get("latency_ms"), field="latency_ms"),
            risk_event_count=_require_non_negative_int(values.get("risk_event_count"), field="risk_event_count"),
            evidence_refs=_string_list(values.get("evidence_refs"), field="evidence_refs"),
            extensions=extensions,
        )


@dataclass(frozen=True, slots=True)
class CollaborationTaskDelegatedPayload(EvoPayload):
    collaboration_id: str
    assignment_id: str
    runtime_role: str
    contract: dict[str, object]
    extensions: dict[str, object] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, raw: object) -> "CollaborationTaskDelegatedPayload":
        values, extensions = _payload_parts(raw, known={"collaboration_id", "assignment_id", "runtime_role", "contract"})
        contract = values.get("contract")
        if not isinstance(contract, Mapping):
            raise EvoEventError("contract must be an object")
        return cls(
            collaboration_id=_require_text(values.get("collaboration_id"), field="collaboration_id"),
            assignment_id=_require_text(values.get("assignment_id"), field="assignment_id"),
            runtime_role=_require_text(values.get("runtime_role"), field="runtime_role"),
            contract={str(key): _json_value(value, field=f"contract.{key}") for key, value in contract.items()},
            extensions=extensions,
        )


def collaboration_claim_fingerprint(
    *,
    claim_key: str,
    conclusion: str,
    evidence_refs: tuple[str, ...],
    source_role: str,
    independence_key: str,
) -> str:
    """Return the v2 fingerprint covering every persisted Claim field."""

    material = "\0".join(
        (
            _require_text(claim_key, field="claim_key"),
            _require_text(conclusion, field="conclusion"),
            *sorted(evidence_refs),
            _require_text(source_role, field="source_role"),
            _require_text(independence_key, field="independence_key"),
        )
    )
    return sha256(material.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CollaborationClaimPayload:
    """A complete, replayable collaboration Claim embedded in a result fact."""

    claim_key: str
    conclusion: str
    evidence_refs: tuple[str, ...]
    source_role: str
    independence_key: str
    fingerprint: str

    def __post_init__(self) -> None:
        _require_text(self.claim_key, field="claim_key")
        _require_text(self.conclusion, field="conclusion")
        if len(self.evidence_refs) != len(set(self.evidence_refs)):
            raise EvoEventError("evidence_refs must not contain duplicates")
        for index, evidence_ref in enumerate(self.evidence_refs):
            _require_text(evidence_ref, field=f"evidence_refs[{index}]")
        if self.source_role not in _COLLABORATION_ROLES:
            raise EvoEventError(f"unknown collaboration role: {self.source_role!r}")
        _require_text(self.independence_key, field="independence_key")
        expected = collaboration_claim_fingerprint(
            claim_key=self.claim_key,
            conclusion=self.conclusion,
            evidence_refs=self.evidence_refs,
            source_role=self.source_role,
            independence_key=self.independence_key,
        )
        if self.fingerprint != expected:
            raise EvoEventError("fingerprint does not match the complete Claim")

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_key": self.claim_key,
            "conclusion": self.conclusion,
            "evidence_refs": list(self.evidence_refs),
            "source_role": self.source_role,
            "independence_key": self.independence_key,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, raw: object) -> "CollaborationClaimPayload":
        if not isinstance(raw, Mapping):
            raise EvoEventError("claims[] must be an object")
        known = {"claim_key", "conclusion", "evidence_refs", "source_role", "independence_key", "fingerprint"}
        unknown = set(raw).difference(known)
        if unknown:
            raise EvoEventError(f"claims[] has unknown field: {sorted(unknown)[0]}")
        claim_key = _require_text(raw.get("claim_key"), field="claims[].claim_key")
        conclusion = _require_text(raw.get("conclusion"), field="claims[].conclusion")
        evidence_refs = _string_list(raw.get("evidence_refs"), field="claims[].evidence_refs")
        if len(evidence_refs) != len(set(evidence_refs)):
            raise EvoEventError("claims[].evidence_refs must not contain duplicates")
        source_role = _require_text(raw.get("source_role"), field="claims[].source_role")
        if source_role not in _COLLABORATION_ROLES:
            raise EvoEventError(f"unknown collaboration role: {source_role!r}")
        independence_key = _require_text(raw.get("independence_key"), field="claims[].independence_key")
        fingerprint = _require_text(raw.get("fingerprint"), field="claims[].fingerprint")
        expected = collaboration_claim_fingerprint(
            claim_key=claim_key,
            conclusion=conclusion,
            evidence_refs=evidence_refs,
            source_role=source_role,
            independence_key=independence_key,
        )
        if fingerprint != expected:
            raise EvoEventError("claims[].fingerprint does not match the complete Claim")
        return cls(
            claim_key=claim_key,
            conclusion=conclusion,
            evidence_refs=evidence_refs,
            source_role=source_role,
            independence_key=independence_key,
            fingerprint=fingerprint,
        )


@dataclass(frozen=True, slots=True)
class CollaborationTaskResultRecordedPayload(EvoPayload):
    collaboration_id: str
    assignment_id: str
    status: str
    summary: str
    evidence_refs: tuple[str, ...]
    confidence: float
    eligible_for_learning: bool
    result_id: str | None = None
    child_run_id: str | None = None
    child_session_id: str | None = None
    claims: tuple[CollaborationClaimPayload, ...] = ()
    claim_fingerprints: tuple[str, ...] = ()
    claim_format: str = "v2_full"
    confidence_source: str = "legacy_unattributed"
    confidence_source_event_id: str | None = None
    files_changed: tuple[str, ...] = ()
    extensions: dict[str, object] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        _require_text(self.collaboration_id, field="collaboration_id")
        _require_text(self.assignment_id, field="assignment_id")
        if self.status not in _COLLABORATION_STATUSES:
            raise EvoEventError(f"unknown collaboration status: {self.status!r}")
        _require_text(self.summary, field="summary")
        _require_confidence(self.confidence)
        if self.claim_format not in _COLLABORATION_CLAIM_FORMATS:
            raise EvoEventError(f"unknown collaboration Claim format: {self.claim_format!r}")
        if self.claim_format == "v1_fingerprint_only" and self.claims:
            raise EvoEventError("v1_fingerprint_only results cannot contain complete claims")
        if self.claims and self.claim_fingerprints != tuple(claim.fingerprint for claim in self.claims):
            raise EvoEventError("claim_fingerprints do not match complete claims")
        evidence_refs = set(self.evidence_refs)
        if any(not set(claim.evidence_refs).issubset(evidence_refs) for claim in self.claims):
            raise EvoEventError("Claim evidence_refs must be a subset of result evidence_refs")
        _require_text(self.confidence_source, field="confidence_source")
        if self.eligible_for_learning and self.claim_format == "v2_full":
            if not self.claims:
                raise EvoEventError("eligible v2 results require at least one complete Claim")
            if self.confidence_source != "outcome_event" or self.confidence_source_event_id is None:
                raise EvoEventError("eligible v2 results require an Outcome confidence source")

    @property
    def claims_rebuildable(self) -> bool:
        return self.claim_format == "v2_full"

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "collaboration_id": self.collaboration_id,
            "assignment_id": self.assignment_id,
            "status": self.status,
            "summary": self.summary,
            "evidence_refs": list(self.evidence_refs),
            "confidence": self.confidence,
            "eligible_for_learning": self.eligible_for_learning,
            "result_id": self.result_id,
            "child_run_id": self.child_run_id,
            "child_session_id": self.child_session_id,
            "claims": [claim.to_dict() for claim in self.claims],
            "claim_fingerprints": list(self.claim_fingerprints),
            "claim_format": self.claim_format,
            "confidence_source": self.confidence_source,
            "confidence_source_event_id": self.confidence_source_event_id,
            "files_changed": list(self.files_changed),
        }
        overlap = set(result).intersection(self.extensions)
        if overlap:
            raise EvoEventError(f"payload extensions conflict with known fields: {sorted(overlap)}")
        for key, value in self.extensions.items():
            if not isinstance(key, str):
                raise EvoEventError("payload extension keys must be strings")
            result[key] = _json_value(value, field=f"payload.{key}")
        return result

    @classmethod
    def from_dict(cls, raw: object) -> "CollaborationTaskResultRecordedPayload":
        values, extensions = _payload_parts(
            raw,
            known={
                "collaboration_id",
                "assignment_id",
                "status",
                "summary",
                "evidence_refs",
                "confidence",
                "eligible_for_learning",
                "result_id",
                "child_run_id",
                "child_session_id",
                "claims",
                "claim_fingerprints",
                "claim_format",
                "confidence_source",
                "confidence_source_event_id",
                "files_changed",
            },
        )
        status = _require_text(values.get("status"), field="status")
        if status not in _COLLABORATION_STATUSES:
            raise EvoEventError(f"unknown collaboration status: {status!r}")
        raw_claims = values.get("claims")
        if raw_claims is None:
            claims: tuple[CollaborationClaimPayload, ...] = ()
        elif not isinstance(raw_claims, list):
            raise EvoEventError("claims must be a list of Claim objects")
        else:
            claims = tuple(CollaborationClaimPayload.from_dict(item) for item in raw_claims)
        claim_fingerprints = _string_list(values.get("claim_fingerprints"), field="claim_fingerprints")
        claim_format = values.get("claim_format")
        if claim_format is None:
            claim_format = "v2_full" if raw_claims is not None else "v1_fingerprint_only"
        claim_format = _require_text(claim_format, field="claim_format")
        if claim_format not in _COLLABORATION_CLAIM_FORMATS:
            raise EvoEventError(f"unknown collaboration Claim format: {claim_format!r}")
        if claim_format == "v1_fingerprint_only" and claims:
            raise EvoEventError("v1_fingerprint_only results cannot contain complete claims")
        if claims:
            expected_fingerprints = tuple(claim.fingerprint for claim in claims)
            if claim_fingerprints and claim_fingerprints != expected_fingerprints:
                raise EvoEventError("claim_fingerprints do not match complete claims")
            claim_fingerprints = expected_fingerprints
        return cls(
            collaboration_id=_require_text(values.get("collaboration_id"), field="collaboration_id"),
            assignment_id=_require_text(values.get("assignment_id"), field="assignment_id"),
            status=status,
            summary=_require_text(values.get("summary"), field="summary"),
            evidence_refs=_string_list(values.get("evidence_refs"), field="evidence_refs"),
            confidence=_require_confidence(values.get("confidence")),
            eligible_for_learning=_require_bool(values.get("eligible_for_learning"), field="eligible_for_learning"),
            result_id=_optional_text(values.get("result_id"), field="result_id"),
            child_run_id=_optional_text(values.get("child_run_id"), field="child_run_id"),
            child_session_id=_optional_text(values.get("child_session_id"), field="child_session_id"),
            claims=claims,
            claim_fingerprints=claim_fingerprints,
            claim_format=claim_format,
            confidence_source=_require_text(values.get("confidence_source", "legacy_unattributed"), field="confidence_source"),
            confidence_source_event_id=_optional_text(values.get("confidence_source_event_id"), field="confidence_source_event_id"),
            files_changed=_string_list(values.get("files_changed"), field="files_changed"),
            extensions=extensions,
        )


@dataclass(frozen=True, slots=True)
class CollaborationConflictDetectedPayload(EvoPayload):
    collaboration_id: str
    conflict_kind: str
    assignment_ids: tuple[str, ...]
    branches: tuple[str, ...]
    resolution_state: str
    resource: str | None = None
    extensions: dict[str, object] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, raw: object) -> "CollaborationConflictDetectedPayload":
        values, extensions = _payload_parts(
            raw,
            known={"collaboration_id", "conflict_kind", "assignment_ids", "branches", "resolution_state", "resource"},
        )
        conflict_kind = _require_text(values.get("conflict_kind"), field="conflict_kind")
        if conflict_kind not in _COLLABORATION_CONFLICT_KINDS:
            raise EvoEventError(f"unknown collaboration conflict kind: {conflict_kind!r}")
        assignment_ids = _string_list(values.get("assignment_ids"), field="assignment_ids")
        if len(assignment_ids) < 2:
            raise EvoEventError("collaboration conflicts require at least two assignment_ids")
        return cls(
            collaboration_id=_require_text(values.get("collaboration_id"), field="collaboration_id"),
            conflict_kind=conflict_kind,
            assignment_ids=assignment_ids,
            branches=_string_list(values.get("branches"), field="branches"),
            resolution_state=_require_text(values.get("resolution_state"), field="resolution_state"),
            resource=_optional_text(values.get("resource"), field="resource"),
            extensions=extensions,
        )


@dataclass(frozen=True, slots=True)
class CollaborationRunAggregatedPayload(EvoPayload):
    collaboration_id: str
    child_run_ids: tuple[str, ...]
    result_event_ids: tuple[str, ...]
    conflict_event_ids: tuple[str, ...]
    evidence_group_count: int
    independent_support_count: int
    eligible_result_ids: tuple[str, ...]
    extensions: dict[str, object] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, raw: object) -> "CollaborationRunAggregatedPayload":
        values, extensions = _payload_parts(
            raw,
            known={
                "collaboration_id",
                "child_run_ids",
                "result_event_ids",
                "conflict_event_ids",
                "evidence_group_count",
                "independent_support_count",
                "eligible_result_ids",
            },
        )
        return cls(
            collaboration_id=_require_text(values.get("collaboration_id"), field="collaboration_id"),
            child_run_ids=_string_list(values.get("child_run_ids"), field="child_run_ids"),
            result_event_ids=_string_list(values.get("result_event_ids"), field="result_event_ids"),
            conflict_event_ids=_string_list(values.get("conflict_event_ids"), field="conflict_event_ids"),
            evidence_group_count=_require_non_negative_int(values.get("evidence_group_count"), field="evidence_group_count"),
            independent_support_count=_require_non_negative_int(values.get("independent_support_count"), field="independent_support_count"),
            eligible_result_ids=_string_list(values.get("eligible_result_ids"), field="eligible_result_ids"),
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
    artifact_id: str | None = None
    evidence_id: str | None = None
    evaluation_id: str | None = None
    promotion_id: str | None = None
    self_model_id: str | None = None
    parent_event_id: str | None = None
    extensions: dict[str, object] = field(default_factory=dict, repr=False)
    context_pack_id: str | None = None

    def __post_init__(self) -> None:
        require_evo_id(self.run_id, field="run_id", kind="run")
        for name in (
            "session_id",
            "plan_id",
            "node_id",
            "memory_id",
            "context_pack_id",
            "candidate_id",
            "artifact_id",
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
            context_pack_id=(None if values.get("context_pack_id") is None else require_evo_id(values["context_pack_id"], field="context_pack_id", kind="context_pack")),
            candidate_id=None if values.get("candidate_id") is None else require_evo_id(values["candidate_id"], field="candidate_id", kind="candidate"),
            artifact_id=None if values.get("artifact_id") is None else require_evo_id(values["artifact_id"], field="artifact_id", kind="artifact"),
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
    "MemoryLifecycleChanged": MemoryLifecycleChangedPayload,
    "ContextPackRecorded": ContextPackRecordedPayload,
    "MemoryUsed": MemoryUsedPayload,
    "ExperienceCandidateCreated": ExperienceCandidateCreatedPayload,
    "CandidateArtifactCreated": CandidateArtifactCreatedPayload,
    "CandidateShadowTrialRecorded": CandidateShadowTrialRecordedPayload,
    "CandidateArtifactControlChanged": CandidateArtifactControlChangedPayload,
    "CandidateMergeProposed": CandidateMergeProposedPayload,
    "CandidateConflictDetected": CandidateConflictDetectedPayload,
    "CandidateReviewRecorded": CandidateReviewRecordedPayload,
    "EvaluationCorpusRegistered": EvaluationCorpusRegisteredPayload,
    "EvaluationTrialRecorded": EvaluationTrialRecordedPayload,
    "EvaluationComparisonCompleted": EvaluationComparisonCompletedPayload,
    "EvaluationCompleted": EvaluationCompletedPayload,
    "PromotionChanged": PromotionChangedPayload,
    "SelfModelObservationRecorded": SelfModelObservationRecordedPayload,
    "SelfModelUpdated": SelfModelUpdatedPayload,
    "CollaborationTaskDelegated": CollaborationTaskDelegatedPayload,
    "CollaborationTaskResultRecorded": CollaborationTaskResultRecordedPayload,
    "CollaborationConflictDetected": CollaborationConflictDetectedPayload,
    "CollaborationRunAggregated": CollaborationRunAggregatedPayload,
}
