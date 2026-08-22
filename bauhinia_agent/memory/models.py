"""P3 memory domain contracts and conservative write rules.

Working context remains owned by :mod:`bauhinia_agent.context`.  This module
models only the five persistable memory layers, before any event-store,
retrieval, or application wiring is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal, Mapping, cast

from bauhinia_agent.evolution.identifiers import EvoIdentifierError, require_evo_id

MemoryLayer = Literal["task", "episodic", "semantic", "procedural", "meta"]
MemoryOrigin = Literal["verified_evidence", "user_confirmation", "inference", "temporary_dialogue"]
MemoryStatus = Literal["active", "proposed", "superseded", "invalidated"]
MemorySensitivity = Literal["public", "internal", "restricted"]
MemoryChangeKind = Literal["supersede", "invalidate", "propose_merge", "confirm"]


class MemoryModelError(ValueError):
    """A persistable memory record violates its P3 domain policy."""


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MemoryModelError(f"{field} must be a non-blank string")
    return value


def _identifier(value: object, *, field: str, kind: str | None = None) -> str:
    try:
        return require_evo_id(value, field=field, kind=cast(object, kind))  # type: ignore[arg-type]
    except EvoIdentifierError as error:
        raise MemoryModelError(str(error)) from error


def _identifier_tuple(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise MemoryModelError(f"{field} must be a list of identifiers")
    result = tuple(_identifier(item, field=f"{field}[]") for item in value)
    if not result:
        raise MemoryModelError(f"{field} must not be empty")
    if len(result) != len(set(result)):
        raise MemoryModelError(f"{field} must not contain duplicates")
    return result


def _confidence(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not 0 <= float(value) <= 1:
        raise MemoryModelError("confidence must be a number between 0 and 1")
    return float(value)


def _utc(value: object, *, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise MemoryModelError(f"{field} must be a timezone-aware datetime")
    return value.astimezone(UTC).replace(microsecond=0)


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_utc(value: object, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise MemoryModelError(f"{field} must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise MemoryModelError(f"{field} must be an ISO-8601 UTC timestamp") from error
    return _utc(parsed, field=field)


@dataclass(frozen=True, slots=True)
class MemoryLayerRule:
    """Retention and access contract for one persistable memory layer."""

    max_lifetime: timedelta
    requires_session_scope: bool
    readable_by: frozenset[str]
    writable_origins: frozenset[MemoryOrigin]


MEMORY_LAYER_RULES: Mapping[MemoryLayer, MemoryLayerRule] = {
    "task": MemoryLayerRule(
        max_lifetime=timedelta(days=30),
        requires_session_scope=True,
        readable_by=frozenset({"session"}),
        writable_origins=frozenset({"verified_evidence", "user_confirmation", "inference"}),
    ),
    "episodic": MemoryLayerRule(
        max_lifetime=timedelta(days=180),
        requires_session_scope=False,
        readable_by=frozenset({"project"}),
        writable_origins=frozenset({"verified_evidence", "user_confirmation"}),
    ),
    "semantic": MemoryLayerRule(
        max_lifetime=timedelta(days=365),
        requires_session_scope=False,
        readable_by=frozenset({"project"}),
        writable_origins=frozenset({"verified_evidence", "user_confirmation"}),
    ),
    "procedural": MemoryLayerRule(
        max_lifetime=timedelta(days=180),
        requires_session_scope=False,
        readable_by=frozenset({"project"}),
        writable_origins=frozenset({"verified_evidence", "user_confirmation"}),
    ),
    "meta": MemoryLayerRule(
        max_lifetime=timedelta(days=90),
        requires_session_scope=False,
        readable_by=frozenset({"project"}),
        writable_origins=frozenset({"verified_evidence"}),
    ),
}


@dataclass(frozen=True, slots=True)
class MemoryScope:
    """Project isolation is mandatory; session scope narrows Task memory."""

    project_id: str
    session_id: str | None = None
    user_id: str | None = None

    def __post_init__(self) -> None:
        _text(self.project_id, field="project_id")
        if self.session_id is not None:
            _identifier(self.session_id, field="session_id", kind="session")
        if self.user_id is not None:
            _text(self.user_id, field="user_id")

    def allows(self, *, project_id: str, session_id: str | None = None, user_id: str | None = None) -> bool:
        """Return whether a read stays inside this record's declared scope."""

        if project_id != self.project_id:
            return False
        if self.session_id is not None and self.session_id != session_id:
            return False
        return self.user_id is None or self.user_id == user_id

    def to_dict(self) -> dict[str, str | None]:
        return {"project_id": self.project_id, "session_id": self.session_id, "user_id": self.user_id}

    @classmethod
    def from_dict(cls, raw: object) -> "MemoryScope":
        if not isinstance(raw, Mapping):
            raise MemoryModelError("scope must be an object")
        unknown = set(raw).difference({"project_id", "session_id", "user_id"})
        if unknown:
            raise MemoryModelError(f"scope has unknown field: {sorted(unknown)[0]}")
        session_id = raw.get("session_id")
        if session_id is not None and not isinstance(session_id, str):
            raise MemoryModelError("scope.session_id must be a string or null")
        user_id = raw.get("user_id")
        if user_id is not None and not isinstance(user_id, str):
            raise MemoryModelError("scope.user_id must be a string or null")
        return cls(project_id=_text(raw.get("project_id"), field="scope.project_id"), session_id=session_id, user_id=user_id)


@dataclass(frozen=True, slots=True)
class MemoryProvenance:
    """The origin is explicit so inference cannot masquerade as verified fact."""

    origin: MemoryOrigin
    source_run_ids: tuple[str, ...] = ()
    source_event_ids: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.origin not in {"verified_evidence", "user_confirmation", "inference", "temporary_dialogue"}:
            raise MemoryModelError(f"unknown provenance origin: {self.origin!r}")
        if self.source_event_ids:
            _identifier_tuple(self.source_event_ids, field="source_event_ids")
        if self.source_run_ids:
            _identifier_tuple(self.source_run_ids, field="source_run_ids")
        if self.evidence_refs:
            _identifier_tuple(self.evidence_refs, field="evidence_refs")
        if not self.source_run_ids:
            raise MemoryModelError("memory provenance requires source_run_ids")
        if not self.source_event_ids and not self.evidence_refs:
            raise MemoryModelError("memory provenance requires a source event or evidence reference")
        if self.origin in {"verified_evidence", "inference"} and not self.evidence_refs:
            raise MemoryModelError(f"{self.origin} requires evidence_refs")
        if self.origin == "user_confirmation" and not self.source_event_ids:
            raise MemoryModelError("user_confirmation requires source_event_ids")
        if self.origin == "temporary_dialogue":
            raise MemoryModelError("temporary dialogue cannot be written as persistable memory")

    def to_dict(self) -> dict[str, object]:
        return {
            "origin": self.origin,
            "source_run_ids": list(self.source_run_ids),
            "source_event_ids": list(self.source_event_ids),
            "evidence_refs": list(self.evidence_refs),
        }

    @classmethod
    def from_dict(cls, raw: object) -> "MemoryProvenance":
        if not isinstance(raw, Mapping):
            raise MemoryModelError("provenance must be an object")
        unknown = set(raw).difference({"origin", "source_run_ids", "source_event_ids", "evidence_refs"})
        if unknown:
            raise MemoryModelError(f"provenance has unknown field: {sorted(unknown)[0]}")
        origin = raw.get("origin")
        if not isinstance(origin, str):
            raise MemoryModelError("provenance.origin must be a string")
        event_ids = raw.get("source_event_ids", [])
        run_ids = raw.get("source_run_ids", [])
        evidence_refs = raw.get("evidence_refs", [])
        return cls(
            origin=cast(MemoryOrigin, origin),
            source_run_ids=_identifier_tuple(run_ids, field="source_run_ids"),
            source_event_ids=() if event_ids == [] else _identifier_tuple(event_ids, field="source_event_ids"),
            evidence_refs=() if evidence_refs == [] else _identifier_tuple(evidence_refs, field="evidence_refs"),
        )


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """A validated, project-isolated candidate for append-only memory events."""

    memory_id: str
    layer: MemoryLayer
    content: str
    scope: MemoryScope
    provenance: MemoryProvenance
    confidence: float
    created_at: datetime
    expires_at: datetime
    conflict_group: str | None = None
    status: MemoryStatus = "active"
    sensitivity: MemorySensitivity = "internal"

    def __post_init__(self) -> None:
        _identifier(self.memory_id, field="memory_id", kind="memory")
        _text(self.content, field="content")
        if self.layer not in MEMORY_LAYER_RULES:
            raise MemoryModelError(f"unknown persistable memory layer: {self.layer!r}")
        if self.status not in {"active", "proposed", "superseded", "invalidated"}:
            raise MemoryModelError(f"unknown memory status: {self.status!r}")
        if self.sensitivity not in {"public", "internal", "restricted"}:
            raise MemoryModelError("sensitivity must be public, internal, or restricted; secrets must not be persisted")
        if self.conflict_group is not None:
            _identifier(self.conflict_group, field="conflict_group")

        object.__setattr__(self, "confidence", _confidence(self.confidence))
        created_at = _utc(self.created_at, field="created_at")
        expires_at = _utc(self.expires_at, field="expires_at")
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "expires_at", expires_at)
        if expires_at <= created_at:
            raise MemoryModelError("expires_at must be after created_at")

        rule = MEMORY_LAYER_RULES[self.layer]
        if expires_at - created_at > rule.max_lifetime:
            raise MemoryModelError(f"{self.layer} memory exceeds its maximum lifetime")
        if rule.requires_session_scope != (self.scope.session_id is not None):
            requirement = "requires" if rule.requires_session_scope else "must not declare"
            raise MemoryModelError(f"{self.layer} memory {requirement} session scope")
        if self.provenance.origin not in rule.writable_origins:
            raise MemoryModelError(f"{self.provenance.origin} cannot write {self.layer} memory")
        if self.provenance.origin == "inference" and self.status != "proposed":
            raise MemoryModelError("inference memory must remain proposed until independently verified")

    @property
    def rule(self) -> MemoryLayerRule:
        return MEMORY_LAYER_RULES[self.layer]

    def is_readable_by(self, *, project_id: str, session_id: str | None = None, user_id: str | None = None) -> bool:
        """Enforce default project isolation before a later retrieval policy ranks it."""

        return self.scope.allows(project_id=project_id, session_id=session_id, user_id=user_id)

    def freshness_weight(self, *, at: datetime) -> float:
        """Return zero for expired memory; P4 may later combine this with relevance."""

        return 0.0 if _utc(at, field="at") >= self.expires_at else 1.0

    def to_dict(self) -> dict[str, object]:
        return {
            "memory_id": self.memory_id,
            "layer": self.layer,
            "content": self.content,
            "scope": self.scope.to_dict(),
            "provenance": self.provenance.to_dict(),
            "confidence": self.confidence,
            "created_at": _utc_text(self.created_at),
            "expires_at": _utc_text(self.expires_at),
            "conflict_group": self.conflict_group,
            "status": self.status,
            "sensitivity": self.sensitivity,
        }

    @classmethod
    def from_dict(cls, raw: object) -> "MemoryRecord":
        if not isinstance(raw, Mapping):
            raise MemoryModelError("memory record must be an object")
        known = {
            "memory_id",
            "layer",
            "content",
            "scope",
            "provenance",
            "confidence",
            "created_at",
            "expires_at",
            "conflict_group",
            "status",
            "sensitivity",
        }
        unknown = set(raw).difference(known)
        if unknown:
            raise MemoryModelError(f"memory record has unknown field: {sorted(unknown)[0]}")
        layer = raw.get("layer")
        status = raw.get("status", "active")
        sensitivity = raw.get("sensitivity", "internal")
        if not isinstance(layer, str) or not isinstance(status, str) or not isinstance(sensitivity, str):
            raise MemoryModelError("layer, status, and sensitivity must be strings")
        return cls(
            memory_id=_identifier(raw.get("memory_id"), field="memory_id", kind="memory"),
            layer=cast(MemoryLayer, layer),
            content=_text(raw.get("content"), field="content"),
            scope=MemoryScope.from_dict(raw.get("scope")),
            provenance=MemoryProvenance.from_dict(raw.get("provenance")),
            confidence=_confidence(raw.get("confidence")),
            created_at=_parse_utc(raw.get("created_at"), field="created_at"),
            expires_at=_parse_utc(raw.get("expires_at"), field="expires_at"),
            conflict_group=raw.get("conflict_group"),
            status=cast(MemoryStatus, status),
            sensitivity=cast(MemorySensitivity, sensitivity),
        )


@dataclass(frozen=True, slots=True)
class MemoryLifecycleChange:
    """Append-only review record; it never mutates the source MemoryRecord."""

    change_id: str
    kind: MemoryChangeKind
    memory_ids: tuple[str, ...]
    occurred_at: datetime
    reason: str
    evidence_refs: tuple[str, ...]
    replacement_memory_id: str | None = None
    proposal_memory_id: str | None = None
    confirmed_by_user_id: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.change_id, field="change_id")
        if self.kind not in {"supersede", "invalidate", "propose_merge", "confirm"}:
            raise MemoryModelError(f"unknown memory change kind: {self.kind!r}")
        _identifier_tuple(self.memory_ids, field="memory_ids")
        object.__setattr__(self, "occurred_at", _utc(self.occurred_at, field="occurred_at"))
        _text(self.reason, field="reason")
        _identifier_tuple(self.evidence_refs, field="evidence_refs")
        if self.replacement_memory_id is not None:
            _identifier(self.replacement_memory_id, field="replacement_memory_id", kind="memory")
        if self.proposal_memory_id is not None:
            _identifier(self.proposal_memory_id, field="proposal_memory_id", kind="memory")
        if self.confirmed_by_user_id is not None:
            _text(self.confirmed_by_user_id, field="confirmed_by_user_id")
        if self.kind == "supersede" and (len(self.memory_ids) != 1 or self.replacement_memory_id is None):
            raise MemoryModelError("supersede requires one source memory and replacement_memory_id")
        if self.kind == "invalidate" and len(self.memory_ids) != 1:
            raise MemoryModelError("invalidate requires exactly one memory")
        if self.kind == "propose_merge" and (len(self.memory_ids) < 2 or self.proposal_memory_id is None):
            raise MemoryModelError("propose_merge requires at least two memories and proposal_memory_id")
        if self.kind == "confirm" and (len(self.memory_ids) != 1 or self.confirmed_by_user_id is None):
            raise MemoryModelError("confirm requires one proposal memory and confirmed_by_user_id")

    def to_dict(self) -> dict[str, object]:
        return {
            "change_id": self.change_id,
            "kind": self.kind,
            "memory_ids": list(self.memory_ids),
            "occurred_at": _utc_text(self.occurred_at),
            "reason": self.reason,
            "evidence_refs": list(self.evidence_refs),
            "replacement_memory_id": self.replacement_memory_id,
            "proposal_memory_id": self.proposal_memory_id,
            "confirmed_by_user_id": self.confirmed_by_user_id,
        }

    @classmethod
    def from_dict(cls, raw: object) -> "MemoryLifecycleChange":
        if not isinstance(raw, Mapping):
            raise MemoryModelError("memory lifecycle change must be an object")
        known = {
            "change_id",
            "kind",
            "memory_ids",
            "occurred_at",
            "reason",
            "evidence_refs",
            "replacement_memory_id",
            "proposal_memory_id",
            "confirmed_by_user_id",
        }
        unknown = set(raw).difference(known)
        if unknown:
            raise MemoryModelError(f"memory lifecycle change has unknown field: {sorted(unknown)[0]}")
        kind = raw.get("kind")
        if not isinstance(kind, str):
            raise MemoryModelError("kind must be a string")
        for field_name in (
            "replacement_memory_id",
            "proposal_memory_id",
            "confirmed_by_user_id",
        ):
            value = raw.get(field_name)
            if value is not None and not isinstance(value, str):
                raise MemoryModelError(f"{field_name} must be a string or null")
        return cls(
            change_id=_identifier(raw.get("change_id"), field="change_id"),
            kind=cast(MemoryChangeKind, kind),
            memory_ids=_identifier_tuple(raw.get("memory_ids"), field="memory_ids"),
            occurred_at=_parse_utc(raw.get("occurred_at"), field="occurred_at"),
            reason=_text(raw.get("reason"), field="reason"),
            evidence_refs=_identifier_tuple(
                raw.get("evidence_refs"),
                field="evidence_refs",
            ),
            replacement_memory_id=raw.get("replacement_memory_id"),  # type: ignore[arg-type]
            proposal_memory_id=raw.get("proposal_memory_id"),  # type: ignore[arg-type]
            confirmed_by_user_id=raw.get("confirmed_by_user_id"),  # type: ignore[arg-type]
        )
