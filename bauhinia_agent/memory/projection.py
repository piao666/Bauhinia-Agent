"""Deterministic Memory projection over append-only Evo events.

``MemoryRecord`` instances are immutable source facts.  Lifecycle events only
derive an effective state and audit relationships; malformed or stale facts are
reported and never partially applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol, cast

from bauhinia_agent.evolution.events import (
    EvidenceRecordedPayload,
    EvoEvent,
    MemoryCreatedPayload,
    MemoryLifecycleChangedPayload,
)
from bauhinia_agent.evolution.evidence import (
    EvidenceIntegrityError,
    resolve_evidence_records,
    user_confirmation_identity,
)
from bauhinia_agent.memory.models import MemoryRecord, MemoryScope, MemoryStatus


@dataclass(frozen=True, slots=True)
class MemoryProjectionDiagnostic:
    """One deterministic reason an append-only fact was not safely applied."""

    code: str
    message: str
    event_id: str | None = None
    memory_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MemoryProjectionEntry:
    """Immutable source record plus its append-only derived lifecycle state."""

    record: MemoryRecord
    effective_status: MemoryStatus
    created_event_id: str
    latest_state_event_id: str
    lifecycle_event_ids: tuple[str, ...] = ()
    replacement_memory_id: str | None = None
    merge_source_memory_ids: tuple[str, ...] = ()
    confirmed_by_user_id: str | None = None
    diagnostics: tuple[MemoryProjectionDiagnostic, ...] = ()

    @property
    def retrieval_eligible(self) -> bool:
        """Only unambiguous active state may be offered to retrieval."""

        return self.effective_status == "active" and not self.diagnostics


@dataclass(frozen=True, slots=True)
class MemoryProjection:
    """A stable, creation-ordered view rebuilt exclusively from source facts."""

    project_id: str
    entries: tuple[MemoryProjectionEntry, ...]
    diagnostics: tuple[MemoryProjectionDiagnostic, ...] = ()

    def get(self, memory_id: str) -> MemoryProjectionEntry | None:
        return next((entry for entry in self.entries if entry.record.memory_id == memory_id), None)


@dataclass(slots=True)
class _MutableEntry:
    record: MemoryRecord
    effective_status: MemoryStatus
    created_event_id: str
    latest_state_event_id: str
    lifecycle_event_ids: list[str] = field(default_factory=list)
    replacement_memory_id: str | None = None
    merge_source_memory_ids: tuple[str, ...] = ()
    confirmed_by_user_id: str | None = None
    diagnostics: list[MemoryProjectionDiagnostic] = field(default_factory=list)

    def freeze(self) -> MemoryProjectionEntry:
        return MemoryProjectionEntry(
            record=self.record,
            effective_status=self.effective_status,
            created_event_id=self.created_event_id,
            latest_state_event_id=self.latest_state_event_id,
            lifecycle_event_ids=tuple(self.lifecycle_event_ids),
            replacement_memory_id=self.replacement_memory_id,
            merge_source_memory_ids=self.merge_source_memory_ids,
            confirmed_by_user_id=self.confirmed_by_user_id,
            diagnostics=tuple(self.diagnostics),
        )


class _DiagnosticReporter(Protocol):
    def __call__(
        self,
        code: str,
        message: str,
        *,
        event_id: str | None,
        memory_ids: tuple[str, ...] = (),
    ) -> None: ...


def build_memory_projection(events: Iterable[EvoEvent], project_id: str) -> MemoryProjection:
    """Reduce ``events`` in append order without mutating source records.

    Invalid facts remain discoverable as diagnostics.  Each lifecycle operation
    is validated in full, including its optimistic-concurrency basis, before any
    entry is changed.
    """

    entries: dict[str, _MutableEntry] = {}
    diagnostics: list[MemoryProjectionDiagnostic] = []
    seen_change_ids: set[str] = set()

    def diagnose(
        code: str,
        message: str,
        *,
        event_id: str | None,
        memory_ids: tuple[str, ...] = (),
    ) -> None:
        diagnostic = MemoryProjectionDiagnostic(code, message, event_id, memory_ids)
        diagnostics.append(diagnostic)
        for memory_id in memory_ids:
            entry = entries.get(memory_id)
            if entry is not None and diagnostic not in entry.diagnostics:
                entry.diagnostics.append(diagnostic)

    event_stream = tuple(events)
    for event_index, event in enumerate(event_stream):
        if event.event_type == "MemoryCreated":
            _apply_created(
                event,
                prior_events=event_stream[:event_index],
                project_id=project_id,
                entries=entries,
                diagnose=diagnose,
            )
            continue
        if event.event_type != "MemoryLifecycleChanged":
            continue
        if not isinstance(event.payload, MemoryLifecycleChangedPayload):
            diagnose(
                "invalid_lifecycle_payload",
                "MemoryLifecycleChanged event does not contain its declared payload contract",
                event_id=event.event_id,
            )
            continue

        payload = event.payload
        affected_ids = _affected_memory_ids(payload, entries)
        if payload.lifecycle_schema_version != "v1":
            diagnose(
                "unsupported_lifecycle_schema",
                f"unsupported Memory lifecycle schema: {payload.lifecycle_schema_version!r}",
                event_id=event.event_id,
                memory_ids=affected_ids,
            )
            continue
        if payload.change_id in seen_change_ids:
            diagnose(
                "duplicate_lifecycle_change",
                f"duplicate Memory lifecycle change_id: {payload.change_id}",
                event_id=event.event_id,
                memory_ids=affected_ids,
            )
            continue
        seen_change_ids.add(payload.change_id)
        if payload.project_id != project_id:
            diagnose(
                "cross_project_lifecycle",
                f"Memory lifecycle project {payload.project_id!r} does not match projection project {project_id!r}",
                event_id=event.event_id,
                memory_ids=affected_ids,
            )
            continue
        primary_target_id = _primary_target_id(payload)
        if primary_target_id is None:
            diagnose(
                "invalid_lifecycle_shape",
                f"Memory lifecycle action {payload.action!r} has no primary target",
                event_id=event.event_id,
                memory_ids=affected_ids,
            )
            continue
        if event.refs.memory_id != primary_target_id:
            diagnose(
                "lifecycle_memory_reference_mismatch",
                f"Memory lifecycle refs.memory_id {event.refs.memory_id!r} does not match primary target {primary_target_id!r}",
                event_id=event.event_id,
                memory_ids=affected_ids,
            )
            continue
        try:
            evidence = resolve_evidence_records(
                event_stream[:event_index],
                payload.evidence_refs,
                run_id=event.refs.run_id,
                require_verified=True,
            )
        except ValueError as error:
            diagnose(
                "invalid_lifecycle_evidence",
                f"Memory lifecycle Evidence is invalid: {error}",
                event_id=event.event_id,
                memory_ids=affected_ids,
            )
            continue
        if payload.action == "confirm":
            confirmation_session_id = payload.extensions.get("confirmation_session_id")
            expected_confirmation = (
                payload.confirmed_by_user_id,
                confirmation_session_id,
            )
            if not isinstance(confirmation_session_id, str) or not any(user_confirmation_identity(record) == expected_confirmation for record in evidence):
                diagnose(
                    "missing_user_confirmation_evidence",
                    "confirm requires prior trusted user-input Evidence for the same user and session",
                    event_id=event.event_id,
                    memory_ids=affected_ids,
                )
                continue
        if not affected_ids:
            diagnose(
                "invalid_lifecycle_shape",
                f"Memory lifecycle action {payload.action!r} has no affected memory",
                event_id=event.event_id,
            )
            continue

        dangling_ids = tuple(memory_id for memory_id in affected_ids if memory_id not in entries)
        if dangling_ids:
            diagnose(
                "dangling_memory_reference",
                f"Memory lifecycle references unknown memories: {', '.join(dangling_ids)}",
                event_id=event.event_id,
                memory_ids=affected_ids,
            )
            continue

        expected_basis = tuple(entries[memory_id].latest_state_event_id for memory_id in affected_ids)
        if tuple(payload.basis_event_ids) != expected_basis:
            diagnose(
                "stale_lifecycle_basis",
                f"Memory lifecycle basis {tuple(payload.basis_event_ids)!r} does not match current basis {expected_basis!r}",
                event_id=event.event_id,
                memory_ids=affected_ids,
            )
            continue

        error = _lifecycle_validation_error(payload, entries)
        if error is not None:
            code, message = error
            diagnose(code, message, event_id=event.event_id, memory_ids=affected_ids)
            continue

        _apply_lifecycle(event.event_id, payload, entries)

    return MemoryProjection(
        project_id=project_id,
        entries=tuple(entry.freeze() for entry in entries.values()),
        diagnostics=tuple(diagnostics),
    )


def _apply_created(
    event: EvoEvent,
    *,
    prior_events: tuple[EvoEvent, ...],
    project_id: str,
    entries: dict[str, _MutableEntry],
    diagnose: _DiagnosticReporter,
) -> None:
    if not isinstance(event.payload, MemoryCreatedPayload):
        diagnose(
            "invalid_memory_created_payload",
            "MemoryCreated event does not contain its declared payload contract",
            event_id=event.event_id,
        )
        return
    raw_record = event.payload.extensions.get("memory_record")
    if raw_record is None:
        diagnose(
            "missing_memory_record",
            "MemoryCreated payload is missing extensions.memory_record",
            event_id=event.event_id,
        )
        return
    try:
        record = MemoryRecord.from_dict(raw_record)
    except (TypeError, ValueError) as error:
        diagnose(
            "invalid_memory_record",
            f"MemoryCreated extensions.memory_record is invalid: {error}",
            event_id=event.event_id,
        )
        return

    memory_id = record.memory_id
    if record.scope.project_id != project_id:
        diagnose(
            "cross_project_memory",
            f"Memory record project {record.scope.project_id!r} does not match projection project {project_id!r}",
            event_id=event.event_id,
            memory_ids=(memory_id,),
        )
        return
    if memory_id in entries:
        diagnose(
            "duplicate_memory_create",
            f"Memory was created more than once: {memory_id}",
            event_id=event.event_id,
            memory_ids=(memory_id,),
        )
        return

    provenance_error = _created_provenance_error(
        event,
        record,
        prior_events=prior_events,
    )
    if provenance_error is not None:
        diagnose(
            "invalid_memory_provenance",
            provenance_error,
            event_id=event.event_id,
            memory_ids=(memory_id,),
        )
        return

    entries[memory_id] = _MutableEntry(
        record=record,
        effective_status=record.status,
        created_event_id=event.event_id,
        latest_state_event_id=event.event_id,
    )
    if event.refs.memory_id != memory_id:
        code = "missing_memory_reference" if event.refs.memory_id is None else "memory_reference_mismatch"
        diagnose(
            code,
            f"MemoryCreated refs.memory_id {event.refs.memory_id!r} does not match record {memory_id!r}",
            event_id=event.event_id,
            memory_ids=(memory_id,),
        )
    if record.status not in {"active", "proposed"}:
        diagnose(
            "invalid_initial_status",
            f"MemoryCreated record cannot start in effective state {record.status!r}",
            event_id=event.event_id,
            memory_ids=(memory_id,),
        )


def _created_provenance_error(
    event: EvoEvent[MemoryCreatedPayload],
    record: MemoryRecord,
    *,
    prior_events: tuple[EvoEvent, ...],
) -> str | None:
    provenance = record.provenance
    if event.refs.run_id != provenance.source_run_ids[0]:
        return "MemoryCreated Run must match the primary provenance source Run"
    payload = event.payload
    expected_scope = "session" if record.scope.session_id is not None else "project"
    if (
        payload.memory_type != record.layer
        or payload.content != record.content
        or payload.scope != expected_scope
        or payload.confidence != record.confidence
        or payload.source_event_ids != provenance.source_event_ids
    ):
        return "MemoryCreated payload does not match its immutable Memory record"

    by_event_id = {item.event_id: item for item in prior_events}
    source_events = tuple(by_event_id.get(source_event_id) for source_event_id in provenance.source_event_ids)
    if any(source is None for source in source_events):
        return "Memory provenance references a missing or future source event"
    source_run_ids = {source.refs.run_id for source in source_events if source is not None}
    if not source_run_ids.issubset(set(provenance.source_run_ids)):
        return "Memory provenance source events belong to an undeclared Run"

    if provenance.evidence_refs:
        try:
            evidence = resolve_evidence_records(
                prior_events,
                provenance.evidence_refs,
                require_verified=provenance.origin == "verified_evidence",
                deterministic_only=provenance.origin == "verified_evidence",
                require_exit_code=provenance.origin == "verified_evidence",
            )
        except (EvidenceIntegrityError, ValueError) as error:
            return f"Memory provenance Evidence is invalid: {error}"
        evidence_run_ids = {item.run_id for item in evidence}
        if not evidence_run_ids.issubset(set(provenance.source_run_ids)):
            return "Memory provenance Evidence belongs to an undeclared Run"

    if provenance.origin == "user_confirmation":
        confirmation_events = [
            source
            for source in source_events
            if source is not None
            and source.event_type == "EvidenceRecorded"
            and isinstance(source.payload, EvidenceRecordedPayload)
            and source.payload.evidence_type == "user_confirmation"
            and source.payload.verified
            and source.payload.source == "user_input_boundary"
        ]
        if not confirmation_events:
            return "user-confirmed Memory requires a trusted user-input confirmation fact"
    return None


def _affected_memory_ids(
    payload: MemoryLifecycleChangedPayload,
    entries: dict[str, _MutableEntry],
) -> tuple[str, ...]:
    target: str | None = None
    if payload.action == "supersede":
        target = payload.replacement_memory_id
    elif payload.action == "propose_merge":
        target = payload.proposal_memory_id
    values = (*payload.memory_ids, *((target,) if target is not None else ()))
    if payload.action == "confirm" and len(payload.memory_ids) == 1:
        pending_target = entries.get(payload.memory_ids[0])
        if pending_target is not None:
            values = (*values, *pending_target.merge_source_memory_ids)
    return tuple(dict.fromkeys(values))


def _primary_target_id(payload: MemoryLifecycleChangedPayload) -> str | None:
    if payload.action == "supersede":
        return payload.replacement_memory_id
    if payload.action == "propose_merge":
        return payload.proposal_memory_id
    return payload.memory_ids[0] if payload.memory_ids else None


def _lifecycle_validation_error(
    payload: MemoryLifecycleChangedPayload,
    entries: dict[str, _MutableEntry],
) -> tuple[str, str] | None:
    if payload.action == "supersede":
        if len(payload.memory_ids) != 1 or payload.replacement_memory_id is None:
            return "invalid_lifecycle_shape", "supersede requires one source and a replacement target"
        source = entries[payload.memory_ids[0]]
        target = entries[payload.replacement_memory_id]
        if source.effective_status != "active" or target.effective_status != "proposed":
            return "invalid_lifecycle_state", "supersede requires an active source and proposed replacement"
        return _compatibility_error((source,), target)

    if payload.action == "invalidate":
        if len(payload.memory_ids) != 1:
            return "invalid_lifecycle_shape", "invalidate requires exactly one memory"
        if entries[payload.memory_ids[0]].effective_status not in {"active", "proposed", "superseded"}:
            return "invalid_lifecycle_state", "invalidate requires active, proposed, or superseded Memory state"
        return None

    if payload.action == "propose_merge":
        if len(payload.memory_ids) < 2 or payload.proposal_memory_id is None:
            return "invalid_lifecycle_shape", "propose_merge requires at least two sources and one proposal"
        if len(set(payload.memory_ids)) != len(payload.memory_ids) or payload.proposal_memory_id in payload.memory_ids:
            return "invalid_lifecycle_shape", "propose_merge requires distinct source and proposal memories"
        sources = tuple(entries[memory_id] for memory_id in payload.memory_ids)
        target = entries[payload.proposal_memory_id]
        if any(source.effective_status != "active" for source in sources) or target.effective_status != "proposed":
            return "invalid_lifecycle_state", "propose_merge requires active sources and a proposed target"
        if target.merge_source_memory_ids:
            return "pending_merge_exists", "merge proposal already has a pending source relationship"
        return _compatibility_error(sources, target)

    if payload.action == "confirm":
        if len(payload.memory_ids) != 1 or not payload.confirmed_by_user_id or payload.actor_kind == "system":
            return "invalid_lifecycle_shape", "confirm requires one proposal and an identified user or maintainer"
        target = entries[payload.memory_ids[0]]
        if target.effective_status != "proposed":
            return "invalid_lifecycle_state", "confirm target must be proposed"
        if target.merge_source_memory_ids:
            sources = tuple(entries.get(memory_id) for memory_id in target.merge_source_memory_ids)
            if any(source is None for source in sources):
                return "dangling_merge_source", "pending merge contains an unknown source Memory"
            resolved_sources = cast(tuple[_MutableEntry, ...], sources)
            if any(source.effective_status != "active" for source in resolved_sources):
                return "invalid_lifecycle_state", "pending merge sources must remain active until confirmation"
            if any(source.latest_state_event_id != target.latest_state_event_id for source in resolved_sources):
                return "stale_pending_merge", "pending merge source state changed after the proposal"
            return _compatibility_error(resolved_sources, target)
        return None

    return "invalid_lifecycle_shape", f"unsupported Memory lifecycle action: {payload.action!r}"


def _compatibility_error(
    sources: tuple[_MutableEntry, ...],
    target: _MutableEntry,
) -> tuple[str, str] | None:
    if any(source.record.layer != target.record.layer for source in sources):
        return "incompatible_memory_layer", "lifecycle target must use the same Memory layer as every source"
    if any(not _scope_within(target.record.scope, source.record.scope) for source in sources):
        return "broader_memory_scope", "lifecycle target cannot broaden a source Memory scope"
    sensitivity_order = {"public": 0, "internal": 1, "restricted": 2}
    if any(sensitivity_order[target.record.sensitivity] < sensitivity_order[source.record.sensitivity] for source in sources):
        return "reduced_memory_sensitivity", "lifecycle target cannot reduce a source Memory sensitivity"
    return None


def _scope_within(target: MemoryScope, source: MemoryScope) -> bool:
    return target.project_id == source.project_id and (source.session_id is None or target.session_id == source.session_id) and (source.user_id is None or target.user_id == source.user_id)


def _apply_lifecycle(
    event_id: str,
    payload: MemoryLifecycleChangedPayload,
    entries: dict[str, _MutableEntry],
) -> None:
    affected_ids = _affected_memory_ids(payload, entries)

    if payload.action == "supersede":
        source = entries[payload.memory_ids[0]]
        target = entries[cast(str, payload.replacement_memory_id)]
        source.effective_status = "superseded"
        source.replacement_memory_id = target.record.memory_id
        target.effective_status = "active"
    elif payload.action == "invalidate":
        entries[payload.memory_ids[0]].effective_status = "invalidated"
    elif payload.action == "propose_merge":
        target = entries[cast(str, payload.proposal_memory_id)]
        target.merge_source_memory_ids = tuple(payload.memory_ids)
    else:
        target = entries[payload.memory_ids[0]]
        source_ids = target.merge_source_memory_ids
        for memory_id in source_ids:
            source = entries[memory_id]
            source.effective_status = "superseded"
            source.replacement_memory_id = target.record.memory_id
        target.effective_status = "active"
        target.confirmed_by_user_id = payload.confirmed_by_user_id

    for memory_id in affected_ids:
        entry = entries[memory_id]
        entry.lifecycle_event_ids.append(event_id)
        entry.latest_state_event_id = event_id
