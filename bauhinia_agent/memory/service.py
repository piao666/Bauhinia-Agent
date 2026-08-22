"""P3 memory projection over the canonical append-only Evo event store."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Callable, Literal

from bauhinia_agent.evolution import (
    EvidenceIntegrityError,
    EvidenceRecord,
    EvoEvent,
    EvoEventStore,
    EvoReferences,
    EvoStoreError,
    MemoryCreatedPayload,
    MemoryLifecycleChangedPayload,
    new_evo_id,
    redact_text,
    require_evo_id,
    resolve_evidence_records,
    user_confirmation_identity,
)
from bauhinia_agent.memory.models import MemoryModelError, MemoryRecord, MemoryScope
from bauhinia_agent.memory.projection import (
    MemoryProjection,
    MemoryProjectionEntry,
    build_memory_projection,
)

MemoryActorKind = Literal["system", "user", "maintainer"]


class MemoryWriteDisabledError(MemoryModelError):
    pass


@dataclass(frozen=True, slots=True)
class MemoryDiagnostic:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class MemoryLifecycleResult:
    persisted: bool
    applied: bool
    event: EvoEvent[MemoryLifecycleChangedPayload] | None = None
    diagnostic: MemoryDiagnostic | None = None


class MemoryService:
    """Writes source events and rebuilds deterministic in-memory projections."""

    def __init__(
        self,
        *,
        store: EvoEventStore,
        project_id: str,
        writes_enabled: bool = True,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._store = store
        self._project_id = project_id
        self._writes_enabled = writes_enabled
        self._clock = clock or (lambda: datetime.now(UTC))

    @property
    def writes_enabled(self) -> bool:
        return self._writes_enabled

    @property
    def project_id(self) -> str:
        return self._project_id

    @property
    def store(self) -> EvoEventStore:
        return self._store

    def set_writes_enabled(self, enabled: bool) -> None:
        self._writes_enabled = bool(enabled)

    def create(self, record: MemoryRecord) -> MemoryRecord:
        if not self._writes_enabled:
            raise MemoryWriteDisabledError("memory writes are disabled")
        if record.scope.project_id != self._project_id:
            raise MemoryModelError("cannot write memory outside this project scope")
        events = self._store.list_events()
        if build_memory_projection(events, project_id=self._project_id).get(record.memory_id) is not None:
            raise MemoryModelError(f"memory already exists: {record.memory_id}")
        event = EvoEvent(
            event_id=_memory_create_event_id(record.memory_id),
            event_type="MemoryCreated",
            refs=EvoReferences(run_id=record.provenance.source_run_ids[0], memory_id=record.memory_id),
            payload=MemoryCreatedPayload(
                memory_type=record.layer,
                content=record.content,
                scope="session" if record.scope.session_id is not None else "project",
                confidence=record.confidence,
                source_event_ids=record.provenance.source_event_ids,
                extensions={"memory_record": record.to_dict()},
            ),
        )
        candidate = build_memory_projection(
            (*events, event),
            project_id=self._project_id,
        )
        invalid = next(
            (item for item in candidate.diagnostics if item.event_id == event.event_id),
            None,
        )
        if invalid is not None:
            raise MemoryModelError(invalid.message)
        try:
            appended = self._store.append(event)
        except EvoStoreError as error:
            if "duplicate event_id" in str(error):
                raise MemoryModelError(f"memory already exists: {record.memory_id}") from error
            raise
        rebuilt = self.projection().get(record.memory_id)
        if rebuilt is None or rebuilt.created_event_id != appended.event.event_id:
            raise MemoryModelError("Memory source fact was persisted but failed closed during projection")
        return record

    def projection(self) -> MemoryProjection:
        """Rebuild effective Memory state from append-only source facts."""

        return build_memory_projection(self._store.list_events(), project_id=self._project_id)

    def get_view(self, memory_id: str) -> MemoryProjectionEntry:
        require_evo_id(memory_id, field="memory_id", kind="memory")
        entry = self.projection().get(memory_id)
        if entry is None:
            raise MemoryModelError(f"unknown memory: {memory_id}")
        return entry

    def list_views(self) -> tuple[MemoryProjectionEntry, ...]:
        return self.projection().entries

    def supersede(
        self,
        memory_id: str,
        replacement_memory_id: str,
        *,
        reason: str,
        evidence_refs: tuple[str, ...],
        actor_kind: MemoryActorKind = "system",
        actor_id: str = "memory_service",
    ) -> MemoryLifecycleResult:
        projection = self._writable_projection()
        source = _entry(projection, memory_id)
        replacement = _entry(projection, replacement_memory_id)
        if source.effective_status != "active":
            raise MemoryModelError("supersede source must be active")
        if replacement.effective_status != "proposed":
            raise MemoryModelError("supersede replacement must be proposed")
        if replacement.merge_source_memory_ids:
            raise MemoryModelError("supersede replacement cannot be a pending merge proposal")
        _require_compatible_target((source,), replacement)
        return self._append_lifecycle(
            projection=projection,
            action="supersede",
            memory_ids=(memory_id,),
            replacement_memory_id=replacement_memory_id,
            reason=reason,
            evidence_refs=evidence_refs,
            actor_kind=actor_kind,
            actor_id=actor_id,
        )

    def invalidate(
        self,
        memory_id: str,
        *,
        reason: str,
        evidence_refs: tuple[str, ...],
        actor_kind: MemoryActorKind = "system",
        actor_id: str = "memory_service",
    ) -> MemoryLifecycleResult:
        projection = self._writable_projection()
        current = _entry(projection, memory_id)
        if current.effective_status == "invalidated":
            raise MemoryModelError("memory is already invalidated")
        return self._append_lifecycle(
            projection=projection,
            action="invalidate",
            memory_ids=(memory_id,),
            reason=reason,
            evidence_refs=evidence_refs,
            actor_kind=actor_kind,
            actor_id=actor_id,
        )

    def propose_merge(
        self,
        memory_ids: tuple[str, ...],
        proposal_memory_id: str,
        *,
        reason: str,
        evidence_refs: tuple[str, ...],
        actor_kind: MemoryActorKind = "system",
        actor_id: str = "memory_service",
    ) -> MemoryLifecycleResult:
        if len(memory_ids) < 2 or len(set(memory_ids)) != len(memory_ids):
            raise MemoryModelError("propose_merge requires at least two unique source memories")
        projection = self._writable_projection()
        sources = tuple(_entry(projection, memory_id) for memory_id in memory_ids)
        proposal = _entry(projection, proposal_memory_id)
        if any(source.effective_status != "active" for source in sources):
            raise MemoryModelError("merge sources must be active")
        if proposal.effective_status != "proposed":
            raise MemoryModelError("merge proposal must be proposed")
        if proposal.merge_source_memory_ids:
            raise MemoryModelError("merge proposal already has a pending source set")
        _require_compatible_target(sources, proposal)
        return self._append_lifecycle(
            projection=projection,
            action="propose_merge",
            memory_ids=memory_ids,
            proposal_memory_id=proposal_memory_id,
            reason=reason,
            evidence_refs=evidence_refs,
            actor_kind=actor_kind,
            actor_id=actor_id,
        )

    def confirm(
        self,
        proposal_memory_id: str,
        *,
        reason: str,
        evidence_refs: tuple[str, ...],
        confirmed_by_user_id: str,
        confirmation_session_id: str,
        actor_id: str | None = None,
    ) -> MemoryLifecycleResult:
        projection = self._writable_projection()
        proposal = _entry(projection, proposal_memory_id)
        if proposal.effective_status != "proposed":
            raise MemoryModelError("confirm target must be proposed")
        if proposal.merge_source_memory_ids:
            sources = tuple(_entry(projection, memory_id) for memory_id in proposal.merge_source_memory_ids)
            if any(source.effective_status != "active" for source in sources):
                raise MemoryModelError("merge confirmation requires every source to remain active")
        return self._append_lifecycle(
            projection=projection,
            action="confirm",
            memory_ids=(proposal_memory_id,),
            reason=reason,
            evidence_refs=evidence_refs,
            actor_kind="user",
            actor_id=actor_id or confirmed_by_user_id,
            confirmed_by_user_id=confirmed_by_user_id,
            confirmation_session_id=confirmation_session_id,
            require_user_confirmation=True,
        )

    def export_metadata(
        self,
        *,
        include_content: bool = False,
        allow_restricted: bool = False,
    ) -> tuple[dict[str, object], ...]:
        """Return a serializable audit view; content is opt-in and protected."""

        exported: list[dict[str, object]] = []
        for entry in self.list_views():
            record = entry.record
            if include_content and record.sensitivity == "restricted" and not allow_restricted:
                raise MemoryModelError("restricted memory content requires explicit authorization")
            item: dict[str, object] = {
                "memory_id": record.memory_id,
                "layer": record.layer,
                "effective_status": entry.effective_status,
                "scope": record.scope.to_dict(),
                "confidence": record.confidence,
                "created_at": record.to_dict()["created_at"],
                "expires_at": record.to_dict()["expires_at"],
                "sensitivity": record.sensitivity,
                "source_run_ids": record.provenance.source_run_ids,
                "source_event_ids": record.provenance.source_event_ids,
                "evidence_refs": record.provenance.evidence_refs,
                "created_event_id": entry.created_event_id,
                "latest_state_event_id": entry.latest_state_event_id,
                "lifecycle_event_ids": entry.lifecycle_event_ids,
                "replacement_memory_id": entry.replacement_memory_id,
                "merge_source_memory_ids": entry.merge_source_memory_ids,
                "confirmed_by_user_id": entry.confirmed_by_user_id,
                "diagnostics": tuple(item.code for item in entry.diagnostics),
                "content_sha256": hashlib.sha256(record.content.encode("utf-8")).hexdigest(),
            }
            if include_content:
                item["content"] = record.content
            exported.append(item)
        return tuple(exported)

    def rebuild(self) -> list[MemoryRecord]:
        records: list[MemoryRecord] = []
        for event in self._store.list_events():
            if event.event_type != "MemoryCreated" or not isinstance(event.payload, MemoryCreatedPayload):
                continue
            raw = event.payload.extensions.get("memory_record")
            if isinstance(raw, dict):
                record = MemoryRecord.from_dict(raw)
                if record.scope.project_id == self._project_id:
                    records.append(record)
        return sorted(records, key=lambda record: (record.created_at, record.memory_id))

    def search(self, query: str, *, user_id: str | None = None, session_id: str | None = None) -> list[MemoryRecord]:
        from bauhinia_agent.memory.retrieval import MemoryRetriever, QuerySignature

        return [
            hit.record
            for hit in MemoryRetriever(self).retrieve(
                QuerySignature(goal=query),
                user_id=user_id,
                session_id=session_id,
                at=self._clock(),
            )
        ]

    def _writable_projection(self) -> MemoryProjection:
        if not self._writes_enabled:
            raise MemoryWriteDisabledError("memory writes are disabled")
        projection = self.projection()
        if projection.diagnostics:
            raise MemoryModelError("memory projection has unresolved diagnostics")
        return projection

    def _append_lifecycle(
        self,
        *,
        projection: MemoryProjection,
        action: str,
        memory_ids: tuple[str, ...],
        reason: str,
        evidence_refs: tuple[str, ...],
        actor_kind: MemoryActorKind,
        actor_id: str,
        replacement_memory_id: str | None = None,
        proposal_memory_id: str | None = None,
        confirmed_by_user_id: str | None = None,
        confirmation_session_id: str | None = None,
        require_user_confirmation: bool = False,
    ) -> MemoryLifecycleResult:
        reason = _text(reason, field="reason")
        actor_id = _text(actor_id, field="actor_id")
        if actor_kind not in {"system", "user", "maintainer"}:
            raise MemoryModelError("actor_kind must be system, user, or maintainer")
        events = self._store.list_events()
        evidence = _lifecycle_evidence(
            events,
            evidence_refs,
            require_user_confirmation=require_user_confirmation,
            confirmed_by_user_id=confirmed_by_user_id,
            confirmation_session_id=confirmation_session_id,
        )
        affected_ids = _affected_ids(
            action,
            memory_ids,
            replacement_memory_id=replacement_memory_id,
            proposal_memory_id=proposal_memory_id,
        )
        if action == "confirm":
            affected_ids = tuple(
                dict.fromkeys(
                    (
                        *affected_ids,
                        *_entry(
                            projection,
                            memory_ids[0],
                        ).merge_source_memory_ids,
                    )
                )
            )
        basis_event_ids = tuple(_entry(projection, memory_id).latest_state_event_id for memory_id in affected_ids)
        current_projection = build_memory_projection(
            events,
            project_id=self._project_id,
        )
        current_basis = tuple(_entry(current_projection, memory_id).latest_state_event_id for memory_id in affected_ids)
        if current_basis != basis_event_ids:
            return MemoryLifecycleResult(
                False,
                False,
                diagnostic=MemoryDiagnostic(
                    "memory_lifecycle_concurrent_change",
                    "Memory state changed before the lifecycle event could be recorded",
                ),
            )
        change_id = new_evo_id("memory_change")
        payload = MemoryLifecycleChangedPayload(
            lifecycle_schema_version="v1",
            change_id=change_id,
            project_id=self._project_id,
            action=action,
            memory_ids=memory_ids,
            reason=redact_text(reason)[0],
            evidence_refs=evidence_refs,
            actor_kind=actor_kind,
            actor_id=redact_text(actor_id)[0],
            basis_event_ids=basis_event_ids,
            replacement_memory_id=replacement_memory_id,
            proposal_memory_id=proposal_memory_id,
            confirmed_by_user_id=(None if confirmed_by_user_id is None else redact_text(_text(confirmed_by_user_id, field="confirmed_by_user_id"))[0]),
            extensions={
                "service_version": "p3-audit-fix-2",
                "confirmation_session_id": (
                    None
                    if confirmation_session_id is None
                    else redact_text(
                        _text(
                            confirmation_session_id,
                            field="confirmation_session_id",
                        )
                    )[0]
                ),
            },
        )
        event = EvoEvent(
            event_id=new_evo_id("event"),
            event_type="MemoryLifecycleChanged",
            sequence=len(events) + 1,
            refs=EvoReferences(
                run_id=evidence[0].run_id,
                memory_id=(replacement_memory_id or proposal_memory_id or memory_ids[0]),
                parent_event_id=basis_event_ids[-1],
            ),
            payload=payload,
        )
        try:
            appended = self._store.append(event)
        except EvoStoreError as error:
            return MemoryLifecycleResult(
                False,
                False,
                diagnostic=MemoryDiagnostic("memory_lifecycle_recording_failed", str(error)),
            )
        except Exception as error:  # noqa: BLE001 - lifecycle failure must not mutate source facts
            return MemoryLifecycleResult(
                False,
                False,
                diagnostic=MemoryDiagnostic(
                    "memory_lifecycle_recording_failed",
                    f"unexpected Memory lifecycle recorder failure: {error}",
                ),
            )
        rebuilt = self.projection()
        diagnostic = next(
            (item for item in rebuilt.diagnostics if item.event_id == appended.event.event_id),
            None,
        )
        if diagnostic is not None:
            return MemoryLifecycleResult(
                True,
                False,
                appended.event,
                MemoryDiagnostic(diagnostic.code, diagnostic.message),
            )
        store_diagnostic = appended.diagnostic
        return MemoryLifecycleResult(
            True,
            True,
            appended.event,
            None if store_diagnostic is None else MemoryDiagnostic(store_diagnostic.code, store_diagnostic.message),
        )


def _entry(projection: MemoryProjection, memory_id: str) -> MemoryProjectionEntry:
    require_evo_id(memory_id, field="memory_id", kind="memory")
    entry = projection.get(memory_id)
    if entry is None:
        raise MemoryModelError(f"unknown memory: {memory_id}")
    if entry.diagnostics:
        raise MemoryModelError(f"memory has unresolved projection diagnostics: {memory_id}")
    return entry


def _require_compatible_target(
    sources: tuple[MemoryProjectionEntry, ...],
    target: MemoryProjectionEntry,
) -> None:
    if not sources:
        raise MemoryModelError("lifecycle target requires at least one source")
    if any(source.record.layer != target.record.layer for source in sources):
        raise MemoryModelError("lifecycle target must use the same Memory layer")
    if any(not _scope_within(target.record.scope, source.record.scope) for source in sources):
        raise MemoryModelError("lifecycle target cannot broaden source scope")
    sensitivity_order = {"public": 0, "internal": 1, "restricted": 2}
    if any(sensitivity_order[target.record.sensitivity] < sensitivity_order[source.record.sensitivity] for source in sources):
        raise MemoryModelError("lifecycle target cannot reduce source sensitivity")


def _scope_within(target: MemoryScope, source: MemoryScope) -> bool:
    return target.project_id == source.project_id and (source.session_id is None or target.session_id == source.session_id) and (source.user_id is None or target.user_id == source.user_id)


def _lifecycle_evidence(
    events: list[EvoEvent],
    evidence_refs: tuple[str, ...],
    *,
    require_user_confirmation: bool,
    confirmed_by_user_id: str | None,
    confirmation_session_id: str | None,
) -> tuple[EvidenceRecord, ...]:
    try:
        records = resolve_evidence_records(
            events,
            evidence_refs,
            require_verified=True,
        )
    except (EvidenceIntegrityError, ValueError) as error:
        raise MemoryModelError(f"Memory lifecycle Evidence is invalid: {error}") from error
    if len({record.run_id for record in records}) != 1:
        raise MemoryModelError("Memory lifecycle Evidence must belong to one Run")
    if require_user_confirmation:
        expected = (
            _text(confirmed_by_user_id, field="confirmed_by_user_id"),
            _text(confirmation_session_id, field="confirmation_session_id"),
        )
        if not any(user_confirmation_identity(record) == expected for record in records):
            raise MemoryModelError("confirm requires trusted user-input Evidence for the same user and session")
    return records


def _affected_ids(
    action: str,
    memory_ids: tuple[str, ...],
    *,
    replacement_memory_id: str | None,
    proposal_memory_id: str | None,
) -> tuple[str, ...]:
    extra = replacement_memory_id if action == "supersede" else proposal_memory_id
    return tuple(dict.fromkeys((*memory_ids, *((extra,) if extra is not None else ()))))


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MemoryModelError(f"{field} must be a non-blank string")
    return value.strip()


def _memory_create_event_id(memory_id: str) -> str:
    """Make duplicate Memory creation collide atomically at the event store lock."""

    digest = hashlib.sha256(memory_id.encode("utf-8")).hexdigest()
    return f"event_memory_create_{digest}"
