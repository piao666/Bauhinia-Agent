from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from bauhinia_agent.evolution import (
    EvidenceAdapter,
    EvidenceError,
    EvidenceInput,
    EvidenceRecord,
    EvoEvent,
    EvoEventStore,
    EvoStoreError,
    new_evo_id,
)
from bauhinia_agent.memory import (
    MemoryModelError,
    MemoryProvenance,
    MemoryRecord,
    MemoryScope,
    MemoryService,
    MemoryWriteDisabledError,
)

CREATED_AT = datetime(2026, 8, 11, 0, 0, tzinfo=UTC)


def test_supersede_is_append_only_scope_safe_and_rebuildable(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    service = MemoryService(store=store, project_id="project_a")
    _create_memory(service, store, "memory_old")
    _create_memory(service, store, "memory_new", status="proposed")
    evidence = _evidence(store, evidence_type="test", exit_code=0)

    result = service.supersede(
        "memory_old",
        "memory_new",
        reason="Verified implementation replaced the old project fact.",
        evidence_refs=(evidence.evidence_id,),
    )

    assert result.persisted and result.applied and result.event is not None
    assert result.event.refs.run_id == evidence.run_id
    assert service.get_view("memory_old").effective_status == "superseded"
    replacement = service.get_view("memory_new")
    assert replacement.effective_status == "active"
    assert replacement.record.status == "proposed"
    assert replacement.latest_state_event_id == result.event.event_id
    rebuilt = MemoryService(store=store, project_id="project_a").projection()
    assert rebuilt == service.projection()
    assert [record.status for record in service.rebuild()] == ["proposed", "active"]


def test_merge_remains_proposed_until_verified_user_confirmation(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    service = MemoryService(store=store, project_id="project_a")
    _create_memory(service, store, "memory_a")
    _create_memory(service, store, "memory_b")
    _create_memory(service, store, "memory_merged", status="proposed")
    merge_evidence = _evidence(store, evidence_type="diff", exit_code=0)

    proposed = service.propose_merge(
        ("memory_a", "memory_b"),
        "memory_merged",
        reason="The two verified facts describe one stable constraint.",
        evidence_refs=(merge_evidence.evidence_id,),
    )

    assert proposed.applied
    assert service.get_view("memory_a").effective_status == "active"
    pending = service.get_view("memory_merged")
    assert pending.effective_status == "proposed"
    assert pending.merge_source_memory_ids == ("memory_a", "memory_b")

    non_user_evidence = _evidence(store, evidence_type="test", exit_code=0)
    with pytest.raises(MemoryModelError, match="trusted user-input"):
        service.confirm(
            "memory_merged",
            reason="A test cannot impersonate user confirmation.",
            evidence_refs=(non_user_evidence.evidence_id,),
            confirmed_by_user_id="user_1",
            confirmation_session_id="session_1",
        )

    recorded_confirmation = EvidenceAdapter(store).record_user_confirmation(
        run_id=new_evo_id("run"),
        user_id="user_1",
        session_id="session_1",
        confirmation_id="confirmation_merge_1",
        summary="The user approved the merged Memory proposal.",
    )
    assert recorded_confirmation.evidence is not None
    confirmation = recorded_confirmation.evidence
    with pytest.raises(MemoryModelError, match="same user and session"):
        service.confirm(
            "memory_merged",
            reason="A different session cannot reuse the confirmation.",
            evidence_refs=(confirmation.evidence_id,),
            confirmed_by_user_id="user_1",
            confirmation_session_id="session_other",
        )
    confirmed = service.confirm(
        "memory_merged",
        reason="The user confirmed the merged project fact.",
        evidence_refs=(confirmation.evidence_id,),
        confirmed_by_user_id="user_1",
        confirmation_session_id="session_1",
    )

    assert confirmed.applied
    assert service.get_view("memory_a").effective_status == "superseded"
    assert service.get_view("memory_b").effective_status == "superseded"
    merged = service.get_view("memory_merged")
    assert merged.effective_status == "active"
    assert merged.record.status == "proposed"
    assert merged.confirmed_by_user_id == "user_1"


def test_generic_evidence_cannot_impersonate_user_confirmation(
    tmp_path: Path,
) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    with pytest.raises(EvidenceError, match="user-input boundary"):
        EvidenceAdapter(store).record(
            EvidenceInput(
                run_id=new_evo_id("run"),
                evidence_type="user_confirmation",
                source="caller-declared",
                summary="not authenticated",
                verified=True,
            )
        )


def test_invalidate_export_and_write_disable_are_auditable(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    service = MemoryService(store=store, project_id="project_a")
    _create_memory(service, store, "memory_sensitive", sensitivity="restricted")
    evidence = _evidence(store, evidence_type="test", exit_code=1)

    invalidated = service.invalidate(
        "memory_sensitive",
        reason="Verification disproved the stored assumption token=private.",
        evidence_refs=(evidence.evidence_id,),
    )

    assert invalidated.applied
    assert service.get_view("memory_sensitive").effective_status == "invalidated"
    metadata = service.export_metadata()
    assert len(metadata) == 1
    assert "content" not in metadata[0]
    assert metadata[0]["content_sha256"]
    with pytest.raises(MemoryModelError, match="explicit authorization"):
        service.export_metadata(include_content=True)
    assert (
        service.export_metadata(
            include_content=True,
            allow_restricted=True,
        )[
            0
        ]["content"]
        == "Provider migration uses append-only events"
    )

    service.set_writes_enabled(False)
    with pytest.raises(MemoryWriteDisabledError):
        service.invalidate(
            "memory_sensitive",
            reason="No further writes are allowed.",
            evidence_refs=(evidence.evidence_id,),
        )


def test_lifecycle_rejects_dangling_evidence_scope_expansion_and_store_failure(
    tmp_path: Path,
) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    service = MemoryService(store=store, project_id="project_a")
    _create_memory(service, store, "memory_scoped", user_id="user_1")
    _create_memory(service, store, "memory_broad", status="proposed")

    with pytest.raises(MemoryModelError, match="does not exist"):
        service.invalidate(
            "memory_scoped",
            reason="Unverifiable invalidation.",
            evidence_refs=(new_evo_id("evidence"),),
        )

    evidence = _evidence(store, evidence_type="test", exit_code=0)
    with pytest.raises(MemoryModelError, match="cannot broaden"):
        service.supersede(
            "memory_scoped",
            "memory_broad",
            reason="A broad replacement would escape user scope.",
            evidence_refs=(evidence.evidence_id,),
        )

    failing = MemoryService(
        store=_FailingAppendStore(store.list_events()),  # type: ignore[arg-type]
        project_id="project_a",
    )
    failed = failing.invalidate(
        "memory_scoped",
        reason="The source Store is unavailable.",
        evidence_refs=(evidence.evidence_id,),
    )
    assert not failed.persisted and not failed.applied
    assert failed.diagnostic is not None
    assert failed.diagnostic.code == "memory_lifecycle_recording_failed"


def _create_memory(
    service: MemoryService,
    store: EvoEventStore,
    memory_id: str,
    *,
    status: str = "active",
    user_id: str | None = None,
    sensitivity: str = "internal",
) -> MemoryRecord:
    source = _evidence(store, evidence_type="test", exit_code=0)
    record = MemoryRecord(
        memory_id=memory_id,
        layer="semantic",
        content="Provider migration uses append-only events",
        scope=MemoryScope(project_id="project_a", user_id=user_id),
        provenance=MemoryProvenance(
            origin="verified_evidence",
            source_run_ids=(source.run_id,),
            source_event_ids=(source.event_id,),
            evidence_refs=(source.evidence_id,),
        ),
        confidence=0.8,
        created_at=CREATED_AT,
        expires_at=CREATED_AT + timedelta(days=30),
        status=status,  # type: ignore[arg-type]
        sensitivity=sensitivity,  # type: ignore[arg-type]
    )
    return service.create(record)


def _evidence(
    store: EvoEventStore,
    *,
    evidence_type: str,
    exit_code: int | None,
) -> EvidenceRecord:
    recorded = EvidenceAdapter(store).record(
        EvidenceInput(
            run_id=new_evo_id("run"),
            evidence_type=evidence_type,
            source="memory-lifecycle-test",
            summary="verified lifecycle evidence",
            verified=True,
            command=None if exit_code is None else "pytest -q",
            exit_code=exit_code,
        )
    )
    assert recorded.evidence is not None
    return recorded.evidence


class _FailingAppendStore:
    def __init__(self, events: list[EvoEvent]) -> None:
        self._events = list(events)

    def list_events(self) -> list[EvoEvent]:
        return list(self._events)

    def append(self, event: EvoEvent) -> object:
        del event
        raise EvoStoreError("store offline")
