from __future__ import annotations

from datetime import UTC, datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from bauhinia_agent.evolution import EvidenceAdapter, EvidenceInput, EvoEventStore
from bauhinia_agent.memory import MemoryModelError, MemoryProvenance, MemoryRecord, MemoryScope, MemoryService, MemoryWriteDisabledError


def _record(
    memory_id: str,
    *,
    project: str = "project_a",
    user: str | None = None,
    created_at: datetime | None = None,
    provenance: MemoryProvenance | None = None,
) -> MemoryRecord:
    now = created_at or datetime.now(UTC).replace(microsecond=0)
    return MemoryRecord(
        memory_id=memory_id,
        layer="semantic",
        content="Provider migration uses append only events",
        scope=MemoryScope(project_id=project, user_id=user),
        provenance=provenance
        or MemoryProvenance(
            origin="verified_evidence",
            source_run_ids=("run_1",),
            source_event_ids=("event_1",),
            evidence_refs=("evidence_1",),
        ),
        confidence=0.8,
        created_at=now,
        expires_at=now + timedelta(days=1),
    )


def test_memory_source_rebuild_and_deterministic_search(tmp_path) -> None:
    created_at = datetime(2026, 8, 8, tzinfo=UTC)
    store = EvoEventStore(tmp_path)
    provenance = _source_provenance(store)
    service = MemoryService(
        store=store,
        project_id="project_a",
        clock=lambda: created_at + timedelta(hours=1),
    )


def _source_provenance(store: EvoEventStore) -> MemoryProvenance:
    result = EvidenceAdapter(store).record(
        EvidenceInput(
            run_id="run_1",
            evidence_type="test",
            source="memory-service-test",
            summary="verified source fact",
            verified=True,
            command="pytest -q",
            exit_code=0,
        )
    )
    assert result.evidence is not None
    return MemoryProvenance(
        origin="verified_evidence",
        source_run_ids=("run_1",),
        source_event_ids=(result.evidence.event_id,),
        evidence_refs=(result.evidence.evidence_id,),
    )
    service.create(_record("memory_b", created_at=created_at, provenance=provenance))
    service.create(_record("memory_a", created_at=created_at, provenance=provenance))

    assert [record.memory_id for record in service.rebuild()] == ["memory_a", "memory_b"]
    assert [record.memory_id for record in service.search("provider events")] == ["memory_a", "memory_b"]


def test_project_and_user_isolation_and_disable_control(tmp_path) -> None:
    store = EvoEventStore(tmp_path)
    provenance = _source_provenance(store)
    service = MemoryService(store=store, project_id="project_a")
    with pytest.raises(MemoryModelError, match="outside this project"):
        service.create(_record("memory_other", project="project_b", provenance=provenance))
    service.create(_record("memory_user", user="user_1", provenance=provenance))
    assert service.search("provider", user_id="user_2") == []
    assert [item.memory_id for item in service.search("provider", user_id="user_1")] == ["memory_user"]
    service.set_writes_enabled(False)
    with pytest.raises(MemoryWriteDisabledError):
        service.create(_record("memory_disabled", provenance=provenance))


def test_create_rejects_missing_provenance_and_concurrent_duplicate(tmp_path) -> None:
    root = tmp_path / ".bauhinia-agent"
    source_store = EvoEventStore(root)
    provenance = _source_provenance(source_store)
    record = _record("memory_concurrent", provenance=provenance)
    barrier = Barrier(2)

    def create_once() -> str:
        service = MemoryService(
            store=EvoEventStore(root),
            project_id="project_a",
        )
        barrier.wait(timeout=5)
        try:
            service.create(record)
        except MemoryModelError:
            return "rejected"
        return "created"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(lambda _index: create_once(), range(2)))

    assert sorted(results) == ["created", "rejected"]
    created = [event for event in source_store.list_events() if event.event_type == "MemoryCreated" and event.refs.memory_id == record.memory_id]
    assert len(created) == 1

    with pytest.raises(MemoryModelError, match="missing or future"):
        MemoryService(
            store=EvoEventStore(tmp_path / "missing-source"),
            project_id="project_a",
        ).create(_record("memory_forged"))
