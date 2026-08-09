from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from bauhinia_agent.evolution import EvoEventStore
from bauhinia_agent.memory import MemoryModelError, MemoryProvenance, MemoryRecord, MemoryScope, MemoryService, MemoryWriteDisabledError


def _record(memory_id: str, *, project: str = "project_a", user: str | None = None, created_at: datetime | None = None) -> MemoryRecord:
    now = created_at or datetime.now(UTC).replace(microsecond=0)
    return MemoryRecord(
        memory_id=memory_id,
        layer="semantic",
        content="Provider migration uses append only events",
        scope=MemoryScope(project_id=project, user_id=user),
        provenance=MemoryProvenance(origin="verified_evidence", source_run_ids=("run_1",), source_event_ids=("event_1",), evidence_refs=("evidence_1",)),
        confidence=0.8,
        created_at=now,
        expires_at=now + timedelta(days=1),
    )


def test_memory_source_rebuild_and_deterministic_search(tmp_path) -> None:
    created_at = datetime(2026, 8, 8, tzinfo=UTC)
    service = MemoryService(
        store=EvoEventStore(tmp_path),
        project_id="project_a",
        clock=lambda: created_at + timedelta(hours=1),
    )
    service.create(_record("memory_b", created_at=created_at))
    service.create(_record("memory_a", created_at=created_at))

    assert [record.memory_id for record in service.rebuild()] == ["memory_a", "memory_b"]
    assert [record.memory_id for record in service.search("provider events")] == ["memory_a", "memory_b"]


def test_project_and_user_isolation_and_disable_control(tmp_path) -> None:
    service = MemoryService(store=EvoEventStore(tmp_path), project_id="project_a")
    with pytest.raises(MemoryModelError, match="outside this project"):
        service.create(_record("memory_other", project="project_b"))
    service.create(_record("memory_user", user="user_1"))
    assert service.search("provider", user_id="user_2") == []
    assert [item.memory_id for item in service.search("provider", user_id="user_1")] == ["memory_user"]
    service.set_writes_enabled(False)
    with pytest.raises(MemoryWriteDisabledError):
        service.create(_record("memory_disabled"))
