from __future__ import annotations

from datetime import UTC, datetime, timedelta

from bauhinia_agent.evolution import EvoEventStore
from bauhinia_agent.memory import MemoryProvenance, MemoryRecord, MemoryRetriever, MemoryScope, MemoryService, QuerySignature


def _record(identifier: str, content: str, group: str | None = None) -> MemoryRecord:
    now = datetime.now(UTC).replace(microsecond=0)
    return MemoryRecord(
        identifier, "semantic", content, MemoryScope("project"), MemoryProvenance("verified_evidence", ("run_1",), ("event_1",), ("evidence_1",)), 0.8, now, now + timedelta(days=1), group
    )


def test_retrieval_pack_and_feedback_are_explainable(tmp_path) -> None:
    service = MemoryService(store=EvoEventStore(tmp_path), project_id="project")
    service.create(_record("memory_a", "provider timeout uses retry", "conflict_provider"))
    service.create(_record("memory_b", "provider timeout uses fallback", "conflict_provider"))
    retriever = MemoryRetriever(service)
    signature = QuerySignature(goal="fix provider timeout", error_type="timeout", constraints=("offline",))
    pack = retriever.pack(signature, token_budget=10)
    assert [hit.record.memory_id for hit in pack.hits] == ["memory_a"]
    assert pack.hits[0].reasons[0] == "project_scope"
    assert pack.omitted == (("memory_b", "conflict_group"),)
    retriever.record_use(pack.hits[0], reason="selected", helpfulness="helpful")
    assert [event.event_type for event in service._store.list_events()][-1] == "MemoryUsed"


def test_pack_explains_budget_omission(tmp_path) -> None:
    service = MemoryService(store=EvoEventStore(tmp_path), project_id="project")
    service.create(_record("memory_a", "provider timeout retry fallback"))
    pack = MemoryRetriever(service).pack(QuerySignature(goal="provider timeout"), token_budget=1)
    assert pack.hits == ()
    assert pack.omitted == (("memory_a", "token_budget"),)


def test_expired_memory_is_not_retrieved(tmp_path) -> None:
    service = MemoryService(store=EvoEventStore(tmp_path), project_id="project")
    now = datetime.now(UTC).replace(microsecond=0)
    record = _record("memory_expired", "provider timeout retry")
    expired = MemoryRecord(record.memory_id, record.layer, record.content, record.scope, record.provenance, record.confidence, now - timedelta(days=2), now - timedelta(days=1))
    service.create(expired)
    assert MemoryRetriever(service).retrieve(QuerySignature(goal="provider timeout"), at=now) == []
