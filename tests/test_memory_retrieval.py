from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from bauhinia_agent.evolution import (
    EvidenceRecordedPayload,
    EvoEvent,
    EvoEventStore,
    EvoReferences,
    OutcomeClassifiedPayload,
)
from bauhinia_agent.memory import (
    MemoryAccessAuthorization,
    MemoryProvenance,
    MemoryRecord,
    MemoryRetriever,
    MemoryScope,
    MemoryService,
    QuerySignature,
)

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)
PROJECT_ID = "project_bauhinia"
SOURCE_RUN_ID = "run_memory_source"
CURRENT_RUN_ID = "run_current_task"


def _record(
    memory_id: str,
    content: str,
    *,
    status: str = "active",
    confidence: float = 0.8,
    created_at: datetime | None = None,
    expires_at: datetime | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    layer: str = "semantic",
    conflict_group: str | None = None,
    sensitivity: str = "internal",
) -> MemoryRecord:
    created = created_at or NOW - timedelta(days=1)
    expires = expires_at or NOW + timedelta(days=9)
    if layer == "task" and session_id is None:
        session_id = "session_default"
    return MemoryRecord(
        memory_id=memory_id,
        layer=layer,  # type: ignore[arg-type]
        content=content,
        scope=MemoryScope(
            project_id=PROJECT_ID,
            user_id=user_id,
            session_id=session_id,
        ),
        provenance=MemoryProvenance(
            origin="verified_evidence",
            source_run_ids=(SOURCE_RUN_ID,),
            source_event_ids=("event_memory_source",),
            evidence_refs=("evidence_memory_source",),
        ),
        confidence=confidence,
        created_at=created,
        expires_at=expires,
        conflict_group=conflict_group,
        status=status,  # type: ignore[arg-type]
        sensitivity=sensitivity,  # type: ignore[arg-type]
    )


def _service(tmp_path: Path) -> tuple[MemoryService, EvoEventStore]:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    _append_evidence(
        store,
        event_id="event_memory_source",
        evidence_id="evidence_memory_source",
        run_id=SOURCE_RUN_ID,
    )
    return MemoryService(store=store, project_id=PROJECT_ID), store


def _append_evidence(
    store: EvoEventStore,
    *,
    event_id: str,
    evidence_id: str,
    run_id: str,
    evidence_type: str = "test",
    verified: bool = True,
    exit_code: int | None = 0,
) -> EvoEvent[EvidenceRecordedPayload]:
    appended = store.append(
        EvoEvent(
            event_id=event_id,
            event_type="EvidenceRecorded",
            refs=EvoReferences(run_id=run_id, evidence_id=evidence_id),
            payload=EvidenceRecordedPayload(
                evidence_type=evidence_type,
                source="memory retrieval test",
                summary="independent verification evidence",
                verified=verified,
                command="pytest -q" if evidence_type == "test" else None,
                exit_code=exit_code,
            ),
        )
    )
    assert isinstance(appended.event.payload, EvidenceRecordedPayload)
    return appended.event


def _append_outcome(
    store: EvoEventStore,
    *,
    event_id: str,
    run_id: str,
    evidence_refs: tuple[str, ...],
) -> EvoEvent[OutcomeClassifiedPayload]:
    evidence = [event.payload for event in store.list_events() if event.event_type == "EvidenceRecorded" and event.refs.run_id == run_id and event.refs.evidence_id in evidence_refs]
    if any(item.evidence_type in {"test", "lint", "type_check", "build", "diff"} and item.exit_code not in {None, 0} for item in evidence):
        outcome, category, confidence = "failure", "verification_failure", 0.95
    elif any(item.evidence_type in {"test", "lint", "type_check", "build", "diff"} and item.verified and item.exit_code == 0 for item in evidence):
        outcome, category, confidence = "success", "task_success", 0.95
    else:
        outcome, category, confidence = "unknown", "unknown", 0.2
    appended = store.append(
        EvoEvent(
            event_id=event_id,
            event_type="OutcomeClassified",
            refs=EvoReferences(run_id=run_id),
            payload=OutcomeClassifiedPayload(
                outcome=outcome,
                category=category,
                summary=f"{category} classified from {len(evidence)} evidence record(s)",
                evidence_refs=evidence_refs,
                confidence=confidence,
            ),
        )
    )
    assert isinstance(appended.event.payload, OutcomeClassifiedPayload)
    return appended.event


def _append_duplicate_create(
    store: EvoEventStore,
    *,
    memory_id: str,
) -> None:
    source = next(event for event in store.list_events() if event.event_type == "MemoryCreated" and event.refs.memory_id == memory_id)
    store.append(
        EvoEvent(
            event_id="event_duplicate_memory_create",
            event_type="MemoryCreated",
            refs=EvoReferences(run_id=SOURCE_RUN_ID, memory_id=memory_id),
            payload=source.payload,
        )
    )


class _CharEstimator:
    estimator_id = "test-char-v1"

    def estimate(self, text: str) -> int:
        return len(text)


CHAR_ESTIMATOR = _CharEstimator()


def test_retrieve_filters_effective_state_expiry_diagnostics_scope_and_sensitivity(
    tmp_path: Path,
) -> None:
    service, store = _service(tmp_path)
    service.create(_record("memory_active", "needle active fact", confidence=0.9))
    service.create(_record("memory_proposed", "needle proposed fact", status="proposed"))
    service.create(
        _record(
            "memory_expired",
            "needle expired fact",
            created_at=NOW - timedelta(days=10),
            expires_at=NOW - timedelta(days=1),
        )
    )
    service.create(
        _record(
            "memory_other_user",
            "needle other user fact",
            user_id="user_other",
        )
    )
    service.create(
        _record(
            "memory_other_session",
            "needle other session fact",
            layer="task",
            session_id="session_other",
        )
    )
    service.create(
        _record(
            "memory_restricted",
            "needle restricted fact",
            sensitivity="restricted",
            confidence=0.85,
        )
    )
    service.create(_record("memory_invalidated", "needle invalidated fact"))
    service.create(_record("memory_superseded", "needle superseded fact"))
    service.create(_record("memory_replacement", "replacement without query term", status="proposed"))
    service.create(_record("memory_diagnostic", "needle diagnostic fact"))
    lifecycle_evidence = _append_evidence(
        store,
        event_id="event_lifecycle_evidence",
        evidence_id="evidence_lifecycle",
        run_id="run_lifecycle",
    )
    assert service.invalidate(
        "memory_invalidated",
        reason="the fact was disproved",
        evidence_refs=(lifecycle_evidence.refs.evidence_id,),  # type: ignore[arg-type]
    ).applied
    assert service.supersede(
        "memory_superseded",
        "memory_replacement",
        reason="a verified replacement exists",
        evidence_refs=(lifecycle_evidence.refs.evidence_id,),  # type: ignore[arg-type]
    ).applied
    _append_duplicate_create(store, memory_id="memory_diagnostic")

    retriever = MemoryRetriever(service, clock=lambda: NOW)
    signature = QuerySignature(goal="needle")

    default_hits = retriever.retrieve(
        signature,
        user_id="user_current",
        session_id="session_current",
    )
    assert [hit.record.memory_id for hit in default_hits] == ["memory_active"]
    assert default_hits[0].reasons[0] == "project_scope"
    assert any(reason.startswith("keyword:") for reason in default_hits[0].reasons)

    with pytest.raises(ValueError, match="authorization context"):
        retriever.retrieve(
            signature,
            user_id="user_current",
            session_id="session_current",
            include_restricted=True,
        )
    authorized_hits = retriever.retrieve(
        signature,
        user_id="user_current",
        session_id="session_current",
        include_restricted=True,
        authorization=MemoryAccessAuthorization(
            project_id=PROJECT_ID,
            user_id="user_current",
            session_id="session_current",
            allow_restricted=True,
        ),
    )
    assert [hit.record.memory_id for hit in authorized_hits] == [
        "memory_active",
        "memory_restricted",
    ]
    assert (
        retriever.retrieve(
            signature,
            user_id="user_current",
            session_id="session_current",
            at=NOW,
        )
        == default_hits
    )


def test_retrieve_uses_stable_lexicographic_ranking_policy(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)
    records = (
        _record("memory_00_exact", "E_CONN", confidence=0.1),
        _record("memory_10_coverage", "provider timeout retry", confidence=0.1),
        _record("memory_20_confidence", "provider timeout", confidence=0.95),
        _record(
            "memory_30_fresh",
            "provider timeout",
            confidence=0.8,
            created_at=NOW - timedelta(days=1),
            expires_at=NOW + timedelta(days=9),
        ),
        _record(
            "memory_31_stale",
            "provider timeout",
            confidence=0.8,
            created_at=NOW - timedelta(days=9),
            expires_at=NOW + timedelta(days=1),
        ),
        _record(
            "memory_40_scoped",
            "provider timeout",
            confidence=0.7,
            user_id="user_1",
        ),
        _record("memory_41_project", "provider timeout", confidence=0.7),
        _record("memory_50_id_a", "provider timeout", confidence=0.6),
        _record("memory_50_id_b", "provider timeout", confidence=0.6),
    )
    for record in records:
        service.create(record)

    hits = MemoryRetriever(service, clock=lambda: NOW).retrieve(
        QuerySignature(
            goal="provider timeout retry",
            error_type="E_CONN",
        ),
        user_id="user_1",
    )

    assert [hit.record.memory_id for hit in hits] == [
        "memory_00_exact",
        "memory_10_coverage",
        "memory_20_confidence",
        "memory_30_fresh",
        "memory_31_stale",
        "memory_40_scoped",
        "memory_41_project",
        "memory_50_id_a",
        "memory_50_id_b",
    ]


def test_unresolved_conflict_group_omits_every_member_from_context(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)
    service.create(
        _record(
            "memory_conflict_a",
            "provider timeout uses retry",
            conflict_group="conflict_provider",
        )
    )
    service.create(
        _record(
            "memory_conflict_b",
            "provider timeout uses fallback",
            conflict_group="conflict_provider",
        )
    )
    service.create(_record("memory_uncontested", "provider timeout has a deadline"))

    pack = MemoryRetriever(service, clock=lambda: NOW).pack(
        QuerySignature(goal="provider timeout"),
        token_budget=10_000,
    )

    assert [hit.record.memory_id for hit in pack.hits] == ["memory_uncontested"]
    assert tuple(memory_id for memory_id, _ in pack.omitted) == (
        "memory_conflict_a",
        "memory_conflict_b",
    )
    assert all("conflict" in reason for _, reason in pack.omitted)
    assert tuple((item.memory_id, item.reason) for item in pack.omissions) == pack.omitted


def test_context_pack_items_preserve_legacy_hits_and_exact_cost_metadata(
    tmp_path: Path,
) -> None:
    service, _ = _service(tmp_path)
    record = _record("memory_pack_item", "provider timeout retry")
    service.create(record)
    pack = MemoryRetriever(
        service,
        clock=lambda: NOW,
        estimator=CHAR_ESTIMATOR,
    ).pack(
        QuerySignature(goal="provider timeout"),
        token_budget=10_000,
    )

    assert len(pack.items) == 1
    assert pack.omissions == ()
    assert [hit.record.memory_id for hit in pack.hits] == [record.memory_id]
    assert pack.omitted == ()
    item = pack.items[0]
    assert item.packed_content == record.content
    assert item.original_token_cost == pack.hits[0].token_cost
    assert item.original_token_cost >= CHAR_ESTIMATOR.estimate(record.content)
    assert item.packed_token_cost == item.original_token_cost
    assert not item.truncated
    assert item.start_offset == 0
    assert item.end_offset == len(record.content)
    assert pack.used_tokens <= pack.token_budget


def test_whole_pack_budget_handles_zero_and_item_metadata_overhead(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)
    long_memory_id = "memory_" + "x" * 180
    service.create(_record(long_memory_id, "needle"))
    retriever = MemoryRetriever(
        service,
        clock=lambda: NOW,
        estimator=CHAR_ESTIMATOR,
    )
    signature = QuerySignature(goal="needle")

    zero = retriever.pack(signature, token_budget=0)
    assert zero.items == ()
    assert zero.hits == ()
    assert zero.used_tokens == 0
    assert zero.omitted == ((long_memory_id, "token_budget"),)

    raw_content_would_fit = retriever.pack(
        signature,
        token_budget=CHAR_ESTIMATOR.estimate("needle") + 1,
    )
    assert raw_content_would_fit.items == ()
    assert raw_content_would_fit.omitted == ((long_memory_id, "token_budget"),)
    assert raw_content_would_fit.used_tokens <= raw_content_would_fit.token_budget


def test_pack_deterministically_truncates_unicode_around_query_match(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)
    content = "前缀🙂" * 30 + "超时错误" + "后缀🚀" * 30
    record = _record("memory_unicode", content)
    service.create(record)
    retriever = MemoryRetriever(
        service,
        clock=lambda: NOW,
        estimator=CHAR_ESTIMATOR,
    )
    signature = QuerySignature(goal="超时错误")
    full = retriever.pack(signature, token_budget=100_000)
    full_item = full.items[0]
    metadata_cost = full_item.original_token_cost - CHAR_ESTIMATOR.estimate(content)
    constrained_budget = metadata_cost + max(
        CHAR_ESTIMATOR.estimate("超时错误") + 8,
        CHAR_ESTIMATOR.estimate(content) // 4,
    )

    first = retriever.pack(signature, token_budget=constrained_budget)
    second = retriever.pack(signature, token_budget=constrained_budget)

    assert first.items == second.items
    assert first.omissions == second.omissions
    assert first.used_tokens == second.used_tokens
    assert first.used_tokens <= constrained_budget
    assert len(first.items) == 1
    item = first.items[0]
    assert item.truncated
    assert "超时错误" in item.packed_content
    assert 0 < item.start_offset < item.end_offset < len(content)
    assert content[item.start_offset : item.end_offset] == item.packed_content
    assert item.original_token_cost >= CHAR_ESTIMATOR.estimate(content)
    assert item.packed_token_cost >= CHAR_ESTIMATOR.estimate(item.packed_content)
    assert item.packed_token_cost < item.original_token_cost
    assert item.packed_content.encode("utf-8").decode("utf-8") == item.packed_content


def test_unicode_substring_fallback_does_not_broaden_ascii_keyword_matching(
    tmp_path: Path,
) -> None:
    service, _ = _service(tmp_path)
    service.create(_record("memory_ascii_fragment", "concatenate values"))
    service.create(_record("memory_cjk_phrase", "前缀超时错误后缀"))
    retriever = MemoryRetriever(service, clock=lambda: NOW)

    assert retriever.retrieve(QuerySignature(goal="cat")) == []
    assert [hit.record.memory_id for hit in retriever.retrieve(QuerySignature(goal="超时错误"))] == ["memory_cjk_phrase"]


def test_injected_clock_and_legacy_at_override_make_expiry_deterministic(
    tmp_path: Path,
) -> None:
    service, _ = _service(tmp_path)
    service.create(
        _record(
            "memory_expiry_boundary",
            "deadline",
            created_at=NOW - timedelta(days=1),
            expires_at=NOW,
        )
    )
    service.create(
        _record(
            "memory_one_second_left",
            "deadline",
            created_at=NOW - timedelta(days=1),
            expires_at=NOW + timedelta(seconds=1),
        )
    )
    retriever = MemoryRetriever(service, clock=lambda: NOW)
    signature = QuerySignature(goal="deadline")

    assert [hit.record.memory_id for hit in retriever.retrieve(signature)] == ["memory_one_second_left"]
    assert retriever.retrieve(signature, at=NOW + timedelta(seconds=2)) == []


def test_pack_with_run_records_context_pack_and_preserves_relationships(
    tmp_path: Path,
) -> None:
    service, store = _service(tmp_path)
    service.create(_record("memory_recorded_pack", "provider timeout"))
    retriever = MemoryRetriever(service, clock=lambda: NOW)
    signature = QuerySignature(goal="provider timeout", plan_node_id="node_1")

    pack = retriever.pack(
        signature,
        token_budget=1000,
        run_id=CURRENT_RUN_ID,
        plan_id="plan_1",
        node_id="node_1",
    )

    assert pack.context_pack_id is not None
    assert pack.recorded_event_id is not None
    recorded = next(event for event in store.list_events() if event.event_id == pack.recorded_event_id)
    assert recorded.event_type == "ContextPackRecorded"
    assert recorded.refs.run_id == CURRENT_RUN_ID
    assert recorded.refs.plan_id == "plan_1"
    assert recorded.refs.node_id == "node_1"
    assert recorded.payload.to_dict()["context_pack_id"] == pack.context_pack_id
    assert [hit.record.memory_id for hit in pack.hits] == ["memory_recorded_pack"]

    event_count = len(store.list_events())
    legacy_pack = retriever.pack(
        QuerySignature(goal="provider timeout"),
        token_budget=1000,
    )
    assert legacy_pack.context_pack_id is not None
    assert legacy_pack.recorded_event_id is None
    assert len(store.list_events()) == event_count


def test_record_use_links_current_run_pack_outcome_and_independent_evidence(
    tmp_path: Path,
) -> None:
    service, store = _service(tmp_path)
    record = _record("memory_used", "provider timeout retry")
    service.create(record)
    retriever = MemoryRetriever(service, clock=lambda: NOW)
    pack = retriever.pack(
        QuerySignature(goal="provider timeout"),
        token_budget=1000,
        run_id=CURRENT_RUN_ID,
        plan_id="plan_1",
        node_id="node_1",
    )
    verification = _append_evidence(
        store,
        event_id="event_current_verification",
        evidence_id="evidence_current_verification",
        run_id=CURRENT_RUN_ID,
    )
    outcome = _append_outcome(
        store,
        event_id="event_current_outcome",
        run_id=CURRENT_RUN_ID,
        evidence_refs=(verification.refs.evidence_id,),  # type: ignore[arg-type]
    )

    retriever.record_use(
        pack,
        record.memory_id,
        run_id=CURRENT_RUN_ID,
        reason="selected for the active plan node",
        outcome_event_id=outcome.event_id,
        verification_evidence_refs=(verification.refs.evidence_id,),  # type: ignore[arg-type]
        feedback_status="helpful",
    )

    used = store.list_events()[-1]
    assert used.event_type == "MemoryUsed"
    assert used.refs.run_id == CURRENT_RUN_ID
    assert used.refs.run_id != SOURCE_RUN_ID
    assert used.refs.memory_id == record.memory_id
    assert used.refs.plan_id == "plan_1"
    assert used.refs.node_id == "node_1"
    assert used.refs.parent_event_id == pack.recorded_event_id
    payload = used.payload.to_dict()
    assert payload["context_pack_id"] == pack.context_pack_id
    assert payload["usage_status"] == "used"
    assert payload["feedback_status"] == "helpful"
    assert payload["outcome_event_id"] == outcome.event_id
    assert payload["verification_evidence_refs"] == [verification.refs.evidence_id]


def test_record_use_uses_persisted_context_pack_not_tampered_in_memory_items(
    tmp_path: Path,
) -> None:
    service, _store = _service(tmp_path)
    first = _record("memory_canonical_first", "alpha provider timeout")
    second = _record("memory_canonical_second", "beta schema mismatch")
    service.create(first)
    service.create(second)
    retriever = MemoryRetriever(service, clock=lambda: NOW)
    recorded = retriever.pack(
        QuerySignature(goal="alpha timeout"),
        token_budget=1000,
        run_id=CURRENT_RUN_ID,
    )
    other = retriever.pack(
        QuerySignature(goal="beta schema"),
        token_budget=1000,
    )
    tampered = replace(recorded, items=other.items, omissions=other.omissions)

    with pytest.raises(ValueError, match="recorded Context Pack"):
        retriever.record_use(
            tampered,
            second.memory_id,
            run_id=CURRENT_RUN_ID,
            reason="caller-mutated pack must not become a fact",
        )


def test_helpful_feedback_requires_canonical_successful_outcome(
    tmp_path: Path,
) -> None:
    service, store = _service(tmp_path)
    record = _record("memory_failed_feedback", "provider timeout")
    service.create(record)
    retriever = MemoryRetriever(service, clock=lambda: NOW)
    pack = retriever.pack(
        QuerySignature(goal="provider timeout"),
        token_budget=1000,
        run_id=CURRENT_RUN_ID,
    )
    failed = _append_evidence(
        store,
        event_id="event_failed_feedback_evidence",
        evidence_id="evidence_failed_feedback",
        run_id=CURRENT_RUN_ID,
        exit_code=1,
    )
    outcome = _append_outcome(
        store,
        event_id="event_failed_feedback_outcome",
        run_id=CURRENT_RUN_ID,
        evidence_refs=(failed.refs.evidence_id,),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="helpful feedback requires"):
        retriever.record_use(
            pack,
            record.memory_id,
            run_id=CURRENT_RUN_ID,
            reason="a failing outcome cannot prove helpfulness",
            outcome_event_id=outcome.event_id,
            verification_evidence_refs=(failed.refs.evidence_id,),  # type: ignore[arg-type]
            feedback_status="helpful",
        )


def test_record_use_rejects_cross_run_outcome_evidence_and_unsupported_helpful_claim(
    tmp_path: Path,
) -> None:
    service, store = _service(tmp_path)
    record = _record("memory_feedback_guard", "provider timeout")
    service.create(record)
    retriever = MemoryRetriever(service, clock=lambda: NOW)
    pack = retriever.pack(
        QuerySignature(goal="provider timeout"),
        token_budget=1000,
        run_id=CURRENT_RUN_ID,
    )
    current_evidence = _append_evidence(
        store,
        event_id="event_feedback_current_evidence",
        evidence_id="evidence_feedback_current",
        run_id=CURRENT_RUN_ID,
    )
    current_outcome = _append_outcome(
        store,
        event_id="event_feedback_current_outcome",
        run_id=CURRENT_RUN_ID,
        evidence_refs=(current_evidence.refs.evidence_id,),  # type: ignore[arg-type]
    )
    other_evidence = _append_evidence(
        store,
        event_id="event_feedback_other_evidence",
        evidence_id="evidence_feedback_other",
        run_id="run_other",
    )
    other_outcome = _append_outcome(
        store,
        event_id="event_feedback_other_outcome",
        run_id="run_other",
        evidence_refs=(other_evidence.refs.evidence_id,),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="Run"):
        retriever.record_use(
            pack,
            record.memory_id,
            run_id=CURRENT_RUN_ID,
            reason="cross-run outcome must not count",
            outcome_event_id=other_outcome.event_id,
        )
    with pytest.raises(ValueError, match="Run"):
        retriever.record_use(
            pack,
            record.memory_id,
            run_id=CURRENT_RUN_ID,
            reason="cross-run Evidence must not count",
            outcome_event_id=current_outcome.event_id,
            verification_evidence_refs=(other_evidence.refs.evidence_id,),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="Evidence|evidence|independent"):
        retriever.record_use(
            pack,
            record.memory_id,
            run_id=CURRENT_RUN_ID,
            reason="self-report is not independent verification",
            outcome_event_id=current_outcome.event_id,
            feedback_status="helpful",
        )


def test_record_use_requires_deterministic_evidence_after_pack_before_outcome(
    tmp_path: Path,
) -> None:
    service, store = _service(tmp_path)
    record = _record("memory_feedback_order", "provider timeout")
    service.create(record)
    early_evidence = _append_evidence(
        store,
        event_id="event_feedback_early_evidence",
        evidence_id="evidence_feedback_early",
        run_id=CURRENT_RUN_ID,
    )
    retriever = MemoryRetriever(service, clock=lambda: NOW)
    pack = retriever.pack(
        QuerySignature(goal="provider timeout"),
        token_budget=1000,
        run_id=CURRENT_RUN_ID,
    )
    early_outcome = _append_outcome(
        store,
        event_id="event_feedback_early_outcome",
        run_id=CURRENT_RUN_ID,
        evidence_refs=(early_evidence.refs.evidence_id,),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="after the Context Pack"):
        retriever.record_use(
            pack,
            record.memory_id,
            run_id=CURRENT_RUN_ID,
            reason="pre-retrieval evidence cannot establish usefulness",
            outcome_event_id=early_outcome.event_id,
            verification_evidence_refs=(early_evidence.refs.evidence_id,),  # type: ignore[arg-type]
            feedback_status="helpful",
        )

    nondeterministic = _append_evidence(
        store,
        event_id="event_feedback_tool_evidence",
        evidence_id="evidence_feedback_tool",
        run_id=CURRENT_RUN_ID,
        evidence_type="tool",
    )
    tool_outcome = _append_outcome(
        store,
        event_id="event_feedback_tool_outcome",
        run_id=CURRENT_RUN_ID,
        evidence_refs=(
            early_evidence.refs.evidence_id,  # type: ignore[arg-type]
            nondeterministic.refs.evidence_id,
        ),
    )
    with pytest.raises(ValueError, match="deterministic|exit code"):
        retriever.record_use(
            pack,
            record.memory_id,
            run_id=CURRENT_RUN_ID,
            reason="a generic tool report is not deterministic verification",
            outcome_event_id=tool_outcome.event_id,
            verification_evidence_refs=(nondeterministic.refs.evidence_id,),  # type: ignore[arg-type]
            feedback_status="helpful",
        )


def test_record_use_rejects_unrecorded_pack_and_run_mismatch(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)
    record = _record("memory_unrecorded_guard", "provider timeout")
    service.create(record)
    retriever = MemoryRetriever(service, clock=lambda: NOW)
    signature = QuerySignature(goal="provider timeout")
    unrecorded = retriever.pack(signature, token_budget=1000)

    with pytest.raises(ValueError, match="recorded|Context Pack|context pack"):
        retriever.record_use(
            unrecorded,
            record.memory_id,
            run_id=CURRENT_RUN_ID,
            reason="an in-memory pack is not an auditable source",
        )

    recorded = retriever.pack(
        signature,
        token_budget=1000,
        run_id=CURRENT_RUN_ID,
    )
    with pytest.raises(ValueError, match="Run|run"):
        retriever.record_use(
            recorded,
            record.memory_id,
            run_id="run_other",
            reason="a different Run cannot claim this pack",
        )


def test_not_used_feedback_can_reference_a_recorded_omission(tmp_path: Path) -> None:
    service, store = _service(tmp_path)
    service.create(
        _record(
            "memory_omitted_a",
            "provider timeout retry",
            conflict_group="conflict_provider",
        )
    )
    service.create(
        _record(
            "memory_omitted_b",
            "provider timeout fallback",
            conflict_group="conflict_provider",
        )
    )
    retriever = MemoryRetriever(service, clock=lambda: NOW)
    pack = retriever.pack(
        QuerySignature(goal="provider timeout"),
        token_budget=1000,
        run_id=CURRENT_RUN_ID,
        plan_id="plan_1",
        node_id="node_1",
    )
    assert pack.hits == ()
    assert tuple(memory_id for memory_id, _ in pack.omitted) == (
        "memory_omitted_a",
        "memory_omitted_b",
    )
    assert all("conflict" in reason for _, reason in pack.omitted)

    retriever.record_use(
        pack,
        "memory_omitted_a",
        run_id=CURRENT_RUN_ID,
        reason="omitted because its conflict remains unresolved",
        usage_status="not_used",
    )

    with pytest.raises(ValueError, match="only used Memory"):
        retriever.record_use(
            pack,
            "memory_omitted_a",
            run_id=CURRENT_RUN_ID,
            reason="an omitted Memory cannot be labeled helpful",
            usage_status="not_used",
            feedback_status="helpful",
        )

    event = store.list_events()[-1]
    assert event.event_type == "MemoryUsed"
    assert event.refs.run_id == CURRENT_RUN_ID
    assert event.refs.memory_id == "memory_omitted_a"
    payload = event.payload.to_dict()
    assert payload["context_pack_id"] == pack.context_pack_id
    assert payload["usage_status"] == "not_used"
    assert payload["feedback_status"] == "unknown"
