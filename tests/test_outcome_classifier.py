from __future__ import annotations

import pytest

from bauhinia_agent.evolution.evidence import EvidenceAdapter, EvidenceInput
from bauhinia_agent.evolution.identifiers import new_evo_id
from bauhinia_agent.evolution.outcomes import OutcomeClassifier
from bauhinia_agent.evolution.store import EvoEventStore, EvoStoreError


@pytest.mark.parametrize(
    ("evidence", "expected_outcome", "expected_category"),
    [
        (("test", "pytest", "1 failed", 1), "failure", "verification_failure"),
        (("tool", "shell", "invalid tool argument: path", 1), "failure", "tool_failure"),
        (("tool", "fetch", "network unavailable", 1), "failure", "environment_failure"),
        (("permission", "permission_manager", "deny: user denied", None), "failure", "permission_denied"),
        (("tool", "shell", "operation cancelled", None), "cancelled", "cancelled"),
        (("test", "pytest", "command timed out", None), "timeout", "timeout"),
        (("test", "evaluator", "evaluation infrastructure unavailable", 1), "failure", "evaluation_infrastructure_failure"),
        (("manual", "operator", "task failed acceptance criteria", None), "failure", "task_failure"),
        (("manual", "operator", "observation incomplete", None), "unknown", "unknown"),
    ],
)
def test_classifier_distinguishes_required_failure_categories(tmp_path, evidence, expected_outcome, expected_category) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    adapter = EvidenceAdapter(store)
    classifier = OutcomeClassifier(store)
    run_id = new_evo_id("run")
    evidence_type, source, summary, exit_code = evidence
    recorded = adapter.record(
        EvidenceInput(
            run_id=run_id,
            evidence_type=evidence_type,
            source=source,
            summary=summary,
            exit_code=exit_code,
        )
    )

    result = classifier.classify(run_id)

    assert result.persisted is True
    assert result.outcome is not None
    assert result.outcome.payload.outcome == expected_outcome
    assert result.outcome.payload.category == expected_category
    assert result.outcome.payload.confidence > 0
    assert recorded.evidence is not None
    assert result.outcome.payload.evidence_refs == (recorded.evidence.evidence_id,)
    assert classifier.list_for_run(run_id) == [result.outcome]


def test_classifier_records_verified_success_with_evidence_chain(tmp_path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    adapter = EvidenceAdapter(store)
    classifier = OutcomeClassifier(store)
    run_id = new_evo_id("run")
    recorded = adapter.record(
        EvidenceInput(
            run_id=run_id,
            evidence_type="test",
            source="pytest",
            summary="12 passed",
            exit_code=0,
            verified=True,
        )
    )

    result = classifier.classify(run_id)

    assert result.outcome is not None
    assert result.outcome.payload.outcome == "success"
    assert result.outcome.payload.category == "task_success"
    assert result.outcome.payload.confidence == 0.95
    assert recorded.evidence is not None
    assert result.outcome.payload.evidence_refs == (recorded.evidence.evidence_id,)


def test_unknown_outcome_without_evidence_is_persisted_with_zero_confidence(tmp_path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    classifier = OutcomeClassifier(store)
    run_id = new_evo_id("run")

    result = classifier.classify(run_id)

    assert result.outcome is not None
    assert result.outcome.payload.outcome == "unknown"
    assert result.outcome.payload.category == "unknown"
    assert result.outcome.payload.confidence == 0
    assert result.outcome.payload.evidence_refs == ()


def test_outcome_recorder_failure_is_reported_without_raising() -> None:
    result = OutcomeClassifier(_FailingStore()).classify(new_evo_id("run"))

    assert result.persisted is False
    assert result.outcome is None
    assert result.diagnostic is not None
    assert result.diagnostic.code == "outcome_recording_failed"


class _FailingStore:
    def append(self, event: object) -> object:
        raise EvoStoreError("store offline")

    def list_events(self) -> list[object]:
        return []
