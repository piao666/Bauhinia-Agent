from __future__ import annotations

from bauhinia_agent.evolution.diagnosis import DiagnosisService
from bauhinia_agent.evolution.evidence import EvidenceAdapter, EvidenceInput
from bauhinia_agent.evolution.identifiers import new_evo_id
from bauhinia_agent.evolution.outcomes import OutcomeClassifier
from bauhinia_agent.evolution.store import EvoEventStore


def test_diagnosis_summarizes_verification_failure_with_evidence(tmp_path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    evidence = EvidenceAdapter(store)
    classifier = OutcomeClassifier(store)
    run_id = new_evo_id("run")
    recorded = evidence.record(
        EvidenceInput(run_id=run_id, evidence_type="test", source="pytest", summary="test_api_login failed", exit_code=1, verified=True)
    )
    classifier.classify(run_id)

    diagnosis = DiagnosisService(store).diagnose(run_id)

    assert diagnosis.failure_location == "verification"
    assert diagnosis.observed_symptoms == ("test: test_api_login failed",)
    assert diagnosis.candidate_causes[0].domain == "verification"
    assert diagnosis.excluded_causes == ()
    assert diagnosis.next_verification == "Inspect the failing verifier output and rerun the focused verification."
    assert "Evidence supports" in diagnosis.uncertainty
    assert recorded.evidence is not None
    assert diagnosis.evidence_refs == (recorded.evidence.evidence_id,)


def test_diagnosis_keeps_permission_rejection_separate_from_tool_failure(tmp_path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    evidence = EvidenceAdapter(store)
    classifier = OutcomeClassifier(store)
    run_id = new_evo_id("run")
    evidence.record_permission(run_id=run_id, action="execute_shell", target="git push", decision="deny", reason="user denied")
    classifier.classify(run_id)

    diagnosis = DiagnosisService(store).diagnose(run_id)

    assert diagnosis.failure_location == "permission"
    assert diagnosis.candidate_causes[0].domain == "permission"
    assert diagnosis.next_verification == "Request or revise permission before retrying the action."
    assert diagnosis.excluded_causes == ()


def test_diagnosis_handles_cancel_timeout_and_unknown_without_overclaiming(tmp_path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    evidence = EvidenceAdapter(store)
    classifier = OutcomeClassifier(store)

    cancelled_run = new_evo_id("run")
    evidence.record(EvidenceInput(run_id=cancelled_run, evidence_type="tool", source="shell", summary="operation cancelled"))
    classifier.classify(cancelled_run)
    assert DiagnosisService(store).diagnose(cancelled_run).failure_location == "cancellation"

    timeout_run = new_evo_id("run")
    evidence.record(EvidenceInput(run_id=timeout_run, evidence_type="test", source="pytest", summary="command timed out"))
    classifier.classify(timeout_run)
    assert DiagnosisService(store).diagnose(timeout_run).failure_location == "timeout"

    unknown_run = new_evo_id("run")
    unknown = DiagnosisService(store).diagnose(unknown_run)
    assert unknown.failure_location is None
    assert unknown.candidate_causes == ()
    assert unknown.excluded_causes == ()
    assert unknown.confidence == 0
    assert unknown.next_verification == "Collect deterministic verification, tool, environment, or permission evidence."
    assert "No Outcome or Evidence" in unknown.uncertainty
