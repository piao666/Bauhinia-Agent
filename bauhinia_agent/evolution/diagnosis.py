"""Evidence-linked, non-authoritative diagnostic summaries for Evo Runs."""

from __future__ import annotations

from dataclasses import dataclass

from bauhinia_agent.evolution.evidence import EvidenceAdapter, EvidenceRecord
from bauhinia_agent.evolution.identifiers import require_evo_id
from bauhinia_agent.evolution.outcomes import OutcomeClassifier, OutcomeRecord
from bauhinia_agent.evolution.store import EvoEventStore


@dataclass(frozen=True, slots=True)
class CandidateCause:
    """A hypothesis that is explicitly limited by linked observable evidence."""

    domain: str
    summary: str
    evidence_refs: tuple[str, ...]
    confidence: float


@dataclass(frozen=True, slots=True)
class DiagnosisSummary:
    """Derived summary for Replan and Experience Compiler consumers.

    This is not an append-only fact and is never a substitute for source
    Evidence. Candidate causes remain hypotheses even for high-confidence
    outcome categories.
    """

    run_id: str
    outcome_event_id: str | None
    failure_location: str | None
    observed_symptoms: tuple[str, ...]
    candidate_causes: tuple[CandidateCause, ...]
    excluded_causes: tuple[str, ...]
    next_verification: str | None
    uncertainty: str
    confidence: float
    evidence_refs: tuple[str, ...]


class DiagnosisService:
    """Build minimal, deterministic diagnostics without modifying a Run."""

    def __init__(self, store: EvoEventStore) -> None:
        self._evidence = EvidenceAdapter(store)
        self._outcomes = OutcomeClassifier(store)

    def diagnose(self, run_id: str) -> DiagnosisSummary:
        """Return a bounded diagnostic summary, refusing unsupported certainty."""

        require_evo_id(run_id, field="run_id", kind="run")
        evidence = self._evidence.list_for_run(run_id)
        outcomes = self._outcomes.list_for_run(run_id)
        if not outcomes:
            return _unclassified_summary(run_id, evidence)
        return _summary_from_outcome(run_id, outcomes[-1], evidence)


def _unclassified_summary(run_id: str, evidence: list[EvidenceRecord]) -> DiagnosisSummary:
    evidence_refs = tuple(record.evidence_id for record in evidence)
    if not evidence:
        uncertainty = "No Outcome or Evidence is available; no cause is asserted."
    else:
        uncertainty = "Evidence exists but no Outcome classification is available; no cause is asserted."
    return DiagnosisSummary(
        run_id=run_id,
        outcome_event_id=None,
        failure_location=None,
        observed_symptoms=_symptoms(evidence),
        candidate_causes=(),
        excluded_causes=(),
        next_verification="Collect deterministic verification, tool, environment, or permission evidence.",
        uncertainty=uncertainty,
        confidence=0.0,
        evidence_refs=evidence_refs,
    )


def _summary_from_outcome(run_id: str, outcome: OutcomeRecord, evidence: list[EvidenceRecord]) -> DiagnosisSummary:
    category = outcome.payload.category
    location, domain, cause, next_verification = _diagnosis_rule(category)
    evidence_refs = outcome.payload.evidence_refs
    candidate_causes: tuple[CandidateCause, ...] = ()
    if cause is not None:
        candidate_causes = (
            CandidateCause(
                domain=domain,
                summary=cause,
                evidence_refs=evidence_refs,
                confidence=min(outcome.payload.confidence, 0.8),
            ),
        )
    if category == "unknown":
        uncertainty = "Outcome remains unknown; the available evidence does not support a candidate cause."
    elif category == "task_success":
        uncertainty = "Evidence supports task success; no failure cause is asserted."
    else:
        uncertainty = f"Evidence supports the {category} classification; root cause remains a hypothesis."
    return DiagnosisSummary(
        run_id=run_id,
        outcome_event_id=outcome.event_id,
        failure_location=location,
        observed_symptoms=_symptoms(evidence),
        candidate_causes=candidate_causes,
        excluded_causes=(),
        next_verification=next_verification,
        uncertainty=uncertainty,
        confidence=outcome.payload.confidence,
        evidence_refs=evidence_refs,
    )


def _diagnosis_rule(category: str) -> tuple[str | None, str, str | None, str | None]:
    rules = {
        "verification_failure": (
            "verification",
            "verification",
            "A deterministic verifier reported failure.",
            "Inspect the failing verifier output and rerun the focused verification.",
        ),
        "tool_failure": (
            "tool",
            "tool",
            "A tool invocation reported failure.",
            "Validate the tool arguments and rerun the smallest safe invocation.",
        ),
        "environment_failure": (
            "environment",
            "environment",
            "The execution environment or network was unavailable.",
            "Check the environment dependency and retry only after it is available.",
        ),
        "permission_denied": (
            "permission",
            "permission",
            "The required permission was denied.",
            "Request or revise permission before retrying the action.",
        ),
        "cancelled": (
            "cancellation",
            "cancellation",
            "The Run was cancelled before completion.",
            "Confirm whether the user wants to resume from the last verified state.",
        ),
        "timeout": (
            "timeout",
            "timeout",
            "The Run exceeded its available time budget.",
            "Reduce scope or increase verification granularity before retrying.",
        ),
        "evaluation_infrastructure_failure": (
            "evaluation_infrastructure",
            "evaluation_infrastructure",
            "The evaluation infrastructure was unavailable.",
            "Repair or replace the evaluator before interpreting task quality.",
        ),
        "task_failure": (
            "task",
            "task",
            "The task acceptance evidence reported failure.",
            "Review the acceptance evidence and create a bounded replan.",
        ),
        "task_success": (None, "task", None, None),
        "unknown": (None, "unknown", None, "Collect more deterministic evidence before planning a retry."),
    }
    return rules.get(category, (None, "unknown", None, "Collect more deterministic evidence before planning a retry."))


def _symptoms(evidence: list[EvidenceRecord]) -> tuple[str, ...]:
    return tuple(f"{record.evidence_type}: {record.payload.summary}" for record in evidence[:5])
