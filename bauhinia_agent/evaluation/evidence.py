"""Canonical evidence attestation for Evaluation Trials."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from bauhinia_agent.evolution.evidence import EvidenceIntegrityError, resolve_evidence_records
from bauhinia_agent.evolution.events import EvoEvent


class EvaluationEvidenceError(ValueError):
    """Raised when a completed Trial is not backed by trustworthy evidence."""


@dataclass(frozen=True, slots=True)
class EvaluationEvidenceAttestation:
    """Values derived from canonical EvidenceRecorded facts."""

    success: bool
    commands: tuple[str, ...]


def attest_evaluation_evidence(
    events: Iterable[EvoEvent],
    evidence_refs: Sequence[str],
    *,
    run_id: str,
    expected_success: bool,
    reported_evidence_success: bool | None = None,
    reported_commands: Sequence[str] = (),
    before_sequence: int | None = None,
) -> EvaluationEvidenceAttestation:
    """Validate a completed Trial and derive its evidence-backed fields."""

    try:
        records = resolve_evidence_records(
            events,
            evidence_refs,
            run_id=run_id,
            require_verified=True,
            deterministic_only=True,
            require_exit_code=True,
            before_sequence=before_sequence,
        )
    except (EvidenceIntegrityError, ValueError) as error:
        raise EvaluationEvidenceError(str(error)) from error

    observed_success = all(record.payload.exit_code == 0 for record in records)
    if observed_success != expected_success:
        expected = "success" if expected_success else "failure"
        raise EvaluationEvidenceError(f"canonical Evidence exit codes do not support the Trial's {expected} outcome")
    if reported_evidence_success is not None and reported_evidence_success != observed_success:
        raise EvaluationEvidenceError("reported evidence_success conflicts with canonical Evidence exit codes")

    commands = tuple(dict.fromkeys(record.payload.command for record in records if record.payload.command is not None))
    supplied_commands = tuple(reported_commands)
    if supplied_commands and supplied_commands != commands:
        raise EvaluationEvidenceError("reported verification_commands conflict with canonical Evidence commands")
    return EvaluationEvidenceAttestation(success=observed_success, commands=commands)
