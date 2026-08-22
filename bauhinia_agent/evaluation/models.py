"""Provider-independent evaluation Case, Variant, Trial, and Evaluator contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

EvalSplit = Literal["source", "development", "held_out"]
EvalVariantKind = Literal["baseline", "candidate"]
EvalTaskOutcome = Literal["task_success", "task_failure", "cancelled", "not_run"]
EvalStatus = Literal["completed", "evaluator_failure", "invalid", "cancelled"]


@dataclass(frozen=True, slots=True)
class EvalCase:
    case_id: str
    corpus_id: str
    corpus_version: str
    split: EvalSplit
    public_input: str
    task_input_hash: str
    workspace_baseline_hash: str
    environment_hash: str
    origin_run_ids: tuple[str, ...] = ()
    origin_evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EvalVariant:
    variant_id: str
    kind: EvalVariantKind
    model_config_hash: str
    strategy_hash: str
    artifact_id: str | None = None
    artifact_version: int | None = None


@dataclass(frozen=True, slots=True)
class EvalObservation:
    task_outcome: EvalTaskOutcome
    evaluation_status: EvalStatus = "completed"
    verification_quality: float | None = None
    cost: float | None = None
    latency_ms: float | None = None
    risk_events: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    verification_commands: tuple[str, ...] = ()
    verification_skipped: bool = False
    verification_coverage: float = 1.0
    claimed_success: bool | None = None
    evidence_success: bool | None = None
    output_truncated: bool = False
    accessed_resource_hashes: tuple[str, ...] = ()
    invalid_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EvalRunInput:
    """Public evaluator input; it deliberately has no private reference answer."""

    run_id: str
    case_id: str
    corpus_id: str
    corpus_version: str
    split: EvalSplit
    public_input: str
    workspace_baseline_hash: str
    environment_hash: str
    variant: EvalVariant
    seed: int


class Evaluator(Protocol):
    version: str

    def evaluate(self, request: EvalRunInput) -> EvalObservation: ...
