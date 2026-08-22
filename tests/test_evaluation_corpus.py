from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from bauhinia_agent.evaluation import (
    EvalCase,
    EvalCorpusCase,
    EvalCorpusError,
    EvalCorpusManifest,
    EvalCorpusRegistry,
    EvalObservation,
    EvalRunInput,
    EvalVariant,
    HeldOutEvalHarness,
    hash_text,
    private_reference_hash,
)
from bauhinia_agent.evolution.evidence import EvidenceAdapter, EvidenceInput
from bauhinia_agent.evolution.events import CandidateArtifactCreatedPayload, EvoEvent, EvoReferences
from bauhinia_agent.evolution.identifiers import new_evo_id
from bauhinia_agent.evolution.store import EvoEventStore


def test_registry_records_licensed_immutable_manifest_without_private_answer(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    registry = EvalCorpusRegistry(store)
    manifest = _manifest()

    first = registry.register(manifest)
    repeated = registry.register(manifest)

    assert first.persisted is True
    assert first.corpus is not None
    assert first.corpus.payload.license_spdx == "MIT"
    assert first.corpus.payload.extensions["private_answers_persisted"] is False
    assert repeated.persisted is False
    assert repeated.diagnostic is not None
    assert repeated.diagnostic.code == "already_registered"
    serialized = (tmp_path / ".bauhinia-agent" / "evo" / "events.jsonl").read_text(encoding="utf-8")
    assert manifest.cases[0].private_reference not in serialized
    mutated_case = replace(
        manifest.cases[0],
        private_reference="A different private answer.",
        private_reference_hash=private_reference_hash("A different private answer."),
    )
    with pytest.raises(EvalCorpusError, match="immutable"):
        registry.register(replace(manifest, cases=(mutated_case,)))


def test_registry_rejects_unapproved_license_and_cross_split_duplicate(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    manifest = _manifest()
    with pytest.raises(EvalCorpusError, match="not approved"):
        EvalCorpusRegistry(store).register(replace(manifest, license_spdx="Proprietary"))

    duplicate = replace(
        manifest.cases[0],
        case=replace(manifest.cases[0].case, case_id="eval_case_duplicate", split="development"),
    )
    with pytest.raises(EvalCorpusError, match="cannot repeat"):
        EvalCorpusRegistry(store).register(replace(manifest, cases=(*manifest.cases, duplicate)))


def test_clean_held_out_trial_exposes_only_public_case_input(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    manifest = _manifest()
    EvalCorpusRegistry(store).register(manifest)
    evaluator = _InspectingEvaluator(store)

    result = HeldOutEvalHarness(store).run(manifest, manifest.cases[0].case.case_id, _baseline(), evaluator, seed=7)

    assert result.persisted is True
    assert result.trial is not None
    assert result.trial.payload.evaluation_status == "completed"
    assert evaluator.saw_public_input is True
    assert not hasattr(evaluator.request, "private_reference")
    assert manifest.cases[0].private_reference not in result.trial.payload.to_dict().values()


@pytest.mark.parametrize("overlap", ["run", "evidence"])
def test_candidate_source_overlap_invalidates_held_out_trial(tmp_path: Path, overlap: str) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    source_run = new_evo_id("run")
    source_evidence = new_evo_id("evidence")
    manifest = _manifest(origin_run_ids=(source_run,), origin_evidence_refs=(source_evidence,))
    EvalCorpusRegistry(store).register(manifest)
    artifact_id = _artifact(
        store,
        source_run_ids=(source_run,) if overlap == "run" else (new_evo_id("run"),),
        evidence_refs=(source_evidence,) if overlap == "evidence" else (new_evo_id("evidence"),),
    )
    evaluator = _InspectingEvaluator(store)

    result = HeldOutEvalHarness(store).run(manifest, manifest.cases[0].case.case_id, _candidate(artifact_id), evaluator, seed=7)

    assert result.trial is not None
    assert result.trial.payload.evaluation_status == "invalid"
    assert result.trial.payload.task_outcome == "not_run"
    assert result.diagnostic is not None
    assert result.diagnostic.code == "held_out_contamination"
    assert evaluator.request is None


def test_accessing_private_reference_resource_invalidates_trial(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    manifest = _manifest()
    EvalCorpusRegistry(store).register(manifest)
    reference_hash = manifest.cases[0].private_reference_hash

    result = HeldOutEvalHarness(store).run(
        manifest,
        manifest.cases[0].case.case_id,
        _baseline(),
        _InspectingEvaluator(store, accessed_resource_hashes=(reference_hash,)),
        seed=7,
    )

    assert result.trial is not None
    assert result.trial.payload.evaluation_status == "invalid"
    assert "private reference" in " ".join(result.trial.payload.invalid_reasons)
    assert result.trial.payload.accessed_resource_hashes == (reference_hash,)


def _manifest(
    *,
    origin_run_ids: tuple[str, ...] = (),
    origin_evidence_refs: tuple[str, ...] = (),
) -> EvalCorpusManifest:
    public_input = "Repair a provider timeout while preserving verification."
    private = "Expected patch and assertions for the hidden timeout edge case."
    case = EvalCase(
        case_id="eval_case_provider_timeout",
        corpus_id="corpus_provider_regressions",
        corpus_version="v1",
        split="held_out",
        public_input=public_input,
        task_input_hash=hash_text(public_input),
        workspace_baseline_hash="b" * 64,
        environment_hash="c" * 64,
        origin_run_ids=origin_run_ids,
        origin_evidence_refs=origin_evidence_refs,
    )
    return EvalCorpusManifest(
        corpus_id=case.corpus_id,
        version=case.corpus_version,
        license_spdx="MIT",
        provenance="Repository-authored deterministic fixture.",
        cases=(EvalCorpusCase(case, private, private_reference_hash(private)),),
    )


def _baseline() -> EvalVariant:
    return EvalVariant(
        variant_id="eval_variant_baseline",
        kind="baseline",
        model_config_hash="d" * 64,
        strategy_hash="e" * 64,
    )


def _candidate(artifact_id: str) -> EvalVariant:
    return EvalVariant(
        variant_id="eval_variant_candidate",
        kind="candidate",
        model_config_hash="d" * 64,
        strategy_hash="f" * 64,
        artifact_id=artifact_id,
        artifact_version=1,
    )


def _artifact(
    store: EvoEventStore,
    *,
    source_run_ids: tuple[str, ...],
    evidence_refs: tuple[str, ...],
) -> str:
    artifact_id = new_evo_id("artifact")
    source_candidate_id = new_evo_id("candidate")
    store.append(
        EvoEvent(
            event_id=new_evo_id("event"),
            event_type="CandidateArtifactCreated",
            refs=EvoReferences(run_id=source_run_ids[0], artifact_id=artifact_id),
            payload=CandidateArtifactCreatedPayload(
                artifact_schema_version="v1",
                lineage_id=artifact_id,
                artifact_version=1,
                kind="plan_template",
                name="provider-held-out",
                description="Held-out provider template.",
                instructions="Apply verified provider steps.",
                inputs=("provider task",),
                outputs=("evidence",),
                dependencies=("pytest",),
                effects=("read",),
                triggers=("provider",),
                scope="project",
                applicability="Provider fixes.",
                risks=("Requires held-out validation.",),
                source_candidate_ids=(source_candidate_id,),
                support_candidate_ids=(source_candidate_id,),
                counterexample_candidate_ids=(),
                source_run_ids=source_run_ids,
                evidence_refs=evidence_refs,
                counterexamples=(),
                confidence=0.6,
                content_hash="a" * 64,
            ),
        )
    )
    return artifact_id


class _InspectingEvaluator:
    version = "deterministic-v1"

    def __init__(self, store: EvoEventStore, *, accessed_resource_hashes: tuple[str, ...] = ()) -> None:
        self._evidence = EvidenceAdapter(store)
        self.accessed_resource_hashes = accessed_resource_hashes
        self.request: EvalRunInput | None = None
        self.saw_public_input = False

    def evaluate(self, request: EvalRunInput) -> EvalObservation:
        self.request = request
        self.saw_public_input = bool(request.public_input)
        recorded = self._evidence.record(
            EvidenceInput(
                run_id=request.run_id,
                evidence_type="test",
                source="pytest",
                summary="held-out verification passed",
                verified=True,
                command="pytest -q",
                exit_code=0,
            )
        )
        assert recorded.persisted and recorded.evidence is not None
        return EvalObservation(
            task_outcome="task_success",
            verification_quality=1.0,
            evidence_refs=(recorded.evidence.evidence_id,),
            verification_commands=("pytest -q",),
            evidence_success=True,
            accessed_resource_hashes=self.accessed_resource_hashes,
        )
