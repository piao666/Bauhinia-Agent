from __future__ import annotations

from pathlib import Path

import pytest

from bauhinia_agent.evolution.candidate_artifacts import (
    CandidateArtifactDraft,
    CandidateArtifactError,
    CandidateArtifactKind,
    CandidateArtifactRegistry,
    CandidateEffectRisk,
    render_skill_markdown,
)
from bauhinia_agent.evolution.candidate_review import CandidateReview, CandidateReviewService
from bauhinia_agent.evolution.events import EvoEvent, EvoReferences, ExperienceCandidateCreatedPayload
from bauhinia_agent.evolution.identifiers import new_evo_id
from bauhinia_agent.evolution.store import EvoEventStore, EvoStoreError
from bauhinia_agent.skills.discovery import discover_project_skills
from bauhinia_agent.skills.loader import SkillLoader


@pytest.mark.parametrize("kind", list(CandidateArtifactKind))
def test_registry_persists_all_artifact_kinds_as_non_runtime_candidates(tmp_path: Path, kind: CandidateArtifactKind) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    candidate_id = _accepted_candidate(store, summary=f"Verified pattern for {kind.value}.")
    before = discover_project_skills(tmp_path)

    result = CandidateArtifactRegistry(store).create(_draft(kind, candidate_id))

    assert result.persisted is True
    assert result.artifact is not None
    assert result.artifact.payload.kind == kind.value
    assert result.artifact.payload.artifact_version == 1
    assert result.artifact.payload.lifecycle_state == "Candidate"
    assert result.artifact.payload.extensions["runtime_enabled"] is False
    assert result.artifact.payload.content_hash
    assert discover_project_skills(tmp_path) == before
    assert all(event.event_type not in {"PromotionChanged", "MemoryCreated"} for event in store.list_events())


def test_unknown_effect_fails_closed_and_minimal_export_excludes_sensitive_content(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    candidate_id = _accepted_candidate(store)
    draft = _draft(
        CandidateArtifactKind.TOOL_INVOCATION_POLICY,
        candidate_id,
        effects=("teleport",),
        instructions="Authorization: Bearer top-secret-value",
        dependencies=("private-token=top-secret-value",),
    )

    result = CandidateArtifactRegistry(store).create(draft)

    assert result.artifact is not None
    assert result.artifact.effect_risk is CandidateEffectRisk.HIGH
    metadata = result.artifact.minimal_metadata()
    assert metadata["effect_risk"] == "high"
    assert metadata["effects"] == ["teleport"]
    assert "instructions" not in metadata
    assert "dependencies" not in metadata
    serialized = (tmp_path / ".bauhinia-agent" / "evo" / "events.jsonl").read_text(encoding="utf-8")
    assert "top-secret-value" not in serialized


def test_manifest_tracks_immutable_versions_and_rejects_identity_changes(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    candidate_id = _accepted_candidate(store)
    registry = CandidateArtifactRegistry(store)
    first = registry.create(_draft(CandidateArtifactKind.PLAN_TEMPLATE, candidate_id)).artifact
    assert first is not None

    second = registry.create(
        _draft(
            CandidateArtifactKind.PLAN_TEMPLATE,
            candidate_id,
            description="A narrower verified provider plan.",
            supersedes_artifact_id=first.artifact_id,
        )
    ).artifact

    assert second is not None
    assert second.payload.artifact_version == 2
    assert second.payload.lineage_id == first.payload.lineage_id
    assert second.payload.supersedes_artifact_id == first.artifact_id
    assert registry.manifest().versions(first.payload.lineage_id) == (first, second)
    assert registry.manifest().latest(first.payload.lineage_id) == second
    with pytest.raises(CandidateArtifactError, match="latest lineage version"):
        registry.create(
            _draft(
                CandidateArtifactKind.PLAN_TEMPLATE,
                candidate_id,
                description="An invalid branch from version one.",
                supersedes_artifact_id=first.artifact_id,
            )
        )
    with pytest.raises(CandidateArtifactError, match="cannot change kind"):
        registry.create(
            _draft(
                CandidateArtifactKind.SKILL_DRAFT,
                candidate_id,
                supersedes_artifact_id=second.artifact_id,
            )
        )


def test_skill_draft_renders_existing_skill_format_only_in_memory(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    candidate_id = _accepted_candidate(store)
    artifact = (
        CandidateArtifactRegistry(store)
        .create(
            _draft(
                CandidateArtifactKind.SKILL_DRAFT,
                candidate_id,
                name="provider-review",
                description="Review provider changes.",
                instructions="# Provider Review\n\nStart by reading `docs/provider.md`.",
                triggers=("provider change", "供应商适配"),
            )
        )
        .artifact
    )
    assert artifact is not None

    content = render_skill_markdown(artifact)
    skill_root = tmp_path / ".agents" / "skills" / "provider-review"
    skill_root.mkdir(parents=True)
    (skill_root / "SKILL.md").write_text(content, encoding="utf-8")
    definition = discover_project_skills(tmp_path).skills[0]
    loaded = SkillLoader().load(definition)

    assert definition.name == "provider-review"
    assert definition.description == "Review provider changes."
    assert definition.triggers == ("provider change", "供应商适配")
    assert "# Provider Review" in loaded.content
    assert loaded.required_files == []


def test_registry_rejects_unreviewed_sources_and_reports_store_failure(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    unreviewed = _candidate(store)
    registry = CandidateArtifactRegistry(store)
    with pytest.raises(CandidateArtifactError, match="human acceptance"):
        registry.create(_draft(CandidateArtifactKind.MEMORY_RULE, unreviewed, name="verified-memory-rule"))

    accepted = _accepted_candidate(store)
    result = CandidateArtifactRegistry(_FailingStore(store.list_events())).create(_draft(CandidateArtifactKind.MEMORY_RULE, accepted, name="verified-memory-rule"))

    assert result.persisted is False
    assert result.artifact is None
    assert result.diagnostic is not None
    assert result.diagnostic.code == "artifact_recording_failed"


def _draft(
    kind: CandidateArtifactKind,
    candidate_id: str,
    *,
    name: str | None = None,
    description: str = "A verified provider workflow.",
    instructions: str = "Run focused provider tests and record evidence.",
    dependencies: tuple[str, ...] = ("pytest",),
    effects: tuple[str, ...] = ("execute",),
    triggers: tuple[str, ...] = ("provider change",),
    supersedes_artifact_id: str | None = None,
) -> CandidateArtifactDraft:
    return CandidateArtifactDraft(
        kind=kind,
        name=name or f"verified-{kind.value.replace('_', '-')}",
        description=description,
        instructions=instructions,
        inputs=("changed provider files",),
        outputs=("verification evidence",),
        dependencies=dependencies,
        effects=effects,
        scope="project",
        applicability="Provider adapter changes with deterministic tests.",
        risks=("Do not use when provider tests are unavailable.",),
        source_candidate_ids=(candidate_id,),
        confidence=0.6,
        triggers=triggers,
        supersedes_artifact_id=supersedes_artifact_id,
    )


def _accepted_candidate(store: EvoEventStore, *, summary: str = "Run focused tests after provider changes.") -> str:
    candidate_id = _candidate(store, summary=summary)
    result = CandidateReviewService(store).review(
        candidate_id,
        CandidateReview(decision="accept", reviewer="curator", reason="Evidence and scope are reviewable."),
    )
    assert result.persisted is True
    return candidate_id


def _candidate(store: EvoEventStore, *, summary: str = "Run focused tests after provider changes.") -> str:
    candidate_id = new_evo_id("candidate")
    run_id = new_evo_id("run")
    store.append(
        EvoEvent(
            event_id=new_evo_id("event"),
            event_type="ExperienceCandidateCreated",
            refs=EvoReferences(run_id=run_id, candidate_id=candidate_id),
            payload=ExperienceCandidateCreatedPayload(
                kind="plan_template",
                summary=summary,
                scope="project",
                applicability="Provider adapter changes.",
                confidence=0.4,
                source_event_ids=(new_evo_id("event"),),
                evidence_refs=(new_evo_id("evidence"),),
                source_run_ids=(run_id,),
            ),
        )
    )
    return candidate_id


class _FailingStore:
    def __init__(self, events: list[EvoEvent]) -> None:
        self._events = events

    def list_events(self) -> list[EvoEvent]:
        return self._events

    def append(self, event: EvoEvent) -> object:
        raise EvoStoreError("store offline")
