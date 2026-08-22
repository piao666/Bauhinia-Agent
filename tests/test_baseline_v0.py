from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = REPOSITORY_ROOT / "benchmarks" / "baseline_v0"


def _load_validator() -> ModuleType:
    path = BASELINE_ROOT / "validate_manifest.py"
    module = ModuleType("p0_baseline_validator")
    module.__file__ = str(path)
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module


def _copy_corpus(destination: Path) -> Path:
    copied_root = destination / "baseline_v0"
    shutil.copytree(
        BASELINE_ROOT,
        copied_root,
        ignore=shutil.ignore_patterns("runs", "__pycache__", "*.pyc"),
    )
    return copied_root


@pytest.fixture(scope="module")
def validator() -> ModuleType:
    return _load_validator()


@pytest.fixture(scope="module")
def baseline_run(validator: ModuleType) -> tuple[dict[str, object], dict[str, object]]:
    manifest = validator.load_manifest(BASELINE_ROOT)
    return manifest, validator.run_baseline(manifest, BASELINE_ROOT)


def test_tracked_baseline_is_complete_and_reproducible(
    validator: ModuleType,
    baseline_run: tuple[dict[str, object], dict[str, object]],
) -> None:
    manifest, summary = baseline_run

    assert validator.validate_corpus(BASELINE_ROOT) == []
    assert manifest["corpus_id"] == "bauhinia-evo-offline-baseline"
    assert manifest["corpus_version"] == "v0"
    assert manifest["case_count"] == 12
    assert summary["result"] == "pass"
    assert summary["case_count"] == 12
    assert summary["fixture_expectation_match_count"] == 12
    assert summary["fixture_observed_pass_count"] == 8
    assert summary["fixture_observed_fail_count"] == 4
    assert all(result["matched"] for result in summary["results"])
    assert all("expected_evidence_type" in result for result in summary["results"])
    assert all("evidence_type" not in result for result in summary["results"])


def test_lock_inventory_is_complete_without_self_reference(
    validator: ModuleType,
) -> None:
    lock = validator.load_lock(BASELINE_ROOT)
    locked_entries = {entry["path"]: entry["sha256"] for entry in lock["entries"]}
    inventory, inventory_errors = validator._corpus_inventory(BASELINE_ROOT)

    assert inventory_errors == []
    assert set(locked_entries) == inventory
    assert "corpus.lock.json" not in locked_entries
    assert "README.md" not in locked_entries
    assert not any(path.startswith("runs/") for path in locked_entries)
    assert lock["corpus_sha256"] == validator.aggregate_sha256(locked_entries)


@pytest.mark.parametrize(
    "relative_path",
    [
        "manifest.json",
        "validate_manifest.py",
        "workspaces/permission-write-denied/scenario.json",
    ],
)
def test_integrity_rejects_tampered_corpus_file(
    validator: ModuleType,
    tmp_path: Path,
    relative_path: str,
) -> None:
    copied_root = _copy_corpus(tmp_path)
    target = copied_root / relative_path
    target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    errors = validator.validate_corpus(copied_root)

    assert f"corpus sha256 mismatch: {relative_path}" in errors


@pytest.mark.parametrize(
    ("relative_path", "specific_error"),
    [
        (
            "workspaces/provider-timeout/scenario.json",
            "provider-timeout: missing scenario.json",
        ),
        ("verify_scenario.py", "missing required corpus file: verify_scenario.py"),
    ],
)
def test_integrity_and_structure_reject_missing_corpus_file(
    validator: ModuleType,
    tmp_path: Path,
    relative_path: str,
    specific_error: str,
) -> None:
    copied_root = _copy_corpus(tmp_path)
    (copied_root / relative_path).unlink()

    errors = validator.validate_corpus(copied_root)

    assert f"locked corpus file is missing: {relative_path}" in errors
    assert specific_error in errors


def test_report_contract_is_versioned_and_files_are_append_only(
    validator: ModuleType,
    baseline_run: tuple[dict[str, object], dict[str, object]],
    tmp_path: Path,
) -> None:
    manifest, summary = baseline_run
    lock = validator.load_lock(BASELINE_ROOT)
    fixed_repository = {
        "commit": "a" * 40,
        "branch": "main",
        "dirty": False,
    }
    fixed_environment = {
        "python_version": "3.12.10",
        "implementation": "CPython",
        "platform": "test-platform",
        "os_name": "nt",
        "byteorder": "little",
    }
    recorded_at = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)
    first_report = validator.build_report(
        manifest,
        lock,
        summary,
        root=BASELINE_ROOT,
        run_id="p0-baseline-v0-contract-001",
        recorded_at=recorded_at,
        repository=fixed_repository,
        environment=fixed_environment,
    )
    report_dir = tmp_path / "reports"
    first_path = validator.write_report(first_report, report_dir)
    first_bytes = first_path.read_bytes()

    second_report = validator.build_report(
        manifest,
        lock,
        summary,
        root=BASELINE_ROOT,
        run_id="p0-baseline-v0-contract-002",
        recorded_at=recorded_at,
        repository=fixed_repository,
        environment=fixed_environment,
    )
    second_path = validator.write_report(second_report, report_dir)

    assert first_path != second_path
    assert first_path.read_bytes() == first_bytes
    assert sorted(path.name for path in report_dir.glob("*.json")) == [
        "p0-baseline-v0-contract-001.json",
        "p0-baseline-v0-contract-002.json",
    ]
    with pytest.raises(FileExistsError):
        validator.write_report(first_report, report_dir)
    assert first_path.read_bytes() == first_bytes

    stored = json.loads(first_path.read_text(encoding="utf-8"))
    assert stored["schema_version"] == 3
    assert stored["report_type"] == "p0_offline_synthetic_fixture_health"
    assert stored["corpus"] == {
        "id": "bauhinia-evo-offline-baseline",
        "version": "v0",
        "sha256": lock["corpus_sha256"],
        "manifest_sha256": validator.normalized_sha256(BASELINE_ROOT / "manifest.json"),
        "case_count": 12,
        "license": "MIT",
    }
    assert stored["repository"] == fixed_repository
    assert stored["environment"] == fixed_environment
    assert stored["verifier"]["version"] == "1.0.0"
    assert len(stored["case_results"]) == 12
    assert {metric["name"] for metric in stored["metrics"]} == {
        "fixture_observed_pass_rate",
        "fixture_expectation_match_rate",
        "tool_failure_fixture_share",
        "verifier_wall_time",
    }
    assert stored["claim_boundary"]["p0_005_satisfied"] is False
    assert set(stored["claim_boundary"]["unmeasured_agent_metrics"]) == {
        "agent_task_success_rate",
        "agent_verification_pass_rate",
        "agent_tool_failure_rate",
        "model_token_count",
        "human_interventions",
        "dangerous_actions",
        "run_to_run_reproducibility",
    }
    assert all("expected_evidence_type" in result for result in stored["case_results"])
    assert all("evidence_type" not in result for result in stored["case_results"])


def test_report_rejects_unsafe_or_duplicate_run_id(
    validator: ModuleType,
    baseline_run: tuple[dict[str, object], dict[str, object]],
    tmp_path: Path,
) -> None:
    manifest, summary = baseline_run
    lock = validator.load_lock(BASELINE_ROOT)
    report = validator.build_report(
        manifest,
        lock,
        summary,
        root=BASELINE_ROOT,
        run_id="safe-run-id",
    )
    validator.write_report(report, tmp_path)

    with pytest.raises(FileExistsError):
        validator.write_report(report, tmp_path)
    report["run_id"] = "../escape"
    with pytest.raises(ValueError, match="not safe"):
        validator.write_report(report, tmp_path)


@pytest.mark.parametrize(
    "relative_path",
    [
        "unexpected.txt",
        "__pycache__/validator.cpython-312.pyc",
        "workspaces/new-function-slug/__pycache__/app.cpython-312.pyc",
        "workspaces/new-function-slug/app.pyc",
    ],
)
def test_inventory_rejects_extra_root_and_bytecode_artifacts(
    validator: ModuleType,
    tmp_path: Path,
    relative_path: str,
) -> None:
    copied_root = _copy_corpus(tmp_path)
    artifact = copied_root / relative_path
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"not trusted")

    errors = validator.validate_corpus(copied_root)

    assert errors
    assert any(marker in error for error in errors for marker in ("unexpected corpus root entry", "bytecode artifact is not allowed"))


def test_inventory_rejects_reparse_points(
    validator: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    copied_root = _copy_corpus(tmp_path)
    target = copied_root / "workspaces" / "new-function-slug" / "app.py"
    original = validator._path_is_reparse
    monkeypatch.setattr(
        validator,
        "_path_is_reparse",
        lambda path: Path(path) == target or original(Path(path)),
    )

    errors = validator.validate_corpus(copied_root)

    assert "reparse point is not allowed in corpus: workspaces/new-function-slug/app.py" in errors


def test_windows_reparse_attribute_is_detected(validator: ModuleType) -> None:
    stat_result = SimpleNamespace(st_file_attributes=0x400)

    assert validator._stat_is_reparse(stat_result) is True


def test_lock_paths_cannot_resolve_outside_corpus(
    validator: ModuleType,
    tmp_path: Path,
) -> None:
    copied_root = _copy_corpus(tmp_path)
    lock = validator.load_lock(copied_root)
    lock["entries"][0]["path"] = "../manifest.json"

    errors = validator.validate_integrity(copied_root, lock)

    assert "corpus lock entry 0 has an unsafe path" in errors


def test_baseline_executes_verified_private_snapshot_without_toctou(
    validator: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    copied_root = _copy_corpus(tmp_path)
    manifest = validator.load_manifest(copied_root)
    source_module = copied_root / "workspaces" / "new-function-slug" / "app.py"
    real_run = subprocess.run
    verifier_commands: list[list[str]] = []

    def mutate_source_then_run(arguments: list[str], **kwargs: object):
        verifier_commands.append([str(value) for value in arguments])
        if len(verifier_commands) == 1:
            source_module.write_text(
                "def slugify(text: str) -> str:\n    return 'hello-world'\n",
                encoding="utf-8",
            )
        return real_run(arguments, **kwargs)

    monkeypatch.setattr(validator.subprocess, "run", mutate_source_then_run)

    summary = validator.run_baseline(manifest, copied_root)

    first = summary["results"][0]
    assert first["id"] == "new-function-slug"
    assert first["observed"] == "fail"
    assert first["matched"] is True
    assert all(command[1:4] == ["-I", "-S", "-B"] for command in verifier_commands)
    assert all(str(copied_root) not in command[-1] for command in verifier_commands)


def test_same_run_id_concurrency_is_exclusive_and_cleans_temporaries(
    validator: ModuleType,
    baseline_run: tuple[dict[str, object], dict[str, object]],
    tmp_path: Path,
) -> None:
    manifest, summary = baseline_run
    report = validator.build_report(
        manifest,
        validator.load_lock(BASELINE_ROOT),
        summary,
        root=BASELINE_ROOT,
        run_id="concurrent-run-id",
    )
    report_dir = tmp_path / "reports"

    def publish() -> str:
        try:
            validator.write_report(report, report_dir)
        except FileExistsError:
            return "exists"
        return "written"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: publish(), range(2)))

    assert sorted(outcomes) == ["exists", "written"]
    assert [path.name for path in report_dir.glob("*.json")] == ["concurrent-run-id.json"]
    assert not list(report_dir.glob("*.tmp"))


def test_report_publish_failure_removes_temporary_file(
    validator: ModuleType,
    baseline_run: tuple[dict[str, object], dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest, summary = baseline_run
    report = validator.build_report(
        manifest,
        validator.load_lock(BASELINE_ROOT),
        summary,
        root=BASELINE_ROOT,
        run_id="publish-failure",
    )
    report_dir = tmp_path / "reports"
    monkeypatch.setattr(validator.os, "link", lambda *_args: (_ for _ in ()).throw(OSError("publish failed")))

    with pytest.raises(OSError, match="publish failed"):
        validator.write_report(report, report_dir)

    assert not list(report_dir.iterdir())


def test_default_report_directory_must_remain_inside_corpus_and_not_reparse(
    validator: ModuleType,
    baseline_run: tuple[dict[str, object], dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest, summary = baseline_run
    report = validator.build_report(
        manifest,
        validator.load_lock(BASELINE_ROOT),
        summary,
        root=BASELINE_ROOT,
        run_id="safe-default",
    )
    copied_root = _copy_corpus(tmp_path / "copy")

    with pytest.raises(ValueError, match="inside the corpus root"):
        validator.write_report(
            report,
            tmp_path / "outside",
            containment_root=copied_root,
        )

    report_dir = copied_root / "runs"
    report_dir.mkdir()
    original = validator._path_is_reparse
    monkeypatch.setattr(
        validator,
        "_path_is_reparse",
        lambda path: Path(path) == report_dir or original(Path(path)),
    )
    with pytest.raises(ValueError, match="reparse"):
        validator.write_report(
            report,
            report_dir,
            containment_root=copied_root,
        )
