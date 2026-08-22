"""Validate and run the tracked P0 synthetic baseline corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "manifest.json"
LOCK_PATH = ROOT / "corpus.lock.json"
VERIFY_PATH = ROOT / "verify_scenario.py"
DEFAULT_REPORT_DIR = ROOT / "runs"
VERIFIER_TIMEOUT_SECONDS = 10.0
REQUIRED_CATEGORIES = {
    "new_function",
    "small_defect",
    "cross_file_change",
    "tool_failure",
    "permission_denied",
    "context_conflict",
    "recovery",
    "cancellation",
    "timeout",
    "mcp_schema",
    "redaction",
    "session_resume",
}
LOCKED_CORE_FILES = {
    "manifest.json",
    "validate_manifest.py",
    "verify_scenario.py",
}
ALLOWED_UNLOCKED_ROOT_ENTRIES = {
    "README.md",
    "corpus.lock.json",
    "runs",
    "workspaces",
}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]+")
_MODULE_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\.py")
_FUNCTION_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def load_manifest(root: Path = ROOT) -> dict[str, Any]:
    return _read_json(root / "manifest.json")


def load_lock(root: Path = ROOT) -> dict[str, Any]:
    return _read_json(root / "corpus.lock.json")


def normalized_sha256(path: Path) -> str:
    """Hash UTF-8 text with normalized LF endings for cross-platform clones."""

    text = path.read_text(encoding="utf-8")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def aggregate_sha256(entries: Mapping[str, str]) -> str:
    material = "".join(f"{path}\0{digest}\n" for path, digest in sorted(entries.items()))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _is_safe_canonical_path(value: object) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        return False
    if ":" in candidate.parts[0]:
        return False
    return candidate.as_posix() == value


def _stat_is_reparse(stat_result: object) -> bool:
    """Return whether a stat result identifies a Windows reparse point."""

    attributes = getattr(stat_result, "st_file_attributes", 0)
    return bool(attributes & 0x400)


def _path_is_reparse(path: Path) -> bool:
    """Detect symlinks and junction-like reparse points without following them."""

    try:
        return path.is_symlink() or _stat_is_reparse(path.lstat())
    except OSError:
        return False


def _corpus_inventory(root: Path) -> tuple[set[str], list[str]]:
    inventory: set[str] = set()
    errors: list[str] = []
    if _path_is_reparse(root):
        return inventory, ["corpus root must not be a reparse point"]
    expected_root_entries = LOCKED_CORE_FILES | ALLOWED_UNLOCKED_ROOT_ENTRIES
    try:
        root_entries = tuple(root.iterdir())
    except OSError as error:
        return inventory, [f"cannot enumerate corpus root: {error}"]
    for path in root_entries:
        relative = path.relative_to(root).as_posix()
        if path.name == "__pycache__" or path.suffix.lower() == ".pyc":
            errors.append(f"bytecode artifact is not allowed in corpus: {relative}")
            continue
        if path.name not in expected_root_entries:
            errors.append(f"unexpected corpus root entry: {relative}")
            continue
        if _path_is_reparse(path):
            errors.append(f"reparse point is not allowed in corpus: {relative}")
    for relative_path in LOCKED_CORE_FILES:
        path = root / relative_path
        if path.is_file() and not _path_is_reparse(path):
            inventory.add(relative_path)
        else:
            errors.append(f"missing required corpus file: {relative_path}")
    workspaces = root / "workspaces"
    if not workspaces.is_dir() or _path_is_reparse(workspaces):
        return inventory, errors + ["missing workspaces directory"]
    pending = [workspaces]
    while pending:
        directory = pending.pop()
        try:
            children = tuple(directory.iterdir())
        except OSError as error:
            errors.append(f"cannot enumerate corpus directory " f"{directory.relative_to(root).as_posix()}: {error}")
            continue
        for path in children:
            relative = path.relative_to(root).as_posix()
            if path.name == "__pycache__" or path.suffix.lower() == ".pyc":
                errors.append(f"bytecode artifact is not allowed in corpus: {relative}")
                continue
            if _path_is_reparse(path):
                errors.append(f"reparse point is not allowed in corpus: {relative}")
                continue
            if path.is_dir():
                pending.append(path)
            elif path.is_file():
                inventory.add(relative)
            else:
                errors.append(f"unsupported corpus entry: {relative}")
    return inventory, errors


def validate_integrity(root: Path, lock: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if lock.get("schema_version") != 1:
        errors.append("corpus lock schema_version must be 1")
    if lock.get("hash_algorithm") != "sha256-normalized-lf":
        errors.append("corpus lock hash_algorithm must be sha256-normalized-lf")
    raw_entries = lock.get("entries")
    if not isinstance(raw_entries, list):
        return errors + ["corpus lock entries must be a list"]

    entries: dict[str, str] = {}
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict):
            errors.append(f"corpus lock entry {index} must be an object")
            continue
        relative_path = raw_entry.get("path")
        digest = raw_entry.get("sha256")
        if not _is_safe_canonical_path(relative_path):
            errors.append(f"corpus lock entry {index} has an unsafe path")
            continue
        if not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest):
            errors.append(f"corpus lock entry {index} has an invalid sha256")
            continue
        if relative_path in entries:
            errors.append(f"duplicate corpus lock path: {relative_path}")
            continue
        entries[relative_path] = digest

    inventory, inventory_errors = _corpus_inventory(root)
    errors.extend(inventory_errors)
    missing_lock_entries = sorted(inventory - entries.keys())
    unexpected_lock_entries = sorted(entries.keys() - inventory)
    errors.extend(f"unlocked corpus file: {relative_path}" for relative_path in missing_lock_entries)
    errors.extend(f"locked corpus file is missing: {relative_path}" for relative_path in unexpected_lock_entries)

    for relative_path, expected_digest in sorted(entries.items()):
        path = root / relative_path
        if not path.is_file() or path.is_symlink():
            continue
        try:
            actual_digest = normalized_sha256(path)
        except (OSError, UnicodeError) as error:
            errors.append(f"cannot hash corpus file {relative_path}: {error}")
            continue
        if actual_digest != expected_digest:
            errors.append(f"corpus sha256 mismatch: {relative_path}")

    expected_aggregate = lock.get("corpus_sha256")
    actual_aggregate = aggregate_sha256(entries)
    if not isinstance(expected_aggregate, str) or not _SHA256_PATTERN.fullmatch(expected_aggregate):
        errors.append("corpus lock corpus_sha256 is invalid")
    elif expected_aggregate != actual_aggregate:
        errors.append("corpus lock aggregate sha256 mismatch")
    return errors


def _validate_scenario(case: dict[str, Any], scenario_path: Path) -> list[str]:
    case_id = str(case.get("id", "<unknown>"))
    try:
        scenario = _read_json(scenario_path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        return [f"{case_id}: invalid scenario.json: {error}"]
    errors: list[str] = []
    if scenario.get("case_id") != case.get("id"):
        errors.append(f"{case_id}: scenario case_id does not match manifest")
    kind = scenario.get("kind")
    if kind not in {"python_call", "observed_value"}:
        errors.append(f"{case_id}: unsupported scenario kind")
    if "expected" not in scenario:
        errors.append(f"{case_id}: scenario is missing expected")
    if kind == "python_call":
        module = scenario.get("module")
        function = scenario.get("function")
        if not isinstance(module, str) or not _MODULE_PATTERN.fullmatch(module):
            errors.append(f"{case_id}: invalid scenario module")
        elif not (scenario_path.parent / module).is_file():
            errors.append(f"{case_id}: missing scenario module")
        if not isinstance(function, str) or not _FUNCTION_PATTERN.fullmatch(function):
            errors.append(f"{case_id}: invalid scenario function")
        if not isinstance(scenario.get("args", []), list):
            errors.append(f"{case_id}: scenario args must be an array")
        if not isinstance(scenario.get("kwargs", {}), dict):
            errors.append(f"{case_id}: scenario kwargs must be an object")
    elif kind == "observed_value" and "observed" not in scenario:
        errors.append(f"{case_id}: observed_value is missing observed")
    return errors


def validate_structure(manifest: dict[str, Any], root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    required_manifest_values = {
        "schema_version": 2,
        "corpus_id": "bauhinia-evo-offline-baseline",
        "corpus_version": "v0",
        "verifier_version": "1.0.0",
        "license": "MIT",
        "immutable": True,
        "case_count": 12,
        "expected_initial_failures": 4,
    }
    for field, expected in required_manifest_values.items():
        if manifest.get(field) != expected:
            errors.append(f"manifest {field} must be {expected!r}")

    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != 12:
        return errors + ["exactly 12 cases are required"]
    if any(not isinstance(case, dict) for case in cases):
        return errors + ["every case must be an object"]

    ids = [case.get("id") for case in cases]
    if len(set(value for value in ids if isinstance(value, str))) != 12 or any(not isinstance(value, str) or not value for value in ids):
        errors.append("case ids must be 12 unique non-blank strings")
    categories = [case.get("category") for case in cases]
    if any(not isinstance(category, str) for category in categories) or set(category for category in categories if isinstance(category, str)) != REQUIRED_CATEGORIES:
        errors.append("case categories do not cover the required P0 scope")
    if sum(case.get("baseline_expected_outcome") == "fail" for case in cases) != 4:
        errors.append("exactly four stable initial failures are required")

    workspace_paths: set[str] = set()
    for case in cases:
        case_id = str(case.get("id", "<unknown>"))
        for field in (
            "task_input",
            "workspace_baseline",
            "acceptance_command",
            "expected_evidence_type",
            "baseline_expected_outcome",
        ):
            if not isinstance(case.get(field), str) or not case[field].strip():
                errors.append(f"{case_id}: missing {field}")
        outcome = case.get("baseline_expected_outcome")
        if outcome not in {"pass", "fail"}:
            errors.append(f"{case_id}: invalid baseline_expected_outcome")

        workspace = case.get("workspace_baseline")
        if not _is_safe_canonical_path(workspace) or not str(workspace).startswith("workspaces/"):
            errors.append(f"{case_id}: unsafe workspace_baseline")
            continue
        if workspace in workspace_paths:
            errors.append(f"{case_id}: duplicate workspace_baseline")
        workspace_paths.add(workspace)
        scenario_path = root / workspace / "scenario.json"
        if not scenario_path.is_file():
            errors.append(f"{case_id}: missing scenario.json")
            continue
        errors.extend(_validate_scenario(case, scenario_path))

        expected_command = "python benchmarks/baseline_v0/verify_scenario.py " f"benchmarks/baseline_v0/{workspace}/scenario.json"
        if case.get("acceptance_command") != expected_command:
            errors.append(f"{case_id}: acceptance_command is not canonical")
    return errors


def validate_corpus(root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    try:
        lock = load_lock(root)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        errors.append(f"invalid corpus.lock.json: {error}")
    else:
        errors.extend(validate_integrity(root, lock))
    try:
        manifest = load_manifest(root)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        errors.append(f"invalid manifest.json: {error}")
    else:
        errors.extend(validate_structure(manifest, root))
    return errors


def _safe_subprocess_environment() -> dict[str, str]:
    environment = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUTF8": "1",
    }
    for name in ("SYSTEMROOT", "WINDIR", "COMSPEC"):
        value = os.environ.get(name)
        if value:
            environment[name] = value
    return environment


def _verifier_payload(stdout: str) -> dict[str, Any] | None:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        return None
    try:
        value = json.loads(lines[-1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def run_baseline(manifest: dict[str, Any], root: Path = ROOT) -> dict[str, Any]:
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    corpus_errors = validate_corpus(root)
    if corpus_errors:
        raise ValueError("cannot execute an invalid corpus: " + "; ".join(corpus_errors))
    lock = load_lock(root)
    with tempfile.TemporaryDirectory(prefix="bauhinia-p0-baseline-") as temporary:
        snapshot_root = Path(temporary) / "baseline_v0"
        snapshot_root.mkdir()
        shutil.copy2(
            root / "corpus.lock.json",
            snapshot_root / "corpus.lock.json",
        )
        for entry in lock["entries"]:
            relative = str(entry["path"])
            source = root / relative
            destination = snapshot_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        snapshot_errors = validate_corpus(snapshot_root)
        if snapshot_errors:
            raise ValueError("private corpus snapshot failed integrity verification: " + "; ".join(snapshot_errors))
        verify_path = snapshot_root / "verify_scenario.py"
        for case in manifest["cases"]:
            scenario = snapshot_root / case["workspace_baseline"] / "scenario.json"
            diagnostic: str | None = None
            try:
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        "-S",
                        "-B",
                        str(verify_path),
                        str(scenario),
                    ],
                    cwd=snapshot_root,
                    env=_safe_subprocess_environment(),
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=VERIFIER_TIMEOUT_SECONDS,
                )
                return_code: int | None = completed.returncode
                payload = _verifier_payload(completed.stdout)
                observed = "pass" if completed.returncode == 0 else "fail"
                valid_payload = bool(payload and payload.get("case_id") == case["id"] and payload.get("result") == observed and payload.get("verifier_version") == manifest["verifier_version"])
                if not valid_payload:
                    diagnostic = "invalid verifier output contract"
            except subprocess.TimeoutExpired:
                return_code = None
                observed = "error"
                valid_payload = False
                diagnostic = "verifier timeout"
            except OSError as error:
                return_code = None
                observed = "error"
                valid_payload = False
                diagnostic = f"verifier execution failed: {type(error).__name__}"
            matched = observed == case["baseline_expected_outcome"] and valid_payload
            result = {
                "id": case["id"],
                "expected": case["baseline_expected_outcome"],
                "observed": observed,
                "matched": matched,
                "return_code": return_code,
                "expected_evidence_type": case["expected_evidence_type"],
            }
            if diagnostic is not None:
                result["diagnostic"] = diagnostic
            results.append(result)
    matched_count = sum(bool(result["matched"]) for result in results)
    observed_pass_count = sum(result["observed"] == "pass" for result in results)
    observed_fail_count = sum(result["observed"] == "fail" for result in results)
    return {
        "case_count": len(results),
        "fixture_expectation_match_count": matched_count,
        "fixture_observed_pass_count": observed_pass_count,
        "fixture_observed_fail_count": observed_fail_count,
        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        "results": results,
        "result": "pass" if matched_count == len(results) else "fail",
    }


def environment_snapshot() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "os_name": os.name,
        "byteorder": sys.byteorder,
    }


def repository_snapshot(start: Path = ROOT) -> dict[str, Any]:
    def run_git(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(start), *arguments],
            text=True,
            capture_output=True,
            check=False,
            timeout=5.0,
        )

    try:
        commit_result = run_git("rev-parse", "HEAD")
        branch_result = run_git("branch", "--show-current")
        status_result = run_git("status", "--porcelain", "--untracked-files=normal")
    except (OSError, subprocess.TimeoutExpired):
        return {"commit": "unavailable", "branch": "unavailable", "dirty": None}
    if commit_result.returncode != 0:
        return {"commit": "unavailable", "branch": "unavailable", "dirty": None}
    return {
        "commit": commit_result.stdout.strip(),
        "branch": branch_result.stdout.strip() or "detached",
        "dirty": bool(status_result.stdout.strip()) if status_result.returncode == 0 else None,
    }


def _new_run_id(corpus_version: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"p0-baseline-{corpus_version}-{timestamp}-{uuid.uuid4().hex[:8]}"


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def build_report(
    manifest: dict[str, Any],
    lock: dict[str, Any],
    summary: dict[str, Any],
    *,
    root: Path = ROOT,
    run_id: str | None = None,
    recorded_at: datetime | None = None,
    repository: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    report_run_id = run_id or _new_run_id(str(manifest["corpus_version"]))
    when = recorded_at or datetime.now(timezone.utc)
    if when.tzinfo is None or when.utcoffset() is None:
        raise ValueError("recorded_at must be timezone-aware")
    case_count = int(summary["case_count"])
    observed_pass_count = int(summary["fixture_observed_pass_count"])
    matched_count = int(summary["fixture_expectation_match_count"])
    tool_failure_count = sum(case.get("category") == "tool_failure" for case in manifest["cases"])
    return {
        "schema_version": 3,
        "report_type": "p0_offline_synthetic_fixture_health",
        "run_id": report_run_id,
        "recorded_at_utc": when.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z"),
        "result": summary["result"],
        "scope": "offline synthetic corpus health baseline",
        "corpus": {
            "id": manifest["corpus_id"],
            "version": manifest["corpus_version"],
            "sha256": lock["corpus_sha256"],
            "manifest_sha256": normalized_sha256(root / "manifest.json"),
            "case_count": manifest["case_count"],
            "license": manifest["license"],
        },
        "repository": dict(repository or repository_snapshot(root)),
        "environment": dict(environment or environment_snapshot()),
        "verifier": {
            "version": manifest["verifier_version"],
            "script_sha256": normalized_sha256(root / "verify_scenario.py"),
            "timeout_seconds": VERIFIER_TIMEOUT_SECONDS,
        },
        "executor": {
            "command": "python benchmarks/baseline_v0/validate_manifest.py --run-baseline",
            "model": None,
            "network_requested": False,
            "network_isolation": "not_enforced_by_fixture_runner",
            "credentials_requested": False,
            "limitation": ("Deterministic fixture-health baseline; not an external-model, " "held-out, or promoted-candidate evaluation."),
        },
        "claim_boundary": {
            "p0_005_satisfied": False,
            "unmeasured_agent_metrics": [
                "agent_task_success_rate",
                "agent_verification_pass_rate",
                "agent_tool_failure_rate",
                "model_token_count",
                "human_interventions",
                "dangerous_actions",
                "run_to_run_reproducibility",
            ],
        },
        "metrics": [
            {
                "name": "fixture_observed_pass_rate",
                "value": _ratio(observed_pass_count, case_count),
                "unit": "ratio",
                "numerator": observed_pass_count,
                "denominator": case_count,
            },
            {
                "name": "fixture_expectation_match_rate",
                "value": _ratio(matched_count, case_count),
                "unit": "ratio",
                "numerator": matched_count,
                "denominator": case_count,
            },
            {
                "name": "tool_failure_fixture_share",
                "value": _ratio(tool_failure_count, case_count),
                "unit": "ratio",
                "numerator": tool_failure_count,
                "denominator": case_count,
            },
            {
                "name": "verifier_wall_time",
                "value": summary["duration_ms"],
                "unit": "milliseconds",
            },
        ],
        "failure_breakdown": {
            "intentional_initial_failures": manifest["expected_initial_failures"],
            "tool_failure_cases": tool_failure_count,
            "permission_denied_cases": sum(case.get("category") == "permission_denied" for case in manifest["cases"]),
            "context_conflict_cases": sum(case.get("category") == "context_conflict" for case in manifest["cases"]),
            "recovery_cases": sum(case.get("category") == "recovery" for case in manifest["cases"]),
            "cancellation_cases": sum(case.get("category") == "cancellation" for case in manifest["cases"]),
            "timeout_cases": sum(case.get("category") == "timeout" for case in manifest["cases"]),
        },
        "case_results": summary["results"],
    }


def write_report(
    report: Mapping[str, Any],
    report_dir: Path,
    *,
    containment_root: Path | None = None,
) -> Path:
    run_id = report.get("run_id")
    if not isinstance(run_id, str) or not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("report run_id is not safe for a file name")
    report_dir = Path(report_dir)
    if containment_root is not None:
        containment = Path(containment_root).resolve()
        resolved_report_dir = report_dir.resolve()
        if resolved_report_dir != containment and containment not in resolved_report_dir.parents:
            raise ValueError("report directory must remain inside the corpus root")
        relative = resolved_report_dir.relative_to(containment)
        current = containment
        for part in relative.parts:
            current = current / part
            if current.exists() and _path_is_reparse(current):
                raise ValueError("report directory must not traverse a reparse point")
        report_dir = resolved_report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    if _path_is_reparse(report_dir):
        raise ValueError("report directory must not be a reparse point")
    report_path = report_dir / f"{run_id}.json"
    temporary_path = report_dir / f".{run_id}.{uuid.uuid4().hex}.tmp"
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        with temporary_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # A same-directory hard link publishes only the complete file and fails
        # rather than replacing an existing report with the same Run ID.
        os.link(temporary_path, report_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-baseline", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args(argv)
    if args.no_report and args.report_dir is not None:
        parser.error("--no-report cannot be combined with --report-dir")

    errors = validate_corpus(ROOT)
    if errors:
        print(
            json.dumps(
                {"result": "invalid", "errors": errors},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 1
    manifest = load_manifest(ROOT)
    lock = load_lock(ROOT)
    if not args.run_baseline:
        print(
            json.dumps(
                {
                    "result": "valid",
                    "case_count": len(manifest["cases"]),
                    "corpus_id": manifest["corpus_id"],
                    "corpus_version": manifest["corpus_version"],
                    "corpus_sha256": lock["corpus_sha256"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    summary = run_baseline(manifest, ROOT)
    output = dict(summary)
    if not args.no_report:
        report = build_report(manifest, lock, summary, root=ROOT)
        report_path = write_report(
            report,
            args.report_dir or DEFAULT_REPORT_DIR,
            containment_root=ROOT if args.report_dir is None else None,
        )
        output["report_path"] = str(report_path)
        output["run_id"] = report["run_id"]
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0 if summary["result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
