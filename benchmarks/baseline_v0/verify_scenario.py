"""Deterministic verifier for one public synthetic P0 workspace."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

VERIFIER_VERSION = "1.0.0"
_MODULE_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\.py")
_FUNCTION_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class ScenarioValidationError(ValueError):
    """Raised when a scenario does not match the public verifier contract."""


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("p0_baseline_case", path)
    if spec is None or spec.loader is None:
        raise ScenarioValidationError(f"cannot load module: {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def _python_call(scenario: dict[str, Any], scenario_path: Path) -> Any:
    module_name = scenario.get("module")
    function_name = scenario.get("function")
    arguments = scenario.get("args", [])
    keyword_arguments = scenario.get("kwargs", {})
    if not isinstance(module_name, str) or not _MODULE_PATTERN.fullmatch(module_name):
        raise ScenarioValidationError("module must be a direct Python file name")
    if not isinstance(function_name, str) or not _FUNCTION_PATTERN.fullmatch(function_name):
        raise ScenarioValidationError("function must be a public Python identifier")
    if not isinstance(arguments, list):
        raise ScenarioValidationError("args must be a JSON array")
    if not isinstance(keyword_arguments, dict):
        raise ScenarioValidationError("kwargs must be a JSON object")
    module_path = scenario_path.parent / module_name
    if not module_path.is_file():
        raise ScenarioValidationError(f"missing module: {module_name}")
    module = _load_module(module_path)
    function = getattr(module, function_name)
    if not callable(function):
        raise ScenarioValidationError(f"not callable: {function_name}")
    return function(*arguments, **keyword_arguments)


def verify(scenario_path: Path) -> tuple[bool, dict[str, Any]]:
    try:
        scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ScenarioValidationError(f"cannot read scenario: {error}") from error
    if not isinstance(scenario, dict):
        raise ScenarioValidationError("scenario must be a JSON object")
    case_id = scenario.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise ScenarioValidationError("case_id must be a non-blank string")

    kind = scenario.get("kind")
    try:
        if kind == "python_call":
            observed = _python_call(scenario, scenario_path)
        elif kind == "observed_value":
            if "observed" not in scenario:
                raise ScenarioValidationError("observed_value requires observed")
            observed = scenario["observed"]
        else:
            raise ScenarioValidationError(f"unsupported scenario kind: {kind}")
    except ScenarioValidationError:
        raise
    except Exception as error:  # Fixture code is intentionally allowed to be broken.
        observed = {"error_type": type(error).__name__, "message": str(error)}

    if "expected" not in scenario:
        raise ScenarioValidationError("scenario requires expected")
    expected = scenario["expected"]
    passed = observed == expected
    return passed, {
        "case_id": case_id,
        "expected": expected,
        "observed": observed,
        "result": "pass" if passed else "fail",
        "verifier_version": VERIFIER_VERSION,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", type=Path)
    args = parser.parse_args(argv)
    try:
        passed, summary = verify(args.scenario)
    except ScenarioValidationError as error:
        summary = {
            "case_id": args.scenario.parent.name,
            "error": str(error),
            "result": "error",
            "verifier_version": VERIFIER_VERSION,
        }
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
