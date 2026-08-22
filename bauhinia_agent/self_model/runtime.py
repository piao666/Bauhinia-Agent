"""Runtime bridge from evidence-backed profiles to conservative planning advice.

The bridge is intentionally advisory.  It can publish the exact profile snapshot
consumed by one Agent Run and render a system-only planning note, but it has no
tool, network, filesystem mutation, or permission-management port.
"""

from __future__ import annotations

import os
import platform
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Mapping

from bauhinia_agent.context.identity import stable_json_hash
from bauhinia_agent.evolution import EvoEventStore, require_evo_id
from bauhinia_agent.self_model.models import (
    ProfileSelector,
    RepositoryScale,
    SelfModelProfile,
    TaskClassification,
)
from bauhinia_agent.self_model.policy import PolicySuggestion, PolicySuggestionEngine
from bauhinia_agent.self_model.service import SelfModelService

RUNTIME_EVALUATOR_VERSION = "agent-loop-outcome-v1"
_SKIPPED_DIRECTORIES = frozenset(
    {
        ".bauhinia-agent",
        ".git",
        ".hg",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "venv",
    }
)
_LANGUAGE_SUFFIXES = {
    ".c": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cs": "csharp",
    ".go": "go",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "javascript",
    ".kt": "kotlin",
    ".php": "php",
    ".py": "python",
    ".rb": "ruby",
    ".rs": "rust",
    ".swift": "swift",
    ".ts": "typescript",
    ".tsx": "typescript",
}


@dataclass(frozen=True, slots=True)
class SelfModelRuntimeDiagnostic:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class SelfModelPlanningSnapshot:
    """Exact profile and suggestions selected before the current Run executes."""

    enabled: bool
    runtime_scope: str
    classification: TaskClassification | None = None
    profile: SelfModelProfile | None = None
    suggestions: tuple[PolicySuggestion, ...] = ()
    diagnostic: SelfModelRuntimeDiagnostic | None = None

    @property
    def profile_event_id(self) -> str | None:
        return None if self.profile is None else self.profile.published_event_id


@dataclass(frozen=True, slots=True)
class SelfModelObservationReceipt:
    recorded: bool
    observation_event_id: str | None = None
    diagnostic: SelfModelRuntimeDiagnostic | None = None


@dataclass(frozen=True, slots=True)
class RuntimeTaskClassifier:
    """Deterministic task dimensions used by the real Agent runtime."""

    project_id: str
    environment_hash: str
    project_language: str = "unknown"
    repository_scale: RepositoryScale = "unknown"
    evaluator_version: str = RUNTIME_EVALUATOR_VERSION

    def __post_init__(self) -> None:
        require_evo_id(self.project_id, field="project_id")
        if len(self.environment_hash) != 64:
            raise ValueError("environment_hash must be a SHA-256 digest")

    @classmethod
    def from_project(cls, project_root: str | Path) -> "RuntimeTaskClassifier":
        root = Path(project_root).resolve()
        language, scale = _inspect_repository(root)
        project_id = f"project_{stable_json_hash({'root': os.path.normcase(str(root))}, length=16)}"
        environment_hash = stable_json_hash(
            {
                "implementation": platform.python_implementation(),
                "machine": platform.machine().lower(),
                "os": os.name,
                "platform": platform.system().lower(),
                "python": [sys.version_info.major, sys.version_info.minor],
                "runtime_contract": RUNTIME_EVALUATOR_VERSION,
            },
            length=64,
        )
        return cls(
            project_id=project_id,
            environment_hash=environment_hash,
            project_language=language,
            repository_scale=scale,
        )

    def classify(
        self,
        task_text: str,
        *,
        provider_name: str,
        provider_model: str,
        request_options: Mapping[str, object] | None = None,
    ) -> TaskClassification:
        normalized = task_text.casefold()
        return TaskClassification(
            project_id=self.project_id,
            model_config_hash=stable_json_hash(
                {
                    "provider": provider_name,
                    "model": provider_model,
                    "request_options": dict(request_options or {}),
                },
                length=64,
            ),
            evaluator_version=self.evaluator_version,
            environment_hash=self.environment_hash,
            language=_language_for_task(normalized, fallback=self.project_language),
            repository_scale=self.repository_scale,
            task_type=_task_type(normalized),
            tool_category=_tool_category(normalized),
            risk_level=_risk_level(normalized),
        )


class SelfModelRuntime:
    """Project-scoped runtime switch and best-effort planning consumer."""

    def __init__(
        self,
        *,
        service: SelfModelService,
        classifier: RuntimeTaskClassifier,
        policy: PolicySuggestionEngine | None = None,
        enabled: bool = True,
    ) -> None:
        self._service = service
        self._classifier = classifier
        self._policy = policy or PolicySuggestionEngine(enabled=enabled)
        self._enabled = bool(enabled)
        self._policy.set_enabled(self._enabled)
        self._lock = RLock()
        self._latest_snapshot: SelfModelPlanningSnapshot | None = None
        self._latest_receipt: SelfModelObservationReceipt | None = None

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    @property
    def runtime_scope(self) -> str:
        return f"process:project:{self._classifier.project_id}"

    @property
    def latest_snapshot(self) -> SelfModelPlanningSnapshot | None:
        with self._lock:
            return self._latest_snapshot

    @property
    def latest_receipt(self) -> SelfModelObservationReceipt | None:
        with self._lock:
            return self._latest_receipt

    def set_enabled(self, enabled: bool) -> None:
        """Toggle planning injection and observation for this project process."""

        with self._lock:
            self._enabled = bool(enabled)
            self._policy.set_enabled(self._enabled)

    def prepare_task(
        self,
        task_text: str,
        *,
        provider_name: str,
        provider_model: str,
        request_options: Mapping[str, object] | None = None,
        run_id: str | None = None,
    ) -> SelfModelPlanningSnapshot:
        """Build at most one pre-execution profile for a caller-controlled turn."""

        if not self.enabled:
            return self._remember_snapshot(
                SelfModelPlanningSnapshot(
                    enabled=False,
                    runtime_scope=self.runtime_scope,
                    diagnostic=SelfModelRuntimeDiagnostic(
                        "self_model_disabled",
                        "runtime Self Model consumption is disabled for this project process",
                    ),
                )
            )
        try:
            classification = self._classifier.classify(
                task_text,
                provider_name=provider_name,
                provider_model=provider_model,
                request_options=request_options,
            )
            if run_id is None:
                return self._remember_snapshot(
                    SelfModelPlanningSnapshot(
                        enabled=True,
                        runtime_scope=self.runtime_scope,
                        classification=classification,
                        diagnostic=SelfModelRuntimeDiagnostic(
                            "self_model_run_unavailable",
                            "no active Evo Run is available to anchor a planning profile",
                        ),
                    )
                )
            selector = ProfileSelector(**classification.to_dict())
            diagnostic: SelfModelRuntimeDiagnostic | None = None
            published = self._service.publish_profile(selector, run_id=run_id)
            profile = published.profile
            if published.diagnostic is not None:
                diagnostic = SelfModelRuntimeDiagnostic(
                    published.diagnostic.code,
                    published.diagnostic.message,
                )
            if profile is None:
                raise RuntimeError("Self Model profile snapshot was not persisted")
            result = self._policy.suggest(profile)
            snapshot = SelfModelPlanningSnapshot(
                enabled=True,
                runtime_scope=self.runtime_scope,
                classification=classification,
                profile=profile,
                suggestions=result.suggestions,
                diagnostic=diagnostic,
            )
        except Exception as error:  # noqa: BLE001 - advisory failure cannot change the Agent result
            snapshot = SelfModelPlanningSnapshot(
                enabled=True,
                runtime_scope=self.runtime_scope,
                diagnostic=SelfModelRuntimeDiagnostic(
                    "self_model_prepare_failed",
                    f"{type(error).__name__}: {error}",
                ),
            )
        return self._remember_snapshot(snapshot)

    def advisory_for(self, snapshot: SelfModelPlanningSnapshot | None) -> str | None:
        """Return a system-only note while the project switch remains enabled."""

        if not self.enabled or snapshot is None or not snapshot.enabled or snapshot.profile is None:
            return None
        return render_system_advisory(snapshot)

    def record_completed_outcome(
        self,
        snapshot: SelfModelPlanningSnapshot | None,
        *,
        outcome_event_id: str | None,
    ) -> SelfModelObservationReceipt:
        """Record only a completed, evidence-backed Outcome selected by the observer."""

        if not self.enabled:
            return self._remember_receipt(
                SelfModelObservationReceipt(
                    False,
                    diagnostic=SelfModelRuntimeDiagnostic(
                        "self_model_disabled",
                        "runtime Self Model observation is disabled for this project process",
                    ),
                )
            )
        if snapshot is None or not snapshot.enabled or snapshot.classification is None:
            return self._remember_receipt(
                SelfModelObservationReceipt(
                    False,
                    diagnostic=SelfModelRuntimeDiagnostic(
                        "self_model_snapshot_unavailable",
                        "no pre-execution Self Model classification is available",
                    ),
                )
            )
        if outcome_event_id is None:
            return self._remember_receipt(
                SelfModelObservationReceipt(
                    False,
                    diagnostic=SelfModelRuntimeDiagnostic(
                        "self_model_outcome_unavailable",
                        "the completed Run produced no evidence-backed Outcome event",
                    ),
                )
            )
        try:
            recorded = self._service.record_observation(
                snapshot.classification,
                source_event_id=outcome_event_id,
            )
            if recorded.observation is not None:
                receipt = SelfModelObservationReceipt(
                    recorded=recorded.persisted,
                    observation_event_id=recorded.observation.event_id,
                    diagnostic=(
                        None
                        if recorded.diagnostic is None
                        else SelfModelRuntimeDiagnostic(
                            recorded.diagnostic.code,
                            recorded.diagnostic.message,
                        )
                    ),
                )
            else:
                receipt = SelfModelObservationReceipt(
                    False,
                    diagnostic=(
                        SelfModelRuntimeDiagnostic(
                            "self_model_observation_failed",
                            "Self Model observation was not persisted",
                        )
                        if recorded.diagnostic is None
                        else SelfModelRuntimeDiagnostic(
                            recorded.diagnostic.code,
                            recorded.diagnostic.message,
                        )
                    ),
                )
        except Exception as error:  # noqa: BLE001 - observation failure cannot alter the completed Run
            receipt = SelfModelObservationReceipt(
                False,
                diagnostic=SelfModelRuntimeDiagnostic(
                    "self_model_observation_failed",
                    f"{type(error).__name__}: {error}",
                ),
            )
        return self._remember_receipt(receipt)

    def render_user_snapshot(self) -> str:
        """Render the latest consumed snapshot; the command layer remains read-only."""

        return render_user_snapshot(
            enabled=self.enabled,
            runtime_scope=self.runtime_scope,
            snapshot=self.latest_snapshot,
            receipt=self.latest_receipt,
        )

    def _remember_snapshot(self, snapshot: SelfModelPlanningSnapshot) -> SelfModelPlanningSnapshot:
        with self._lock:
            self._latest_snapshot = snapshot
        return snapshot

    def _remember_receipt(self, receipt: SelfModelObservationReceipt) -> SelfModelObservationReceipt:
        with self._lock:
            self._latest_receipt = receipt
        return receipt


def create_self_model_runtime(
    *,
    project_root: str | Path,
    data_root: str | Path,
    enabled: bool = True,
) -> SelfModelRuntime:
    classifier = RuntimeTaskClassifier.from_project(project_root)
    service = SelfModelService(
        store=EvoEventStore(data_root),
        project_id=classifier.project_id,
    )
    return SelfModelRuntime(
        service=service,
        classifier=classifier,
        enabled=enabled,
    )


def render_system_advisory(snapshot: SelfModelPlanningSnapshot) -> str:
    """Bounded system guidance with an append-only profile as the audit anchor."""

    profile = snapshot.profile
    classification = snapshot.classification
    if profile is None or classification is None:
        return ""
    lines = [
        "Self Model planning advisory (system-generated; not a user message or permission grant).",
        f"Profile event: {profile.published_event_id or 'not_persisted'}",
        (
            "Scope: "
            f"project={classification.project_id}; language={classification.language}; "
            f"repository_scale={classification.repository_scale}; task_type={classification.task_type}; "
            f"tool_category={classification.tool_category}; risk={classification.risk_level}."
        ),
        (
            f"Evidence summary: status={profile.status}; samples={profile.sample_count}; "
            f"window={_utc(profile.window_start)}..{_utc(profile.window_end)}; "
            f"success_rate={_optional_rate(profile.success_rate)}; "
            f"confidence_interval={_interval(profile)}; uncertainty={_optional_rate(profile.uncertainty)}."
        ),
        (
            f"Provenance: observation_events={len(profile.source_event_ids)} "
            f"{_bounded_refs(profile.source_event_ids)}; source_runs={len(profile.source_run_ids)} "
            f"{_bounded_refs(profile.source_run_ids)}."
        ),
    ]
    if snapshot.suggestions:
        lines.append("Conservative planning suggestions (all permission_effect=none):")
        for suggestion in snapshot.suggestions:
            lines.append(f"- {suggestion.action} [{suggestion.reason_code}]: {suggestion.rationale}")
    else:
        lines.append("Conservative planning suggestions: none for this scoped profile.")
    lines.append(
        "Treat this only as planning/verification guidance: it may increase checks, decomposition, "
        "confirmation, or caution, but it must never broaden tools, filesystem/network scope, or permissions."
    )
    return "\n".join(lines)


def render_user_snapshot(
    *,
    enabled: bool,
    runtime_scope: str,
    snapshot: SelfModelPlanningSnapshot | None,
    receipt: SelfModelObservationReceipt | None,
) -> str:
    lines = [
        f"Self Model: {'enabled' if enabled else 'disabled'}",
        f"Runtime switch scope: {runtime_scope}",
        "Authority effect: none (advisory only; PermissionManager remains authoritative)",
    ]
    if snapshot is None:
        lines.append("Planning snapshot: none (run a task to build a scoped profile)")
        return "\n".join(lines)
    if snapshot.profile is None or snapshot.classification is None:
        lines.append("Planning snapshot: unavailable")
        if snapshot.diagnostic is not None:
            lines.append(f"Diagnostic: {snapshot.diagnostic.code}: {snapshot.diagnostic.message}")
        return "\n".join(lines)
    profile = snapshot.profile
    classification = snapshot.classification
    lines.extend(
        [
            f"Planning profile event: {profile.published_event_id or 'not persisted'}",
            (
                "Scope: "
                f"language={classification.language}, repository_scale={classification.repository_scale}, "
                f"task_type={classification.task_type}, tool_category={classification.tool_category}, "
                f"risk={classification.risk_level}"
            ),
            f"Status: {profile.status}",
            f"Samples: {profile.sample_count} ({profile.success_count} successful)",
            f"Window: {_utc(profile.window_start)} .. {_utc(profile.window_end)}",
            f"Success rate: {_optional_rate(profile.success_rate)}",
            f"95% confidence interval: {_interval(profile)}",
            f"Uncertainty: {_optional_rate(profile.uncertainty)}",
            f"Profile confidence: {profile.confidence:.3f}",
            f"Source observation events: {len(profile.source_event_ids)} {_bounded_refs(profile.source_event_ids)}",
            f"Source runs: {len(profile.source_run_ids)} {_bounded_refs(profile.source_run_ids)}",
            "Suggestions:",
        ]
    )
    if snapshot.suggestions:
        lines.extend(f"- {item.action} [{item.severity}/{item.reason_code}] permission_effect={item.permission_effect}" for item in snapshot.suggestions)
    else:
        lines.append("- none")
    if snapshot.diagnostic is not None:
        lines.append(f"Snapshot diagnostic: {snapshot.diagnostic.code}: {snapshot.diagnostic.message}")
    if receipt is not None:
        lines.append("Latest observation: " + (f"recorded as {receipt.observation_event_id}" if receipt.recorded else "not recorded"))
        if receipt.diagnostic is not None:
            lines.append(f"Observation diagnostic: {receipt.diagnostic.code}: {receipt.diagnostic.message}")
    return "\n".join(lines)


def _inspect_repository(root: Path) -> tuple[str, RepositoryScale]:
    suffix_counts: Counter[str] = Counter()
    file_count = 0
    try:
        for current_root, directory_names, file_names in os.walk(root):
            directory_names[:] = sorted(name for name in directory_names if name not in _SKIPPED_DIRECTORIES and not name.startswith("."))
            for file_name in sorted(file_names):
                file_count += 1
                language = _LANGUAGE_SUFFIXES.get(Path(file_name).suffix.casefold())
                if language is not None:
                    suffix_counts[language] += 1
                if file_count > 2_500:
                    return _dominant_language(suffix_counts), "large"
    except OSError:
        return "unknown", "unknown"
    scale: RepositoryScale = "small" if file_count <= 250 else "medium"
    return _dominant_language(suffix_counts), scale


def _dominant_language(counts: Counter[str]) -> str:
    if not counts:
        return "unknown"
    return min(counts, key=lambda language: (-counts[language], language))


def _language_for_task(text: str, *, fallback: str) -> str:
    markers = (
        ("python", ("python", "pytest", ".py")),
        ("typescript", ("typescript", ".ts", ".tsx")),
        ("javascript", ("javascript", ".js", ".jsx", "node.js")),
        ("rust", ("rust", "cargo", ".rs")),
        ("go", ("golang", "go test", ".go")),
        ("java", ("java", "maven", "gradle", ".java")),
    )
    for language, candidates in markers:
        if any(candidate in text for candidate in candidates):
            return language
    return fallback


def _task_type(text: str) -> str:
    rules = (
        ("migration", ("migration", "migrate", "迁移")),
        ("debugging", ("bug", "fix", "error", "traceback", "报错", "修复")),
        ("testing", ("test", "pytest", "测试", "验证")),
        ("refactoring", ("refactor", "重构")),
        ("documentation", ("document", "readme", "文档")),
        ("review", ("review", "audit", "审计", "检查")),
        ("implementation", ("implement", "build", "create", "新增", "实现")),
    )
    return next((name for name, markers in rules if any(marker in text for marker in markers)), "unknown")


def _tool_category(text: str) -> str:
    rules = (
        ("test", ("test", "pytest", "测试", "验证")),
        ("database", ("database", "sql", "schema", "数据库")),
        ("git", (" git ", "commit", "branch", "提交", "分支")),
        ("network", ("http", "api", "network", "联网", "网络")),
        ("filesystem", ("file", "directory", "path", "文件", "目录")),
    )
    padded = f" {text} "
    return next((name for name, markers in rules if any(marker in padded for marker in markers)), "unknown")


def _risk_level(text: str) -> str:
    high = (
        "credential",
        "delete",
        "deploy",
        "migration",
        "permission",
        "production",
        "secret",
        "删除",
        "部署",
        "权限",
        "生产",
        "迁移",
        "密钥",
    )
    if any(marker in text for marker in high):
        return "high"
    medium = ("change", "create", "edit", "fix", "implement", "refactor", "修改", "修复", "实现", "新增", "重构")
    return "medium" if any(marker in text for marker in medium) else "low"


def _utc(value) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _optional_rate(value: float | None) -> str:
    return "insufficient_data" if value is None else f"{value:.3f}"


def _interval(profile: SelfModelProfile) -> str:
    if profile.confidence_low is None or profile.confidence_high is None:
        return "insufficient_data"
    return f"[{profile.confidence_low:.3f}, {profile.confidence_high:.3f}]"


def _bounded_refs(values: tuple[str, ...], *, limit: int = 5) -> str:
    if not values:
        return "[]"
    visible = values[-limit:]
    omitted = len(values) - len(visible)
    suffix = f" (+{omitted} earlier; full list is in the profile event)" if omitted else ""
    return f"[{', '.join(visible)}]{suffix}"
