from __future__ import annotations

import json
import shutil
from pathlib import Path

from bauhinia_agent.evaluation.agent_baseline import (
    AgentBaselineConfig,
    RecordingProvider,
    _run_attempt,
    fixed_prompt,
    run_agent_baseline,
)
from bauhinia_agent.providers.base import ChatProvider
from bauhinia_agent.providers.types import (
    ChatRequest,
    ChatResponse,
    ProviderCapabilities,
    ProviderDiagnostics,
    TokenUsage,
    ToolCall,
)

CORPUS_ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "baseline_v0"


class _StopProvider(ChatProvider):
    def __init__(self) -> None:
        self.call_count = 0

    @property
    def name(self) -> str:
        return "deepseek"

    @property
    def model(self) -> str:
        return "deepseek-v4-flash"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities()

    def complete(self, request: ChatRequest) -> ChatResponse:
        self.call_count += 1
        return ChatResponse(
            provider=self.name,
            model=self.model,
            content="done",
            finish_reason="stop",
            usage=TokenUsage(input_tokens=10, output_tokens=2, total_tokens=12),
            diagnostics=ProviderDiagnostics(reasoning="private reasoning must not persist"),
        )


class _EditProvider(_StopProvider):
    def complete(self, request: ChatRequest) -> ChatResponse:
        self.call_count += 1
        if self.call_count == 1:
            return ChatResponse(
                provider=self.name,
                model=self.model,
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_edit",
                        name="edit",
                        arguments={
                            "path": "app.py",
                            "old": "return value > 0",
                            "new": "return value >= 0",
                        },
                    )
                ],
                finish_reason="tool_calls",
                usage=TokenUsage(input_tokens=20, output_tokens=5, total_tokens=25),
            )
        return super().complete(request)


def _manifest_case(case_id: str) -> dict[str, object]:
    manifest = json.loads((CORPUS_ROOT / "manifest.json").read_text(encoding="utf-8"))
    return next(case for case in manifest["cases"] if case["id"] == case_id)


def test_recording_provider_scrubs_reasoning_and_counts_usage() -> None:
    provider = RecordingProvider(_StopProvider())

    response = provider.complete(ChatRequest(messages=[]))

    assert response.diagnostics.reasoning is None
    assert provider.calls[0].total_tokens == 12


def test_fixed_prompt_protects_scenario_and_has_stable_task_slot() -> None:
    prompt = fixed_prompt("repair it")

    assert "Never modify scenario.json" in prompt
    assert prompt.endswith("repair it\n")


def test_live_attempt_resumes_safe_write_and_passes_independent_verifier(tmp_path: Path) -> None:
    attempt = _run_attempt(
        provider=_EditProvider(),
        config=AgentBaselineConfig(
            corpus_root=CORPUS_ROOT,
            output_root=tmp_path,
            repeats=2,
            thinking="disabled",
        ),
        corpus_root=CORPUS_ROOT,
        case=dict(_manifest_case("small-defect-zero-boundary")),
        attempt_number=1,
        attempt_root=tmp_path / "attempt",
    )

    assert attempt["task_success"] is True
    assert attempt["verification_passed"] is True
    assert attempt["scenario_unchanged"] is True
    assert attempt["human_intervention_required_count"] == 1
    assert attempt["scripted_operator_decision_count"] == 1
    assert attempt["actual_human_input_count"] == 0
    assert attempt["evo_run_id"].startswith("run_")


def test_full_batch_reports_all_required_metrics_and_reproducibility(tmp_path: Path) -> None:
    report, report_path = run_agent_baseline(
        _StopProvider(),
        AgentBaselineConfig(
            corpus_root=CORPUS_ROOT,
            output_root=tmp_path / "runs",
            repeats=2,
            thinking="disabled",
        ),
    )

    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted == report
    assert report["report_type"] == "p0_real_agent_baseline"
    assert report["claim_boundary"]["p0_005_satisfied"] is True
    assert report["metrics"]["attempt_count"] == 24
    assert report["metrics"]["tokens"]["total_tokens"] == 24 * 12
    assert report["runner"]["version"] == "1.0.0"
    assert len(report["runner"]["module_sha256"]) == 64
    assert report["metrics"]["run_to_run_outcome_reproducibility"] == 1.0
    assert len(report["case_reproducibility"]) == 12
    assert all(attempt["evo_run_id"].startswith("run_") for attempt in report["attempts"])


def test_full_batch_fails_closed_when_locked_corpus_is_modified(tmp_path: Path) -> None:
    corpus = tmp_path / "baseline_v0"
    shutil.copytree(CORPUS_ROOT, corpus)
    manifest = corpus / "manifest.json"
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    try:
        run_agent_baseline(
            _StopProvider(),
            AgentBaselineConfig(corpus_root=corpus, output_root=tmp_path / "runs", repeats=2),
        )
    except ValueError as error:
        assert str(error) == "P0 corpus integrity validation failed"
    else:
        raise AssertionError("modified corpus must be rejected before Agent execution")
