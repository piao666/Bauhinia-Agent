from __future__ import annotations

from pathlib import Path

from bauhinia_agent.evolution.evidence import EvidenceAdapter, EvidenceInput
from bauhinia_agent.evolution.events import EvidenceRecordedPayload
from bauhinia_agent.evolution.identifiers import new_evo_id
from bauhinia_agent.evolution.store import EvoEventStore, EvoStoreError


def test_evidence_payload_round_trips_execution_metadata() -> None:
    payload = EvidenceRecordedPayload.from_dict(
        {
            "evidence_type": "test",
            "source": "pytest",
            "summary": "2 passed",
            "command": "python -m pytest",
            "input_summary": "tests/test_example.py",
            "cwd": "C:/repo",
            "exit_code": 0,
            "redacted": True,
        }
    )

    assert payload.to_dict() == {
        "evidence_type": "test",
        "source": "pytest",
        "summary": "2 passed",
        "locator": None,
        "verified": False,
        "command": "python -m pytest",
        "input_summary": "tests/test_example.py",
        "cwd": "C:/repo",
        "exit_code": 0,
        "redacted": True,
    }


def test_records_sanitized_test_evidence_and_queries_by_run(tmp_path: Path) -> None:
    adapter = EvidenceAdapter(EvoEventStore(tmp_path / ".bauhinia-agent"))
    run_id = new_evo_id("run")

    recorded = adapter.record(
        EvidenceInput(
            run_id=run_id,
            evidence_type="test",
            source="pytest",
            summary="Authorization: Bearer top-secret failed",
            command="OPENAI_API_KEY=sk-live-secret python -m pytest",
            cwd="C:/repo",
            exit_code=1,
            verified=True,
        )
    )

    assert recorded.persisted is True
    assert recorded.evidence is not None
    assert recorded.evidence.payload.summary == "Authorization: Bearer [REDACTED] failed"
    assert recorded.evidence.payload.command == "OPENAI_API_KEY=[REDACTED] python -m pytest"
    assert recorded.evidence.payload.exit_code == 1
    assert adapter.list_for_run(run_id) == [recorded.evidence]
    persisted = (tmp_path / ".bauhinia-agent" / "evo" / "events.jsonl").read_text(encoding="utf-8")
    assert "top-secret" not in persisted
    assert "sk-live-secret" not in persisted


def test_records_tool_and_permission_evidence_for_the_same_run(tmp_path: Path) -> None:
    adapter = EvidenceAdapter(EvoEventStore(tmp_path / ".bauhinia-agent"))
    run_id = new_evo_id("run")

    tool = adapter.record_tool(
        run_id=run_id,
        tool_name="shell",
        tool_call_id="call_1",
        arguments={"command": "echo $TOKEN=private"},
        ok=False,
        summary="command failed with token=private",
    )
    permission = adapter.record_permission(
        run_id=run_id,
        action="execute_shell",
        target="curl https://example.invalid -H 'Authorization: Bearer private'",
        decision="deny",
        reason="user denied",
    )

    records = adapter.list_for_run(run_id)
    assert [record.evidence_type for record in records] == ["tool", "permission"]
    assert tool.evidence is not None
    assert tool.evidence.payload.input_summary == '{"command":"echo $TOKEN=[REDACTED]"}'
    assert permission.evidence is not None
    assert permission.evidence.payload.input_summary == "curl https://example.invalid -H 'Authorization: Bearer [REDACTED]'"


def test_recorder_failure_returns_a_discoverable_diagnostic_without_raising() -> None:
    adapter = EvidenceAdapter(_FailingStore())

    recorded = adapter.record(
        EvidenceInput(
            run_id=new_evo_id("run"),
            evidence_type="test",
            source="pytest",
            summary="passed",
        )
    )

    assert recorded.persisted is False
    assert recorded.evidence is None
    assert recorded.diagnostic is not None
    assert "offline" in recorded.diagnostic.message


class _FailingStore:
    def append(self, event: object) -> object:
        raise EvoStoreError("store offline")
