from __future__ import annotations

import json
import subprocess
import sys

import pytest

from bauhinia_agent.evolution.events import EvoEvent, EvoReferences, PlanCreatedPayload
from bauhinia_agent.evolution.store import EvoEventStore, EvoStoreCorruptError, EvoStoreLockError


def _event(event_id: str, *, sequence: int | None = None, event_type: str = "PlanCreated") -> EvoEvent:
    return EvoEvent(
        event_id=event_id,
        event_type=event_type,
        refs=EvoReferences(run_id="run_1", plan_id="plan_1"),
        payload=PlanCreatedPayload(goal=event_id),
        occurred_at="2026-08-01T00:00:00Z",
        sequence=sequence,
    )


def test_append_assigns_sequence_and_updates_projection(tmp_path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")

    first = store.append(_event("event_1"))
    second = store.append(_event("event_2"))

    assert first.event.sequence == 1
    assert first.projection_applied
    assert second.event.sequence == 2
    assert [event.event_id for event in store.list_events()] == ["event_1", "event_2"]
    assert store.projection_stats().event_count == 2
    assert store.projection_stats().last_sequence == 2
    assert [row["event_id"] for row in store.projection_events()] == ["event_1", "event_2"]
    assert json.loads(store.events_path.read_text(encoding="utf-8").splitlines()[0])["sequence"] == 1


def test_append_rejects_duplicate_and_caller_forged_sequence(tmp_path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    store.append(_event("event_1"))

    with pytest.raises(Exception, match="duplicate event_id"):
        store.append(_event("event_1"))
    with pytest.raises(Exception, match="sequence must be 2"):
        store.append(_event("event_2", sequence=99))


def test_projection_can_be_deleted_and_rebuilt_from_raw_source(tmp_path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    store.append(_event("event_1"))
    store.append(_event("event_2"))
    store.projection_path.unlink()

    stats = store.rebuild_projection()

    assert stats.event_count == 2
    assert stats.last_event_id == "event_2"
    assert [row["event_id"] for row in store.projection_events()] == ["event_1", "event_2"]


def test_unknown_event_is_stored_and_projected_without_blocking_replay(tmp_path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    unknown = EvoEvent.from_dict(
        {
            "event_id": "event_future",
            "event_type": "FutureEvent",
            "schema_version": "v9",
            "occurred_at": "2026-08-01T00:00:00Z",
            "refs": {"run_id": "run_1"},
            "payload": {"future": True},
        }
    )

    result = store.append(unknown)

    assert result.event.sequence == 1
    assert store.list_events()[0].event_type == "FutureEvent"
    assert store.projection_events()[0]["event_type"] == "FutureEvent"


def test_incomplete_tail_is_diagnosed_and_requires_explicit_repair(tmp_path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    store.append(_event("event_1"))
    with store.events_path.open("ab") as file:
        file.write(b'{"event_id":"broken"')

    diagnostics = store.diagnose()
    assert diagnostics[0].code == "truncated_tail"
    with pytest.raises(EvoStoreCorruptError):
        store.append(_event("event_2"))

    repaired = store.repair_tail()
    assert repaired.changed
    assert repaired.operation == "truncate_incomplete_tail"
    assert repaired.recovery_log is not None and repaired.recovery_log.exists()
    assert [event.event_id for event in store.list_events()] == ["event_1"]
    assert store.append(_event("event_2")).event.sequence == 2


def test_complete_invalid_line_is_not_automatically_repaired(tmp_path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    store.append(_event("event_1"))
    with store.events_path.open("ab") as file:
        file.write(b'{"not_an_event":true}\n')

    repaired = store.repair_tail()

    assert not repaired.changed
    assert repaired.operation == "not_repairable"
    with pytest.raises(EvoStoreCorruptError):
        store.list_events()


def test_projection_failure_returns_diagnostic_but_keeps_raw_event(tmp_path, monkeypatch) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")

    def fail_projection(events):
        raise RuntimeError("projection unavailable")

    monkeypatch.setattr(store, "_synchronize_projection_unlocked", fail_projection)
    result = store.append(_event("event_1"))

    assert not result.projection_applied
    assert result.diagnostic is not None
    assert result.diagnostic.code == "projection_update_failed"
    assert [event.event_id for event in store.list_events()] == ["event_1"]


def test_lock_timeout_is_visible(tmp_path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent", lock_timeout=0.05)
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import portalocker, sys, time\n"
            "with portalocker.Lock(sys.argv[1], mode='a+b', timeout=5, flags=portalocker.LOCK_EX):\n"
            "    print('locked', flush=True)\n"
            "    time.sleep(2)\n",
            str(store.lock_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "locked"
        with pytest.raises(EvoStoreLockError):
            store.list_events()
    finally:
        child.terminate()
        child.wait(timeout=10)
