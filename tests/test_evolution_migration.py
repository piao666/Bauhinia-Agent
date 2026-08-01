from __future__ import annotations

from bauhinia_agent.context.events import SessionEvent
from bauhinia_agent.context.store import JsonlSessionStore
from bauhinia_agent.evolution.events import EvoEvent
from bauhinia_agent.evolution.migration import EvoMigrationManager
from bauhinia_agent.evolution.store import EvoEventStore


def _legacy_line(event_id: str = "event_legacy") -> str:
    return (
        '{"id":"'
        + event_id
        + '","type":"PlanCreated","created_at":"2026-08-01T00:00:00Z",'
        '"sequence":1,"run_id":"run_1","payload":{"goal":"legacy","legacy_field":true}}\n'
    )


def test_detect_and_read_legacy_schema_without_mutating_source(tmp_path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    store.events_path.write_text(_legacy_line(), encoding="utf-8")
    manager = EvoMigrationManager(store)

    report = manager.detect_schema()
    events = manager.read_compatible()

    assert report.source_exists
    assert report.event_count == 1
    assert report.schema_versions == {"v0": 1}
    assert events[0].schema_version == "v0"
    assert events[0].payload.to_dict()["legacy_field"] is True
    assert store.events_path.read_text(encoding="utf-8") == _legacy_line()


def test_migrate_to_current_logs_backup_and_supports_rollback(tmp_path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    store.events_path.write_text(_legacy_line(), encoding="utf-8")
    manager = EvoMigrationManager(store)

    result = manager.migrate_to_current()

    assert result.changed
    assert result.backup_path is not None and result.backup_path.exists()
    assert [event.schema_version for event in store.list_events()] == ["v1"]
    assert store.list_events()[0].payload.to_dict()["legacy_field"] is True
    assert store.projection_stats().event_count == 1
    log_text = result.log_path.read_text(encoding="utf-8")
    assert '"status":"started"' in log_text
    assert '"status":"completed"' in log_text

    manager.rollback(result.migration_id)

    assert [event.schema_version for event in store.list_events()] == ["v0"]
    assert store.projection_stats().event_count == 1
    assert result.backup_path.exists()
    assert '"status":"rolled_back"' in result.log_path.read_text(encoding="utf-8")


def test_missing_evo_data_is_a_noop_and_old_session_still_rebuilds(tmp_path) -> None:
    data_root = tmp_path / ".bauhinia-agent"
    session_store = JsonlSessionStore(data_root)
    session_store.append_event(
        SessionEvent(
            id="evt_session",
            session_id="sess_old",
            type="session_created",
            payload={"title": "old session"},
            created_at="2026-08-01T00:00:00Z",
        )
    )
    evo_store = EvoEventStore(data_root)
    manager = EvoMigrationManager(evo_store)

    report = manager.detect_schema()
    result = manager.migrate_to_current()

    assert not report.source_exists
    assert report.event_count == 0
    assert not evo_store.has_evo_data()
    assert not result.changed
    assert session_store.rebuild_session_view("sess_old").metadata["title"] == "old session"


def test_migration_detects_partial_tail_without_rewriting_it(tmp_path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    store.events_path.write_text(_legacy_line() + '{"id":"broken"', encoding="utf-8")
    manager = EvoMigrationManager(store)

    report = manager.detect_schema()

    assert report.schema_versions == {"v0": 1}
    assert report.diagnostics
    assert report.diagnostics[0].code == "truncated_tail"


def test_import_appends_events_and_reassigns_destination_sequence(tmp_path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    store.append(
        EvoEvent.from_dict(
            {
                "event_id": "event_existing",
                "event_type": "PlanCreated",
                "schema_version": "v1",
                "occurred_at": "2026-08-01T00:00:00Z",
                "sequence": 1,
                "refs": {"run_id": "run_1"},
                "payload": {"goal": "existing"},
            }
        )
    )
    source = tmp_path / "import.jsonl"
    source.write_text(_legacy_line("event_imported"), encoding="utf-8")

    result = EvoMigrationManager(store).import_events(source)

    assert result.imported_count == 1
    assert result.source_versions == {"v0": 1}
    assert result.projection_applied
    events = store.list_events()
    assert [(event.event_id, event.sequence) for event in events] == [("event_existing", 1), ("event_imported", 2)]
