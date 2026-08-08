"""Version detection, compatible reads, migration logging, and rollback."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import portalocker

from bauhinia_agent.evolution.events import EVO_EVENT_SCHEMA_VERSION, EvoEvent
from bauhinia_agent.evolution.store import EvoEventStore, EvoStoreDiagnostic


class EvoMigrationError(RuntimeError):
    """Raised when an explicit migration or rollback cannot complete safely."""


@dataclass(frozen=True, slots=True)
class EvoSchemaReport:
    source_exists: bool
    event_count: int
    schema_versions: dict[str, int]
    diagnostics: tuple[EvoStoreDiagnostic, ...] = ()

    @property
    def current(self) -> bool:
        return all(version == EVO_EVENT_SCHEMA_VERSION for version in self.schema_versions)


@dataclass(frozen=True, slots=True)
class EvoMigrationResult:
    migration_id: str
    changed: bool
    event_count: int
    source_versions: dict[str, int]
    target_version: str
    backup_path: Path | None
    log_path: Path


@dataclass(frozen=True, slots=True)
class EvoImportResult:
    imported_count: int
    source_versions: dict[str, int]
    projection_applied: bool
    diagnostic: EvoStoreDiagnostic | None
    log_path: Path


class EvoMigrationManager:
    """Manage explicit, reversible canonicalization of compatible Evo events."""

    def __init__(self, store: EvoEventStore) -> None:
        self.store = store
        self.log_path = store.evo_root / "migration-log.jsonl"

    def detect_schema(self) -> EvoSchemaReport:
        """Detect mixed/legacy versions without changing source or projection."""

        with self.store._locked(portalocker.LOCK_SH):
            if not self.store.events_path.exists():
                return EvoSchemaReport(source_exists=False, event_count=0, schema_versions={})
            scan = self.store._scan_unlocked()
        versions: dict[str, int] = {}
        for event in scan.events:
            versions[event.schema_version] = versions.get(event.schema_version, 0) + 1
        return EvoSchemaReport(
            source_exists=True,
            event_count=len(scan.events),
            schema_versions=versions,
            diagnostics=scan.diagnostics,
        )

    def read_compatible(self) -> list[EvoEvent]:
        """Read current and legacy events through the P1-001 compatibility parser."""

        return self.store.list_events()

    def import_events(self, source: str | Path) -> EvoImportResult:
        """Validate and append events from an explicit local JSONL file.

        Imported sequence numbers are never trusted: the destination assigns
        the next contiguous sequence while preserving event IDs, schema versions,
        unknown fields, and canonical payloads.
        """

        source_path = Path(source).resolve()
        if not source_path.is_file():
            raise EvoMigrationError(f"import source is not a regular file: {source_path}")
        if source_path == self.store.events_path.resolve():
            raise EvoMigrationError("import source cannot be the destination events.jsonl")
        imported = _read_import_events(source_path)
        source_versions = _version_counts(tuple(imported))
        with self.store._locked(portalocker.LOCK_EX):
            existing_scan = self.store._scan_unlocked()
            self.store._raise_on_any(existing_scan.diagnostics)
            existing_ids = {event.event_id for event in existing_scan.events}
            duplicate_ids = existing_ids.intersection(event.event_id for event in imported)
            if duplicate_ids:
                raise EvoMigrationError(f"import contains existing event IDs: {sorted(duplicate_ids)}")
            start_sequence = len(existing_scan.events) + 1
            persisted = tuple(replace(event, sequence=start_sequence + index) for index, event in enumerate(imported))
            with self.store.events_path.open("ab") as file:
                for event in persisted:
                    file.write((event.to_json() + "\n").encode("utf-8"))
                file.flush()
                os.fsync(file.fileno())
            all_events = (*existing_scan.events, *persisted)
            diagnostic: EvoStoreDiagnostic | None = None
            projection_applied = True
            try:
                self.store._build_projection_unlocked(all_events)
            except Exception as error:  # noqa: BLE001 - source append remains authoritative
                projection_applied = False
                diagnostic = EvoStoreDiagnostic(
                    code="projection_update_failed",
                    message=f"imported raw events but projection update failed: {error}",
                    recoverable=True,
                )
            self._append_log(
                {
                    "status": "imported",
                    "source_path": str(source_path),
                    "imported_count": len(persisted),
                    "source_versions": source_versions,
                    "projection_applied": projection_applied,
                    "diagnostic": diagnostic.to_dict() if diagnostic else None,
                }
            )
            return EvoImportResult(len(persisted), source_versions, projection_applied, diagnostic, self.log_path)

    def migrate_to_current(self) -> EvoMigrationResult:
        """Canonicalize legacy events to the current schema with a retained backup."""

        migration_id = f"migration_{uuid4().hex[:16]}"
        with self.store._locked(portalocker.LOCK_EX):
            if not self.store.events_path.exists():
                self._append_log(
                    {
                        "migration_id": migration_id,
                        "status": "no_source",
                        "target_version": EVO_EVENT_SCHEMA_VERSION,
                    }
                )
                return EvoMigrationResult(migration_id, False, 0, {}, EVO_EVENT_SCHEMA_VERSION, None, self.log_path)

            scan = self.store._scan_unlocked()
            self.store._raise_on_any(scan.diagnostics)
            versions = _version_counts(scan.events)
            legacy = [event for event in scan.events if event.schema_version != EVO_EVENT_SCHEMA_VERSION]
            if not legacy:
                self._append_log(
                    {
                        "migration_id": migration_id,
                        "status": "no_change",
                        "event_count": len(scan.events),
                        "source_versions": versions,
                        "target_version": EVO_EVENT_SCHEMA_VERSION,
                    }
                )
                return EvoMigrationResult(migration_id, False, len(scan.events), versions, EVO_EVENT_SCHEMA_VERSION, None, self.log_path)

            source_bytes = self.store.events_path.read_bytes()
            backup_path = self.store.evo_root / f"events.jsonl.backup.{migration_id}"
            temp_path = self.store.evo_root / f"events.jsonl.migrate.{migration_id}.tmp"
            shutil.copy2(self.store.events_path, backup_path)
            self._append_log(
                {
                    "migration_id": migration_id,
                    "status": "started",
                    "event_count": len(scan.events),
                    "source_versions": versions,
                    "target_version": EVO_EVENT_SCHEMA_VERSION,
                    "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                    "backup_path": str(backup_path.name),
                }
            )
            try:
                with temp_path.open("wb") as file:
                    for event in scan.events:
                        migrated = replace(event, schema_version=EVO_EVENT_SCHEMA_VERSION)
                        file.write((migrated.to_json() + "\n").encode("utf-8"))
                    file.flush()
                    os.fsync(file.fileno())
                os.replace(temp_path, self.store.events_path)
            except Exception as error:  # noqa: BLE001 - preserve source and log failure
                if temp_path.exists():
                    temp_path.unlink()
                self._append_log({"migration_id": migration_id, "status": "failed", "error": str(error)})
                raise EvoMigrationError(f"Evo migration failed; original source remains: {error}") from error

            self._append_log(
                {
                    "migration_id": migration_id,
                    "status": "completed",
                    "event_count": len(scan.events),
                    "source_versions": versions,
                    "target_version": EVO_EVENT_SCHEMA_VERSION,
                    "backup_path": str(backup_path.name),
                }
            )
            self.store._build_projection_unlocked(tuple(replace(event, schema_version=EVO_EVENT_SCHEMA_VERSION) for event in scan.events))
            return EvoMigrationResult(migration_id, True, len(scan.events), versions, EVO_EVENT_SCHEMA_VERSION, backup_path, self.log_path)

    def rollback(self, migration_id: str) -> Path:
        """Restore a retained migration backup without deleting the backup."""

        if not migration_id or any(character in migration_id for character in "\\/.."):
            raise EvoMigrationError("invalid migration_id")
        backup_path = self.store.evo_root / f"events.jsonl.backup.{migration_id}"
        if not backup_path.exists():
            raise EvoMigrationError(f"migration backup not found: {migration_id}")
        with self.store._locked(portalocker.LOCK_EX):
            temp_path = self.store.evo_root / f"events.jsonl.rollback.{migration_id}.tmp"
            shutil.copy2(backup_path, temp_path)
            try:
                os.replace(temp_path, self.store.events_path)
            finally:
                if temp_path.exists():
                    temp_path.unlink()
            scan = self.store._scan_unlocked()
            self.store._raise_on_any(scan.diagnostics)
            self.store._build_projection_unlocked(scan.events)
            self._append_log(
                {
                    "migration_id": migration_id,
                    "status": "rolled_back",
                    "event_count": len(scan.events),
                    "restored_backup": str(backup_path.name),
                }
            )
        return backup_path

    def _append_log(self, record: dict[str, object]) -> None:
        payload = {
            "recorded_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            **record,
        }
        with self.log_path.open("a", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())


def _version_counts(events: tuple[EvoEvent, ...]) -> dict[str, int]:
    result: dict[str, int] = {}
    for event in events:
        result[event.schema_version] = result.get(event.schema_version, 0) + 1
    return result


def _read_import_events(source: Path) -> list[EvoEvent]:
    data = source.read_bytes()
    events: list[EvoEvent] = []
    for line_number, raw_line in enumerate(data.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            event = EvoEvent.from_json(raw_line.decode("utf-8"))
            event.validate_persisted()
        except (UnicodeDecodeError, ValueError) as error:
            raise EvoMigrationError(f"invalid import event at line {line_number}: {error}") from error
        events.append(event)
    return events
