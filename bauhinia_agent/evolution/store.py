"""Append-only Evo event source and rebuildable SQLite projection."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator, Literal

import portalocker

from bauhinia_agent.evolution.events import EvoEvent, EvoEventError


class EvoStoreError(RuntimeError):
    """Base error for source, lock, projection, and recovery failures."""


class EvoStoreCorruptError(EvoStoreError):
    """The source cannot be safely replayed without explicit recovery."""

    def __init__(self, diagnostics: list["EvoStoreDiagnostic"]) -> None:
        self.diagnostics = tuple(diagnostics)
        message = "; ".join(item.message for item in self.diagnostics) or "Evo event source is corrupt"
        super().__init__(message)


class EvoStoreLockError(EvoStoreError):
    """The source lock could not be acquired within the configured timeout."""


DiagnosticSeverity = Literal["error", "warning"]


@dataclass(frozen=True, slots=True)
class EvoStoreDiagnostic:
    code: str
    message: str
    severity: DiagnosticSeverity = "error"
    line_number: int | None = None
    byte_offset: int | None = None
    event_id: str | None = None
    recoverable: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "line_number": self.line_number,
            "byte_offset": self.byte_offset,
            "event_id": self.event_id,
            "recoverable": self.recoverable,
        }


@dataclass(frozen=True, slots=True)
class EvoAppendResult:
    event: EvoEvent
    projection_applied: bool
    diagnostic: EvoStoreDiagnostic | None = None


@dataclass(frozen=True, slots=True)
class EvoRepairResult:
    changed: bool
    operation: str
    diagnostic: EvoStoreDiagnostic | None = None
    recovery_log: Path | None = None


@dataclass(frozen=True, slots=True)
class EvoProjectionStats:
    event_count: int
    last_sequence: int
    last_event_id: str | None


@dataclass(frozen=True, slots=True)
class _ScanResult:
    events: tuple[EvoEvent, ...]
    diagnostics: tuple[EvoStoreDiagnostic, ...]


class EvoEventStore:
    """Persist Evo facts to JSONL and maintain a derived SQLite projection.

    ``root`` is the existing project data root, normally ``.bauhinia-agent``.
    The store owns only the ``evo`` child directory and never reads or writes the
    existing session JSONL files.
    """

    def __init__(self, root: str | Path, *, lock_timeout: float = 5.0) -> None:
        self.root = Path(root)
        self.evo_root = self.root / "evo"
        self.events_path = self.evo_root / "events.jsonl"
        self.lock_path = self.evo_root / "events.lock"
        self.projection_path = self.evo_root / "projection.sqlite3"
        self.recovery_log_path = self.evo_root / "recovery-log.jsonl"
        self.lock_timeout = lock_timeout
        self.evo_root.mkdir(parents=True, exist_ok=True)

    def append(self, event: EvoEvent) -> EvoAppendResult:
        """Append one fact and best-effort update its derived projection."""

        with self._locked(portalocker.LOCK_EX):
            scan = self._scan_unlocked()
            self._raise_on_any(scan.diagnostics)
            existing_ids = {item.event_id for item in scan.events}
            if event.event_id in existing_ids:
                raise EvoStoreError(f"duplicate event_id: {event.event_id}")
            next_sequence = len(scan.events) + 1
            if event.sequence is not None and event.sequence != next_sequence:
                raise EvoStoreError(f"event sequence must be {next_sequence}, got {event.sequence}")
            persisted = replace(event, sequence=next_sequence)
            persisted.validate_persisted()
            self._append_raw_unlocked(persisted)

            try:
                self._synchronize_projection_unlocked((*scan.events, persisted))
            except Exception as error:  # noqa: BLE001 - projection must not change source success
                diagnostic = EvoStoreDiagnostic(
                    code="projection_update_failed",
                    message=f"raw Evo event {persisted.event_id} was persisted but projection update failed: {error}",
                    event_id=persisted.event_id,
                    recoverable=True,
                )
                return EvoAppendResult(event=persisted, projection_applied=False, diagnostic=diagnostic)
            return EvoAppendResult(event=persisted, projection_applied=True)

    def list_events(self) -> list[EvoEvent]:
        """Read all valid source events, failing closed on corruption."""

        with self._locked(portalocker.LOCK_SH):
            scan = self._scan_unlocked()
            self._raise_on_any(scan.diagnostics)
            return list(scan.events)

    def diagnose(self) -> list[EvoStoreDiagnostic]:
        """Return structured source diagnostics without modifying any file."""

        with self._locked(portalocker.LOCK_SH):
            return list(self._scan_unlocked().diagnostics)

    def rebuild_projection(self) -> EvoProjectionStats:
        """Rebuild SQLite from the canonical source using an atomic replacement."""

        with self._locked(portalocker.LOCK_EX):
            scan = self._scan_unlocked()
            self._raise_on_any(scan.diagnostics)
            self._build_projection_unlocked(scan.events)
            return self._projection_stats_unlocked()

    def projection_stats(self) -> EvoProjectionStats:
        if not self.projection_path.exists():
            return EvoProjectionStats(event_count=0, last_sequence=0, last_event_id=None)
        try:
            return self._projection_stats_unlocked()
        except sqlite3.Error as error:
            raise EvoStoreError(f"cannot read Evo projection: {error}") from error

    def projection_events(self, *, event_type: str | None = None) -> list[dict[str, object]]:
        """Read derived event rows for diagnostics and future query services."""

        if not self.projection_path.exists():
            return []
        try:
            connection = sqlite3.connect(self.projection_path)
            try:
                query = "SELECT event_id, event_type, schema_version, occurred_at, sequence, run_id, canonical_json FROM event_index"
                parameters: tuple[object, ...] = ()
                if event_type is not None:
                    query += " WHERE event_type = ?"
                    parameters = (event_type,)
                query += " ORDER BY sequence"
                rows = connection.execute(query, parameters).fetchall()
            finally:
                connection.close()
        except sqlite3.Error as error:
            raise EvoStoreError(f"cannot query Evo projection: {error}") from error
        return [
            {
                "event_id": row[0],
                "event_type": row[1],
                "schema_version": row[2],
                "occurred_at": row[3],
                "sequence": row[4],
                "run_id": row[5],
                "canonical_json": row[6],
            }
            for row in rows
        ]

    def repair_tail(self) -> EvoRepairResult:
        """Explicitly repair only a final missing newline or incomplete line.

        Complete invalid lines, semantic errors, duplicate IDs, and sequence gaps
        are never repaired automatically.
        """

        with self._locked(portalocker.LOCK_EX):
            if not self.events_path.exists():
                return EvoRepairResult(changed=False, operation="no_source")
            data = self.events_path.read_bytes()
            if not data:
                return EvoRepairResult(changed=False, operation="empty_source")
            if data.endswith(b"\n"):
                diagnostics = self._scan_unlocked().diagnostics
                return EvoRepairResult(
                    changed=False,
                    operation="not_repairable",
                    diagnostic=diagnostics[0] if diagnostics else None,
                )

            last_newline = data.rfind(b"\n")
            tail_offset = last_newline + 1
            tail = data[tail_offset:]
            operation = "append_final_newline"
            try:
                candidate = EvoEvent.from_json(tail.decode("utf-8"))
                candidate.validate_persisted()
            except (UnicodeDecodeError, EvoEventError, ValueError):
                operation = "truncate_incomplete_tail"
                candidate = None

            before_hash = hashlib.sha256(data).hexdigest()
            diagnostic = EvoStoreDiagnostic(
                code="missing_final_newline" if candidate is not None else "truncated_tail",
                message=(
                    "final Evo event is valid but missing its newline"
                    if candidate is not None
                    else "final Evo JSONL tail is incomplete and will be truncated"
                ),
                line_number=data.count(b"\n") + 1,
                byte_offset=tail_offset,
                event_id=candidate.event_id if candidate is not None else None,
                severity="warning",
                recoverable=True,
            )
            if candidate is not None:
                repaired = data + b"\n"
            else:
                repaired = data[:tail_offset]
                prior_scan = self._scan_bytes(data[:tail_offset])
                self._raise_on_errors(prior_scan.diagnostics)

            self._write_recovery_log_unlocked(
                {
                    "operation": operation,
                    "source_size": len(data),
                    "source_sha256": before_hash,
                    "dropped_bytes": len(data) - len(repaired),
                    "diagnostic": diagnostic.to_dict(),
                }
            )
            with self.events_path.open("wb") as file:
                file.write(repaired)
                file.flush()
                os.fsync(file.fileno())
            return EvoRepairResult(changed=True, operation=operation, diagnostic=diagnostic, recovery_log=self.recovery_log_path)

    def has_evo_data(self) -> bool:
        return self.events_path.exists() and self.events_path.stat().st_size > 0

    @contextmanager
    def _locked(self, flags: int) -> Iterator[None]:
        try:
            with portalocker.Lock(str(self.lock_path), mode="a+b", timeout=self.lock_timeout, flags=flags | portalocker.LOCK_NB):
                yield
        except portalocker.exceptions.LockException as error:
            raise EvoStoreLockError(f"could not acquire Evo store lock {self.lock_path}: {error}") from error

    def _append_raw_unlocked(self, event: EvoEvent) -> None:
        encoded = (event.to_json() + "\n").encode("utf-8")
        with self.events_path.open("ab") as file:
            file.write(encoded)
            file.flush()
            os.fsync(file.fileno())

    def _scan_unlocked(self) -> _ScanResult:
        if not self.events_path.exists():
            return _ScanResult(events=(), diagnostics=())
        return self._scan_bytes(self.events_path.read_bytes())

    def _scan_bytes(self, data: bytes) -> _ScanResult:
        if not data:
            return _ScanResult(events=(), diagnostics=())
        events: list[EvoEvent] = []
        diagnostics: list[EvoStoreDiagnostic] = []
        expected_sequence = 1
        seen_ids: set[str] = set()
        lines = data.splitlines(keepends=True)
        offset = 0
        for line_number, raw_line in enumerate(lines, start=1):
            current_offset = offset
            offset += len(raw_line)
            has_newline = raw_line.endswith(b"\n")
            content = raw_line[:-1] if has_newline else raw_line
            if not content:
                continue
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError as error:
                diagnostics.append(
                    EvoStoreDiagnostic(
                        code="invalid_utf8",
                        message=f"invalid UTF-8 at line {line_number}: {error}",
                        line_number=line_number,
                        byte_offset=current_offset,
                        recoverable=not has_newline,
                    )
                )
                if not has_newline:
                    break
                continue
            try:
                event = EvoEvent.from_json(text)
                event.validate_persisted()
            except (EvoEventError, ValueError, json.JSONDecodeError) as error:
                diagnostics.append(
                    EvoStoreDiagnostic(
                        code="truncated_tail" if not has_newline else "invalid_event",
                        message=f"invalid Evo event at line {line_number}: {error}",
                        line_number=line_number,
                        byte_offset=current_offset,
                        recoverable=not has_newline,
                    )
                )
                if not has_newline:
                    break
                continue
            if not has_newline:
                diagnostics.append(
                    EvoStoreDiagnostic(
                        code="missing_final_newline",
                        message=f"valid Evo event {event.event_id} is missing its final newline",
                        severity="warning",
                        line_number=line_number,
                        byte_offset=current_offset,
                        event_id=event.event_id,
                        recoverable=True,
                    )
                )
            if event.event_id in seen_ids:
                diagnostics.append(
                    EvoStoreDiagnostic(
                        code="duplicate_event_id",
                        message=f"duplicate event_id {event.event_id}",
                        line_number=line_number,
                        byte_offset=current_offset,
                        event_id=event.event_id,
                    )
                )
            seen_ids.add(event.event_id)
            if event.sequence != expected_sequence:
                diagnostics.append(
                    EvoStoreDiagnostic(
                        code="sequence_gap",
                        message=f"expected sequence {expected_sequence}, got {event.sequence}",
                        line_number=line_number,
                        byte_offset=current_offset,
                        event_id=event.event_id,
                    )
                )
                expected_sequence = max(expected_sequence, event.sequence + 1)
            else:
                expected_sequence += 1
            events.append(event)
        return _ScanResult(events=tuple(events), diagnostics=tuple(diagnostics))

    @staticmethod
    def _raise_on_errors(diagnostics: tuple[EvoStoreDiagnostic, ...] | list[EvoStoreDiagnostic]) -> None:
        errors = [item for item in diagnostics if item.severity == "error"]
        if errors:
            raise EvoStoreCorruptError(errors)

    @staticmethod
    def _raise_on_any(diagnostics: tuple[EvoStoreDiagnostic, ...] | list[EvoStoreDiagnostic]) -> None:
        if diagnostics:
            raise EvoStoreCorruptError(list(diagnostics))

    def _synchronize_projection_unlocked(self, events: tuple[EvoEvent, ...]) -> None:
        expected_last = events[-1].sequence if events else 0
        if not self.projection_path.exists():
            self._build_projection_unlocked(events)
            return
        try:
            stats = self._projection_stats_unlocked()
        except sqlite3.Error:
            self._build_projection_unlocked(events)
            return
        if stats.last_sequence != expected_last - 1:
            self._build_projection_unlocked(events)
            return
        connection = sqlite3.connect(self.projection_path)
        try:
            self._create_schema(connection)
            connection.execute("BEGIN")
            self._insert_projection_event(connection, events[-1])
            self._set_meta(connection, "last_sequence", str(events[-1].sequence))
            self._set_meta(connection, "last_event_id", events[-1].event_id)
            connection.commit()
        finally:
            connection.close()

    def _build_projection_unlocked(self, events: tuple[EvoEvent, ...]) -> None:
        temp_path = self.projection_path.with_name(f"{self.projection_path.name}.tmp")
        if temp_path.exists():
            temp_path.unlink()
        try:
            connection = sqlite3.connect(temp_path)
            try:
                self._create_schema(connection)
                connection.execute("BEGIN")
                for event in events:
                    self._insert_projection_event(connection, event)
                self._set_meta(connection, "last_sequence", str(events[-1].sequence if events else 0))
                self._set_meta(connection, "last_event_id", events[-1].event_id if events else "")
                connection.commit()
            finally:
                connection.close()
            os.replace(temp_path, self.projection_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS projection_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS event_index (
                event_id TEXT PRIMARY KEY,
                sequence INTEGER NOT NULL UNIQUE,
                event_type TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                run_id TEXT NOT NULL,
                session_id TEXT,
                plan_id TEXT,
                node_id TEXT,
                memory_id TEXT,
                candidate_id TEXT,
                evidence_id TEXT,
                evaluation_id TEXT,
                promotion_id TEXT,
                self_model_id TEXT,
                parent_event_id TEXT,
                canonical_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_event_type ON event_index(event_type);
            CREATE INDEX IF NOT EXISTS idx_run_sequence ON event_index(run_id, sequence);
            CREATE INDEX IF NOT EXISTS idx_parent_event ON event_index(parent_event_id);
            """
        )

    @staticmethod
    def _insert_projection_event(connection: sqlite3.Connection, event: EvoEvent) -> None:
        refs = event.refs
        connection.execute(
            """
            INSERT INTO event_index (
                event_id, sequence, event_type, schema_version, occurred_at, run_id,
                session_id, plan_id, node_id, memory_id, candidate_id, evidence_id,
                evaluation_id, promotion_id, self_model_id, parent_event_id, canonical_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.sequence,
                event.event_type,
                event.schema_version,
                event.occurred_at,
                refs.run_id,
                refs.session_id,
                refs.plan_id,
                refs.node_id,
                refs.memory_id,
                refs.candidate_id,
                refs.evidence_id,
                refs.evaluation_id,
                refs.promotion_id,
                refs.self_model_id,
                refs.parent_event_id,
                event.to_json(),
            ),
        )

    @staticmethod
    def _set_meta(connection: sqlite3.Connection, key: str, value: str) -> None:
        connection.execute("INSERT INTO projection_meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

    def _projection_stats_unlocked(self) -> EvoProjectionStats:
        connection = sqlite3.connect(self.projection_path)
        try:
            row = connection.execute("SELECT COUNT(*), COALESCE(MAX(sequence), 0) FROM event_index").fetchone()
            last_event = connection.execute("SELECT event_id FROM event_index ORDER BY sequence DESC LIMIT 1").fetchone()
        finally:
            connection.close()
        return EvoProjectionStats(event_count=int(row[0]), last_sequence=int(row[1]), last_event_id=last_event[0] if last_event else None)

    def _write_recovery_log_unlocked(self, record: dict[str, object]) -> None:
        encoded = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        with self.recovery_log_path.open("ab") as file:
            file.write(encoded)
            file.flush()
            os.fsync(file.fileno())
