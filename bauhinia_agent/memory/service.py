"""P3 memory projection over the canonical append-only Evo event store."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Callable

from bauhinia_agent.evolution import EvoEvent, EvoEventStore, EvoReferences, MemoryCreatedPayload, new_evo_id
from bauhinia_agent.memory.models import MemoryModelError, MemoryRecord

_WORDS = re.compile(r"[\w-]+", re.UNICODE)


class MemoryWriteDisabledError(MemoryModelError):
    pass


class MemoryService:
    """Writes source events and rebuilds deterministic in-memory projections."""

    def __init__(
        self,
        *,
        store: EvoEventStore,
        project_id: str,
        writes_enabled: bool = True,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._store = store
        self._project_id = project_id
        self._writes_enabled = writes_enabled
        self._clock = clock or (lambda: datetime.now(UTC))

    @property
    def writes_enabled(self) -> bool:
        return self._writes_enabled

    def set_writes_enabled(self, enabled: bool) -> None:
        self._writes_enabled = bool(enabled)

    def create(self, record: MemoryRecord) -> MemoryRecord:
        if not self._writes_enabled:
            raise MemoryWriteDisabledError("memory writes are disabled")
        if record.scope.project_id != self._project_id:
            raise MemoryModelError("cannot write memory outside this project scope")
        event = EvoEvent(
            event_id=new_evo_id("event"),
            event_type="MemoryCreated",
            refs=EvoReferences(run_id=record.provenance.source_run_ids[0], memory_id=record.memory_id),
            payload=MemoryCreatedPayload(
                memory_type=record.layer,
                content=record.content,
                scope="project",
                confidence=record.confidence,
                source_event_ids=record.provenance.source_event_ids,
                extensions={"memory_record": record.to_dict()},
            ),
        )
        self._store.append(event)
        return record

    def rebuild(self) -> list[MemoryRecord]:
        records: list[MemoryRecord] = []
        for event in self._store.list_events():
            if event.event_type != "MemoryCreated" or not isinstance(event.payload, MemoryCreatedPayload):
                continue
            raw = event.payload.extensions.get("memory_record")
            if isinstance(raw, dict):
                record = MemoryRecord.from_dict(raw)
                if record.scope.project_id == self._project_id:
                    records.append(record)
        return sorted(records, key=lambda record: (record.created_at, record.memory_id))

    def search(self, query: str, *, user_id: str | None = None, session_id: str | None = None) -> list[MemoryRecord]:
        terms = set(_WORDS.findall(query.lower()))
        if not terms:
            return []
        scored: list[tuple[int, MemoryRecord]] = []
        for record in self.rebuild():
            if not record.is_readable_by(project_id=self._project_id, user_id=user_id, session_id=session_id):
                continue
            score = len(terms.intersection(_WORDS.findall(record.content.lower())))
            if score and record.freshness_weight(at=self._clock()):
                scored.append((score, record))
        return [record for _, record in sorted(scored, key=lambda item: (-item[0], item[1].memory_id))]
