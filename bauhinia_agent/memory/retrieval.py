"""P4 deterministic retrieval, context packing, and usage feedback."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Iterable

from bauhinia_agent.evolution import EvoEvent, EvoReferences, MemoryUsedPayload, new_evo_id
from bauhinia_agent.memory.models import MemoryRecord
from bauhinia_agent.memory.service import MemoryService

_WORDS = re.compile(r"[\w-]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class QuerySignature:
    goal: str
    repository_features: tuple[str, ...] = ()
    error_type: str | None = None
    tool_environment: tuple[str, ...] = ()
    plan_node_id: str | None = None
    constraints: tuple[str, ...] = ()

    @property
    def terms(self) -> frozenset[str]:
        return frozenset(_WORDS.findall(" ".join((self.goal, *self.repository_features, *(self.tool_environment), *(self.constraints), self.error_type or "")).lower()))


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    record: MemoryRecord
    score: int
    reasons: tuple[str, ...]
    token_cost: int


@dataclass(frozen=True, slots=True)
class ContextPack:
    hits: tuple[RetrievalHit, ...]
    omitted: tuple[tuple[str, str], ...]
    token_budget: int
    used_tokens: int


class MemoryRetriever:
    def __init__(self, service: MemoryService) -> None:
        self._service = service

    def retrieve(self, signature: QuerySignature, *, user_id: str | None = None, session_id: str | None = None, at: datetime | None = None) -> list[RetrievalHit]:
        terms = signature.terms
        retrieved_at = datetime.now(UTC) if at is None else at
        hits: list[RetrievalHit] = []
        for record in self._service.rebuild():
            if not record.is_readable_by(project_id=self._service._project_id, user_id=user_id, session_id=session_id):
                continue
            matched = terms.intersection(_WORDS.findall(record.content.lower()))
            if not matched or not record.freshness_weight(at=retrieved_at):
                continue
            reasons = ["project_scope", "keyword:" + ",".join(sorted(matched))]
            if signature.error_type and signature.error_type.lower() in record.content.lower():
                reasons.append("exact_error_type")
            hits.append(RetrievalHit(record, len(matched), tuple(reasons), len(_WORDS.findall(record.content))))
        return sorted(hits, key=lambda hit: (-hit.score, hit.record.memory_id))

    def pack(self, signature: QuerySignature, *, token_budget: int, user_id: str | None = None, session_id: str | None = None, at: datetime | None = None) -> ContextPack:
        if token_budget < 0:
            raise ValueError("token_budget must be non-negative")
        hits, omitted, used, conflicts = [], [], 0, set()
        for hit in self.retrieve(signature, user_id=user_id, session_id=session_id, at=at):
            group = hit.record.conflict_group
            if group and group in conflicts:
                omitted.append((hit.record.memory_id, "conflict_group"))
                continue
            if used + hit.token_cost > token_budget:
                omitted.append((hit.record.memory_id, "token_budget"))
                continue
            hits.append(hit)
            used += hit.token_cost
            if group:
                conflicts.add(group)
        return ContextPack(tuple(hits), tuple(omitted), token_budget, used)

    def record_use(self, hit: RetrievalHit, *, reason: str, helpfulness: str | None = None) -> None:
        self._service._store.append(
            EvoEvent(
                event_id=new_evo_id("event"),
                event_type="MemoryUsed",
                refs=EvoReferences(run_id=hit.record.provenance.source_run_ids[0], memory_id=hit.record.memory_id),
                payload=MemoryUsedPayload(reason=reason, helpfulness=helpfulness, extensions={"retrieval_reasons": list(hit.reasons), "score": hit.score}),
            )
        )
