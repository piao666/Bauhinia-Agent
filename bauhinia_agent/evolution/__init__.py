"""Evo domain contracts.

This package contains versioned, provider-independent event and identifier models.
It deliberately does not own persistence or application wiring.
"""

from bauhinia_agent.evolution.events import (
    EVO_EVENT_SCHEMA_VERSION,
    EVO_EVENT_TYPES,
    DecisionRecordedPayload,
    EvaluationCompletedPayload,
    EvidenceRecordedPayload,
    EvoEvent,
    EvoEventError,
    EvoPayload,
    EvoReferences,
    ExperienceCandidateCreatedPayload,
    MemoryCreatedPayload,
    MemoryUsedPayload,
    OutcomeClassifiedPayload,
    PlanCreatedPayload,
    PlanNodeUpdatedPayload,
    PromotionChangedPayload,
    SelfModelUpdatedPayload,
    UnknownEvoPayload,
)
from bauhinia_agent.evolution.identifiers import (
    IdentifierKind,
    EvoIdentifierError,
    new_evo_id,
    require_evo_id,
)
from bauhinia_agent.evolution.store import (
    EvoAppendResult,
    EvoEventStore,
    EvoProjectionStats,
    EvoRepairResult,
    EvoStoreCorruptError,
    EvoStoreDiagnostic,
    EvoStoreError,
    EvoStoreLockError,
)
from bauhinia_agent.evolution.migration import (
    EvoMigrationError,
    EvoMigrationManager,
    EvoMigrationResult,
    EvoImportResult,
    EvoSchemaReport,
)

__all__ = [
    "DecisionRecordedPayload",
    "EVO_EVENT_SCHEMA_VERSION",
    "EVO_EVENT_TYPES",
    "EvaluationCompletedPayload",
    "EvidenceRecordedPayload",
    "EvoEvent",
    "EvoEventError",
    "EvoEventStore",
    "EvoAppendResult",
    "EvoIdentifierError",
    "EvoImportResult",
    "EvoMigrationError",
    "EvoMigrationManager",
    "EvoMigrationResult",
    "EvoPayload",
    "EvoProjectionStats",
    "EvoRepairResult",
    "EvoReferences",
    "EvoSchemaReport",
    "EvoStoreCorruptError",
    "EvoStoreDiagnostic",
    "EvoStoreError",
    "EvoStoreLockError",
    "ExperienceCandidateCreatedPayload",
    "IdentifierKind",
    "MemoryCreatedPayload",
    "MemoryUsedPayload",
    "OutcomeClassifiedPayload",
    "PlanCreatedPayload",
    "PlanNodeUpdatedPayload",
    "PromotionChangedPayload",
    "SelfModelUpdatedPayload",
    "UnknownEvoPayload",
    "new_evo_id",
    "require_evo_id",
]
