"""Shared value types for ontology-evolution emergence (#505)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Member:
    """A node being considered for emergent re-typing."""

    uuid: str
    name: str
    embedding: list[float]
    signature: Counter  # Counter[(relation_type, neighbor_type)]
    current_type: str
    hermes_type_hint: str | None
    neighbors: list[dict[str, Any]] = field(default_factory=list)
    model: str | None = None  # embedding model (for centroid upsert)


@dataclass
class EmergentCluster:
    """A group of members that agree on both signals and may become a type."""

    members: list[Member]

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def embeddings(self) -> list[list[float]]:
        return [m.embedding for m in self.members]


@dataclass
class NameResult:
    """Hermes' answer to 'what binds these together?' -- a label, the member
    uuids that don't fit it (outliers to leave in the base type, #504), and an
    optional graft ``parent``: when the label is newly coined, the existing type
    to attach it under so a concept-cluster lands under ``concept`` etc.; None on
    reuse or when no listed category is a sensible parent."""

    label: str
    description: str
    confidence: float
    removed: list[str] = field(default_factory=list)
    parent: str | None = None


@dataclass(frozen=True)
class TypeClusterResult:
    """Hermes v2 /type-cluster verdict: a type ``name``, an optional existing
    ``parent`` to mint it under (None means reuse ``name`` as-is), and member
    ``residual_ids`` that do not fit (already mapped back by Hermes)."""

    name: str
    parent: str | None
    residual_ids: list[str]
