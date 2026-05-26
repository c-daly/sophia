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
    """Hermes' answer to 'what binds these together?' -- just a label."""

    label: str
    description: str
    confidence: float
