"""Sophia: Non-linguistic cognitive core for Project LOGOS."""

__version__ = "0.1.0"

from sophia.knowledge_graph.graph import KnowledgeGraph
from sophia.storage.database import Database
from sophia.orchestrator.orchestrator import Orchestrator
from sophia.cwm_a.memory import ContinuousWorkingMemoryAssociative
from sophia.cwm_g.memory import ContinuousWorkingMemoryGenerative
from sophia.planner.planner import Planner
from sophia.executor.executor import Executor

__all__ = [
    "KnowledgeGraph",
    "Database",
    "Orchestrator",
    "ContinuousWorkingMemoryAssociative",
    "ContinuousWorkingMemoryGenerative",
    "Planner",
    "Executor",
    "__version__",
]
