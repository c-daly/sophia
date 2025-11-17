"""Sophia: Non-linguistic cognitive core for Project LOGOS."""

__version__ = "0.1.0"

from sophia.knowledge_graph.graph import KnowledgeGraph
from sophia.storage.database import Database

__all__ = ["KnowledgeGraph", "Database", "__version__"]
