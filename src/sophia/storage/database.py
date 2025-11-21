"""Database abstraction layer for persistent storage."""

from typing import Any, Dict, List, Optional, cast

from sqlalchemy import create_engine, Column, String, JSON, MetaData, Table
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine import CursorResult


class Database:
    """Database abstraction for storing knowledge graph data.

    Provides a simple interface for persisting nodes and edges.
    Uses SQLAlchemy for database operations.
    """

    def __init__(self, db_url: str = "sqlite:///sophia.db") -> None:
        """Initialize database connection.

        Args:
            db_url: SQLAlchemy database URL (default: sqlite:///sophia.db)
        """
        self.engine = create_engine(db_url)
        self.metadata = MetaData()

        # Define nodes table
        self.nodes_table = Table(
            "nodes",
            self.metadata,
            Column("id", String, primary_key=True),
            Column("type", String, nullable=False),
            Column("properties", JSON),
        )

        # Define edges table
        self.edges_table = Table(
            "edges",
            self.metadata,
            Column("id", String, primary_key=True),
            Column("source", String, nullable=False),
            Column("target", String, nullable=False),
            Column("relation", String, nullable=False),
            Column("properties", JSON),
        )

        # Create tables if they don't exist
        self.metadata.create_all(self.engine)

        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine)

    def _get_session(self) -> Session:
        """Get a new database session.

        Returns:
            SQLAlchemy session
        """
        return self.SessionLocal()

    def store_node(
        self, node_id: str, node_type: str, properties: Dict[str, Any]
    ) -> None:
        """Store a node in the database.

        Args:
            node_id: Unique identifier for the node
            node_type: Type/category of the node
            properties: Additional properties
        """
        session = self._get_session()
        try:
            # Delete existing node if present
            session.execute(
                self.nodes_table.delete().where(self.nodes_table.c.id == node_id)
            )

            # Insert new node
            session.execute(
                self.nodes_table.insert().values(
                    id=node_id, type=node_type, properties=properties
                )
            )
            session.commit()
        finally:
            session.close()

    def store_edge(
        self,
        edge_id: str,
        source: str,
        target: str,
        relation: str,
        properties: Dict[str, Any],
    ) -> None:
        """Store an edge in the database.

        Args:
            edge_id: Unique identifier for the edge
            source: Source node ID
            target: Target node ID
            relation: Relationship type
            properties: Additional properties
        """
        session = self._get_session()
        try:
            # Delete existing edge if present
            session.execute(
                self.edges_table.delete().where(self.edges_table.c.id == edge_id)
            )

            # Insert new edge
            session.execute(
                self.edges_table.insert().values(
                    id=edge_id,
                    source=source,
                    target=target,
                    relation=relation,
                    properties=properties,
                )
            )
            session.commit()
        finally:
            session.close()

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a node from the database.

        Args:
            node_id: ID of the node to retrieve

        Returns:
            Node data as dictionary, or None if not found
        """
        session = self._get_session()
        try:
            result = session.execute(
                self.nodes_table.select().where(self.nodes_table.c.id == node_id)
            ).fetchone()

            if result is None:
                return None

            return {
                "id": result.id,
                "type": result.type,
                "properties": result.properties or {},
            }
        finally:
            session.close()

    def get_edge(self, edge_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an edge from the database.

        Args:
            edge_id: ID of the edge to retrieve

        Returns:
            Edge data as dictionary, or None if not found
        """
        session = self._get_session()
        try:
            result = session.execute(
                self.edges_table.select().where(self.edges_table.c.id == edge_id)
            ).fetchone()

            if result is None:
                return None

            return {
                "id": result.id,
                "source": result.source,
                "target": result.target,
                "relation": result.relation,
                "properties": result.properties or {},
            }
        finally:
            session.close()

    def get_all_nodes(self) -> List[Dict[str, Any]]:
        """Retrieve all nodes from the database.

        Returns:
            List of node dictionaries
        """
        session = self._get_session()
        try:
            results = session.execute(self.nodes_table.select()).fetchall()
            return [
                {"id": row.id, "type": row.type, "properties": row.properties or {}}
                for row in results
            ]
        finally:
            session.close()

    def get_all_edges(self) -> List[Dict[str, Any]]:
        """Retrieve all edges from the database.

        Returns:
            List of edge dictionaries
        """
        session = self._get_session()
        try:
            results = session.execute(self.edges_table.select()).fetchall()
            return [
                {
                    "id": row.id,
                    "source": row.source,
                    "target": row.target,
                    "relation": row.relation,
                    "properties": row.properties or {},
                }
                for row in results
            ]
        finally:
            session.close()

    def delete_node(self, node_id: str) -> bool:
        """Delete a node from the database.

        Args:
            node_id: ID of the node to delete

        Returns:
            True if node was deleted, False if it didn't exist
        """
        session = self._get_session()
        try:
            result = cast(
                CursorResult[Any],
                session.execute(
                    self.nodes_table.delete().where(self.nodes_table.c.id == node_id)
                ),
            )
            session.commit()
            return bool(result.rowcount and result.rowcount > 0)
        finally:
            session.close()

    def delete_edge(self, edge_id: str) -> bool:
        """Delete an edge from the database.

        Args:
            edge_id: ID of the edge to delete

        Returns:
            True if edge was deleted, False if it didn't exist
        """
        session = self._get_session()
        try:
            result = cast(
                CursorResult[Any],
                session.execute(
                    self.edges_table.delete().where(self.edges_table.c.id == edge_id)
                ),
            )
            session.commit()
            return bool(result.rowcount and result.rowcount > 0)
        finally:
            session.close()

    def clear_all(self) -> None:
        """Clear all data from the database."""
        session = self._get_session()
        try:
            session.execute(self.edges_table.delete())
            session.execute(self.nodes_table.delete())
            session.commit()
        finally:
            session.close()
