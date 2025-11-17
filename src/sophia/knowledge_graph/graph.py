"""Knowledge graph implementation using NetworkX."""

from typing import Dict, List, Optional

import networkx as nx

from sophia.knowledge_graph.node import Node
from sophia.knowledge_graph.edge import Edge


class KnowledgeGraph:
    """In-memory knowledge graph using NetworkX.

    Provides basic operations for managing nodes and edges in a directed graph.
    """

    def __init__(self) -> None:
        """Initialize an empty knowledge graph."""
        self._graph: nx.DiGraph = nx.DiGraph()
        self._nodes: Dict[str, Node] = {}
        self._edges: Dict[str, Edge] = {}

    def add_node(self, node: Node) -> None:
        """Add a node to the knowledge graph.

        Args:
            node: The node to add
        """
        self._nodes[node.id] = node
        self._graph.add_node(node.id, data=node)

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the knowledge graph.

        Args:
            edge: The edge to add

        Raises:
            ValueError: If source or target node doesn't exist
        """
        if edge.source not in self._nodes:
            raise ValueError(f"Source node {edge.source} not found")
        if edge.target not in self._nodes:
            raise ValueError(f"Target node {edge.target} not found")

        self._edges[edge.id] = edge
        self._graph.add_edge(edge.source, edge.target, data=edge)

    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID.

        Args:
            node_id: The ID of the node to retrieve

        Returns:
            The node if found, None otherwise
        """
        return self._nodes.get(node_id)

    def get_edge(self, edge_id: str) -> Optional[Edge]:
        """Get an edge by ID.

        Args:
            edge_id: The ID of the edge to retrieve

        Returns:
            The edge if found, None otherwise
        """
        return self._edges.get(edge_id)

    def get_neighbors(self, node_id: str) -> List[Node]:
        """Get all neighbors of a node.

        Args:
            node_id: The ID of the node

        Returns:
            List of neighboring nodes
        """
        if node_id not in self._nodes:
            return []

        neighbor_ids = self._graph.neighbors(node_id)
        return [self._nodes[nid] for nid in neighbor_ids]

    def get_edges_from(self, node_id: str) -> List[Edge]:
        """Get all outgoing edges from a node.

        Args:
            node_id: The ID of the node

        Returns:
            List of outgoing edges
        """
        if node_id not in self._nodes:
            return []

        edges = []
        for _, target, edge_data in self._graph.out_edges(node_id, data=True):
            if "data" in edge_data:
                edges.append(edge_data["data"])

        return edges

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and its edges from the graph.

        Args:
            node_id: The ID of the node to remove

        Returns:
            True if node was removed, False if it didn't exist
        """
        if node_id not in self._nodes:
            return False

        # Remove associated edges
        edges_to_remove = [
            edge_id
            for edge_id, edge in self._edges.items()
            if edge.source == node_id or edge.target == node_id
        ]
        for edge_id in edges_to_remove:
            del self._edges[edge_id]

        # Remove node
        del self._nodes[node_id]
        self._graph.remove_node(node_id)

        return True

    def node_count(self) -> int:
        """Get the number of nodes in the graph.

        Returns:
            Number of nodes
        """
        return len(self._nodes)

    def edge_count(self) -> int:
        """Get the number of edges in the graph.

        Returns:
            Number of edges
        """
        return len(self._edges)

    def clear(self) -> None:
        """Remove all nodes and edges from the graph."""
        self._nodes.clear()
        self._edges.clear()
        self._graph.clear()
