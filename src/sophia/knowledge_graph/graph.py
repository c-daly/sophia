"""Knowledge graph implementation using NetworkX.

Edges are stored as reified nodes connected to source and target via
structural :FROM/:TO relationships, matching the Neo4j reified-edge model
used in the logos foundry.
"""

from typing import Dict, List, Optional

import networkx as nx

from sophia.knowledge_graph.node import Node
from sophia.knowledge_graph.edge import Edge


class KnowledgeGraph:
    """In-memory knowledge graph using NetworkX.

    Provides basic operations for managing nodes and edges in a directed graph.
    Edges are reified: each Edge is stored as a graph node with structural
    ``FROM`` and ``TO`` connections to the source and target content nodes.
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
        self._graph.add_node(node.id, data=node, is_edge=False)

    def add_edge(self, edge: Edge) -> None:
        """Add a reified edge to the knowledge graph.

        The edge is stored as a node in the NetworkX graph connected via
        structural ``FROM`` and ``TO`` relationships::

            (source)<--[FROM]--(edge_node)--[TO]-->(target)

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
        # Store edge as a graph node
        self._graph.add_node(edge.id, data=edge, is_edge=True)
        # Structural relationships: edge --FROM--> source, edge --TO--> target
        self._graph.add_edge(edge.id, edge.source, rel="FROM")
        self._graph.add_edge(edge.id, edge.target, rel="TO")

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
        """Get all content-node neighbors of a node via reified edges.

        Traverses edge nodes that connect to *node_id* via :FROM or :TO
        and returns the content nodes on the other side.

        Args:
            node_id: The ID of the node

        Returns:
            List of neighboring content nodes (deduplicated)
        """
        if node_id not in self._nodes:
            return []

        seen: set[str] = set()
        neighbors: List[Node] = []

        # Find edge nodes that point TO this node (outgoing edges from node_id)
        # and edge nodes that have FROM this node (incoming edges to node_id)
        for edge in self._edges.values():
            if edge.source == node_id:
                if edge.target not in seen:
                    seen.add(edge.target)
                    neighbors.append(self._nodes[edge.target])
            elif edge.target == node_id:
                if edge.source not in seen:
                    seen.add(edge.source)
                    neighbors.append(self._nodes[edge.source])

        return neighbors

    def get_edges_from(self, node_id: str) -> List[Edge]:
        """Get all outgoing edges from a node.

        Args:
            node_id: The ID of the source node

        Returns:
            List of edges whose source is *node_id*
        """
        if node_id not in self._nodes:
            return []

        return [e for e in self._edges.values() if e.source == node_id]

    def get_edges_to(self, node_id: str) -> List[Edge]:
        """Get all incoming edges to a node.

        Args:
            node_id: The ID of the target node

        Returns:
            List of edges whose target is *node_id*
        """
        if node_id not in self._nodes:
            return []

        return [e for e in self._edges.values() if e.target == node_id]

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and any edge nodes that reference it.

        Args:
            node_id: The ID of the node to remove

        Returns:
            True if node was removed, False if it didn't exist
        """
        if node_id not in self._nodes:
            return False

        # Collect edge nodes that reference this node
        edges_to_remove = [
            edge_id
            for edge_id, edge in self._edges.items()
            if edge.source == node_id or edge.target == node_id
        ]
        for edge_id in edges_to_remove:
            # Remove the edge's graph-node and its structural relationships
            if self._graph.has_node(edge_id):
                self._graph.remove_node(edge_id)
            del self._edges[edge_id]

        # Remove content node
        del self._nodes[node_id]
        self._graph.remove_node(node_id)

        return True

    def node_count(self) -> int:
        """Get the number of content nodes in the graph.

        Returns:
            Number of content nodes (excludes reified edge nodes)
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
