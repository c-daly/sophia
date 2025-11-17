"""Tests for Database class."""

from sophia.storage.database import Database


def test_database_creation(temp_db_path: str) -> None:
    """Test creating a database."""
    db = Database(f"sqlite:///{temp_db_path}")

    assert db.engine is not None


def test_store_and_get_node(temp_db_path: str) -> None:
    """Test storing and retrieving a node."""
    db = Database(f"sqlite:///{temp_db_path}")

    db.store_node("n1", "concept", {"name": "test"})
    node = db.get_node("n1")

    assert node is not None
    assert node["id"] == "n1"
    assert node["type"] == "concept"
    assert node["properties"]["name"] == "test"


def test_store_and_get_edge(temp_db_path: str) -> None:
    """Test storing and retrieving an edge."""
    db = Database(f"sqlite:///{temp_db_path}")

    # Store nodes first
    db.store_node("n1", "concept", {})
    db.store_node("n2", "concept", {})

    # Store edge
    db.store_edge("e1", "n1", "n2", "relates_to", {"weight": 1.0})
    edge = db.get_edge("e1")

    assert edge is not None
    assert edge["id"] == "e1"
    assert edge["source"] == "n1"
    assert edge["target"] == "n2"
    assert edge["relation"] == "relates_to"
    assert edge["properties"]["weight"] == 1.0


def test_get_nonexistent_node(temp_db_path: str) -> None:
    """Test retrieving a node that doesn't exist."""
    db = Database(f"sqlite:///{temp_db_path}")

    node = db.get_node("nonexistent")
    assert node is None


def test_get_all_nodes(temp_db_path: str) -> None:
    """Test retrieving all nodes."""
    db = Database(f"sqlite:///{temp_db_path}")

    db.store_node("n1", "concept", {"name": "node1"})
    db.store_node("n2", "entity", {"name": "node2"})

    nodes = db.get_all_nodes()

    assert len(nodes) == 2
    node_ids = {n["id"] for n in nodes}
    assert node_ids == {"n1", "n2"}


def test_get_all_edges(temp_db_path: str) -> None:
    """Test retrieving all edges."""
    db = Database(f"sqlite:///{temp_db_path}")

    db.store_node("n1", "concept", {})
    db.store_node("n2", "concept", {})
    db.store_edge("e1", "n1", "n2", "relates_to", {})
    db.store_edge("e2", "n2", "n1", "inverse_of", {})

    edges = db.get_all_edges()

    assert len(edges) == 2
    edge_ids = {e["id"] for e in edges}
    assert edge_ids == {"e1", "e2"}


def test_delete_node(temp_db_path: str) -> None:
    """Test deleting a node."""
    db = Database(f"sqlite:///{temp_db_path}")

    db.store_node("n1", "concept", {})
    assert db.delete_node("n1") is True
    assert db.get_node("n1") is None


def test_delete_nonexistent_node(temp_db_path: str) -> None:
    """Test deleting a node that doesn't exist."""
    db = Database(f"sqlite:///{temp_db_path}")

    assert db.delete_node("nonexistent") is False


def test_delete_edge(temp_db_path: str) -> None:
    """Test deleting an edge."""
    db = Database(f"sqlite:///{temp_db_path}")

    db.store_node("n1", "concept", {})
    db.store_node("n2", "concept", {})
    db.store_edge("e1", "n1", "n2", "relates_to", {})

    assert db.delete_edge("e1") is True
    assert db.get_edge("e1") is None


def test_clear_all(temp_db_path: str) -> None:
    """Test clearing all data."""
    db = Database(f"sqlite:///{temp_db_path}")

    db.store_node("n1", "concept", {})
    db.store_node("n2", "concept", {})
    db.store_edge("e1", "n1", "n2", "relates_to", {})

    db.clear_all()

    assert len(db.get_all_nodes()) == 0
    assert len(db.get_all_edges()) == 0


def test_update_node(temp_db_path: str) -> None:
    """Test updating a node."""
    db = Database(f"sqlite:///{temp_db_path}")

    db.store_node("n1", "concept", {"version": 1})
    db.store_node("n1", "concept", {"version": 2})

    node = db.get_node("n1")
    assert node is not None
    assert node["properties"]["version"] == 2
