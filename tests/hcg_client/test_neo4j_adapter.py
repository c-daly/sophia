"""Unit tests for Neo4j adapter (without live database)."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from sophia.hcg_client.neo4j_adapter import Neo4jAdapter
from sophia.hcg_client.shacl_validator import SHACLValidator


class TestNeo4jAdapterUnit:
    """Unit tests for Neo4j adapter without live database."""

    @pytest.fixture
    def mock_validator(self) -> SHACLValidator:
        """Create a mock validator that always passes."""
        validator = Mock(spec=SHACLValidator)
        validator.validate_node.return_value = (True, [])
        validator.validate_edge.return_value = (True, [])
        return validator

    @patch('sophia.hcg_client.neo4j_adapter.GraphDatabase')
    def test_adapter_initialization(
        self,
        mock_graphdb: Mock,
        mock_validator: SHACLValidator,
    ) -> None:
        """Test Neo4j adapter initialization."""
        mock_driver = Mock()
        mock_graphdb.driver.return_value = mock_driver
        
        adapter = Neo4jAdapter(
            uri="bolt://test:7687",
            username="test_user",
            password="test_pass",
            validator=mock_validator,
        )
        
        assert adapter is not None
        mock_graphdb.driver.assert_called_once()

    @patch('sophia.hcg_client.neo4j_adapter.GraphDatabase')
    def test_add_node_with_validation(
        self,
        mock_graphdb: Mock,
        mock_validator: SHACLValidator,
    ) -> None:
        """Test adding a node with validation."""
        mock_driver = Mock()
        mock_session = MagicMock()
        mock_result = Mock()
        mock_record = Mock()
        mock_record.__getitem__ = Mock(return_value="node1")
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = Mock(return_value=None)
        mock_graphdb.driver.return_value = mock_driver
        
        adapter = Neo4jAdapter(validator=mock_validator)
        
        node_data = {
            "id": "node1",
            "type": "concept",
            "properties": {"name": "test"},
        }
        
        result = adapter.add_node(node_data)
        
        assert result == "node1"
        mock_validator.validate_node.assert_called_once_with(node_data)

    @patch('sophia.hcg_client.neo4j_adapter.GraphDatabase')
    def test_add_node_validation_failure(
        self,
        mock_graphdb: Mock,
    ) -> None:
        """Test that invalid node raises ValueError."""
        mock_driver = Mock()
        mock_graphdb.driver.return_value = mock_driver
        
        validator = Mock(spec=SHACLValidator)
        validator.validate_node.return_value = (False, ["Invalid node"])
        
        adapter = Neo4jAdapter(validator=validator)
        
        node_data = {
            "id": "node1",
            "properties": {},
        }
        
        with pytest.raises(ValueError, match="validation failed"):
            adapter.add_node(node_data)

    @patch('sophia.hcg_client.neo4j_adapter.GraphDatabase')
    def test_add_edge_with_validation(
        self,
        mock_graphdb: Mock,
        mock_validator: SHACLValidator,
    ) -> None:
        """Test adding an edge with validation."""
        mock_driver = Mock()
        mock_session = MagicMock()
        mock_result = Mock()
        mock_record = Mock()
        mock_record.__getitem__ = Mock(return_value="edge1")
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = Mock(return_value=None)
        mock_graphdb.driver.return_value = mock_driver
        
        adapter = Neo4jAdapter(validator=mock_validator)
        
        edge_data = {
            "id": "edge1",
            "source": "node1",
            "target": "node2",
            "relation": "connects",
            "properties": {},
        }
        
        result = adapter.add_edge(edge_data)
        
        assert result == "edge1"
        mock_validator.validate_edge.assert_called_once_with(edge_data)

    @patch('sophia.hcg_client.neo4j_adapter.GraphDatabase')
    def test_add_edge_validation_failure(
        self,
        mock_graphdb: Mock,
    ) -> None:
        """Test that invalid edge raises ValueError."""
        mock_driver = Mock()
        mock_graphdb.driver.return_value = mock_driver
        
        validator = Mock(spec=SHACLValidator)
        validator.validate_edge.return_value = (False, ["Invalid edge"])
        
        adapter = Neo4jAdapter(validator=validator)
        
        edge_data = {
            "id": "edge1",
            "source": "node1",
        }
        
        with pytest.raises(ValueError, match="validation failed"):
            adapter.add_edge(edge_data)

    @patch('sophia.hcg_client.neo4j_adapter.GraphDatabase')
    def test_close(
        self,
        mock_graphdb: Mock,
        mock_validator: SHACLValidator,
    ) -> None:
        """Test closing the adapter."""
        mock_driver = Mock()
        mock_graphdb.driver.return_value = mock_driver
        
        adapter = Neo4jAdapter(validator=mock_validator)
        adapter.close()
        
        mock_driver.close.assert_called_once()
