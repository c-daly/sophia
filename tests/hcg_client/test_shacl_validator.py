"""Unit tests for SHACL validator."""

import pytest
from sophia.hcg_client.shacl_validator import SHACLValidator


class TestSHACLValidator:
    """Test SHACL validator functionality."""

    @pytest.fixture
    def validator(self) -> SHACLValidator:
        """Create a SHACL validator instance."""
        return SHACLValidator()

    def test_validator_creation(self, validator: SHACLValidator) -> None:
        """Test that validator can be created."""
        assert validator is not None
        assert validator._shapes is not None

    def test_validate_valid_node(self, validator: SHACLValidator) -> None:
        """Test validation of a valid node."""
        node_data = {
            "id": "node1",
            "type": "concept",
            "properties": {"name": "test"},
        }

        is_valid, errors = validator.validate_node(node_data)
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_node_missing_type(self, validator: SHACLValidator) -> None:
        """Test validation fails for node without type."""
        node_data = {
            "id": "node1",
            "properties": {"name": "test"},
        }

        is_valid, errors = validator.validate_node(node_data)
        assert is_valid is False
        assert len(errors) > 0

    def test_validate_valid_edge(self, validator: SHACLValidator) -> None:
        """Test validation of a valid edge."""
        edge_data = {
            "id": "edge1",
            "source": "node1",
            "target": "node2",
            "relation": "connects",
            "properties": {},
        }

        is_valid, errors = validator.validate_edge(edge_data)
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_edge_missing_source(self, validator: SHACLValidator) -> None:
        """Test validation fails for edge without source."""
        edge_data = {
            "id": "edge1",
            "target": "node2",
            "relation": "connects",
        }

        is_valid, errors = validator.validate_edge(edge_data)
        assert is_valid is False
        assert len(errors) > 0

    def test_validate_edge_missing_target(self, validator: SHACLValidator) -> None:
        """Test validation fails for edge without target."""
        edge_data = {
            "id": "edge1",
            "source": "node1",
            "relation": "connects",
        }

        is_valid, errors = validator.validate_edge(edge_data)
        assert is_valid is False
        assert len(errors) > 0

    def test_validate_edge_missing_relation(self, validator: SHACLValidator) -> None:
        """Test validation fails for edge without relation."""
        edge_data = {
            "id": "edge1",
            "source": "node1",
            "target": "node2",
        }

        is_valid, errors = validator.validate_edge(edge_data)
        assert is_valid is False
        assert len(errors) > 0

    def test_validate_mutation_add_node(self, validator: SHACLValidator) -> None:
        """Test mutation validation for add_node."""
        node_data = {
            "id": "node1",
            "type": "concept",
            "properties": {},
        }

        is_valid, errors = validator.validate_mutation("add_node", node_data)
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_mutation_add_edge(self, validator: SHACLValidator) -> None:
        """Test mutation validation for add_edge."""
        edge_data = {
            "id": "edge1",
            "source": "node1",
            "target": "node2",
            "relation": "connects",
        }

        is_valid, errors = validator.validate_mutation("add_edge", edge_data)
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_mutation_unknown_type(self, validator: SHACLValidator) -> None:
        """Test mutation validation for unknown mutation type."""
        is_valid, errors = validator.validate_mutation("unknown", {})
        assert is_valid is False
        assert "Unknown mutation type" in errors[0]

    def test_custom_shapes_graph(self) -> None:
        """Test validator with custom SHACL shapes."""
        custom_shapes = """
        @prefix sh: <http://www.w3.org/ns/shacl#> .
        @prefix ex: <http://example.org/hcg/> .
        
        ex:CustomNodeShape a sh:NodeShape ;
            sh:targetClass ex:Node ;
            sh:property [
                sh:path ex:nodeType ;
                sh:minCount 1 ;
            ] .
        """

        validator = SHACLValidator(shapes_graph=custom_shapes)
        assert validator is not None

        # Valid node should pass
        node_data = {
            "id": "node1",
            "type": "concept",
        }
        is_valid, errors = validator.validate_node(node_data)
        assert is_valid is True
