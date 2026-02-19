"""Unit tests for the HCG seeder utilities."""

import pytest

from sophia.hcg_client.seeder import TYPE_HIERARCHY, REQUIRED_TYPES


pytestmark = pytest.mark.unit


class TestSeederConfig:
    """Tests for seeder configuration."""

    def test_required_types_have_hierarchy_entries(self):
        """All required types must have entries in TYPE_HIERARCHY dict."""
        missing = []
        for type_name in REQUIRED_TYPES:
            if type_name not in TYPE_HIERARCHY:
                missing.append(type_name)

        assert not missing, f"Required types missing from TYPE_HIERARCHY: {missing}"

    def test_hierarchy_values_are_strings(self):
        """All TYPE_HIERARCHY entries must map to non-empty parent type strings."""
        invalid = []
        for type_name, parent in TYPE_HIERARCHY.items():
            if not isinstance(parent, str):
                invalid.append(f"{type_name}: parent is not a string")
            elif not parent:
                invalid.append(f"{type_name}: empty parent")

        assert not invalid, f"Invalid TYPE_HIERARCHY entries: {invalid}"

    def test_required_types_not_empty(self):
        """REQUIRED_TYPES must contain at least simulation and execution."""
        assert "simulation" in REQUIRED_TYPES
        assert "execution" in REQUIRED_TYPES
        assert "process" in REQUIRED_TYPES

    def test_parallel_container_types_exist(self):
        """Both imagined (simulation) and observed (execution) container types exist."""
        # Imagined containers
        assert "simulation" in TYPE_HIERARCHY
        assert "imagined_process" in TYPE_HIERARCHY
        assert "imagined_state" in TYPE_HIERARCHY

        # Observed containers (parallel structure)
        assert "execution" in TYPE_HIERARCHY
        assert "process" in TYPE_HIERARCHY
        assert "state" in TYPE_HIERARCHY

    def test_entity_is_root(self):
        """entity should be a root type (value in hierarchy, not a key with entity parent)."""
        # entity appears as a parent but should itself have no parent
        # (it's the root of the hierarchy)
        assert "entity" not in TYPE_HIERARCHY or TYPE_HIERARCHY.get("entity") is None
