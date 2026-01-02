"""Unit tests for the HCG seeder utilities."""

import pytest

from sophia.hcg_client.seeder import ANCESTORS, REQUIRED_TYPES


pytestmark = pytest.mark.unit


class TestSeederConfig:
    """Tests for seeder configuration."""

    def test_required_types_have_ancestors(self):
        """All required types must have entries in ANCESTORS dict."""
        missing = []
        for type_name in REQUIRED_TYPES:
            if type_name not in ANCESTORS:
                missing.append(type_name)

        assert not missing, f"Required types missing from ANCESTORS: {missing}"

    def test_ancestors_are_valid_lists(self):
        """All ANCESTORS entries must be non-empty lists."""
        invalid = []
        for type_name, ancestors in ANCESTORS.items():
            if not isinstance(ancestors, list):
                invalid.append(f"{type_name}: not a list")
            elif not ancestors:
                invalid.append(f"{type_name}: empty list")

        assert not invalid, f"Invalid ANCESTORS entries: {invalid}"

    def test_required_types_not_empty(self):
        """REQUIRED_TYPES must contain at least simulation and execution."""
        assert "simulation" in REQUIRED_TYPES
        assert "execution" in REQUIRED_TYPES
        assert "process" in REQUIRED_TYPES

    def test_parallel_container_types_exist(self):
        """Both imagined (simulation) and observed (execution) container types exist."""
        # Imagined containers
        assert "simulation" in ANCESTORS
        assert "imagined_process" in ANCESTORS
        assert "imagined_state" in ANCESTORS

        # Observed containers (parallel structure)
        assert "execution" in ANCESTORS
        assert "process" in ANCESTORS
        assert "state" in ANCESTORS
