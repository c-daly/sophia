"""Unit tests for the HCG seeder utilities."""

import pytest

from sophia.hcg_client.seeder import (
    seed_pick_and_place_data,
    seed_plan_data,
    seed_persona_entries,
)


pytestmark = pytest.mark.unit


class TestSeederExports:
    """Tests that seeder module exports the expected functions."""

    def test_seed_pick_and_place_data_is_callable(self):
        """seed_pick_and_place_data should be importable and callable."""
        assert callable(seed_pick_and_place_data)

    def test_seed_plan_data_is_callable(self):
        """seed_plan_data should be importable and callable."""
        assert callable(seed_plan_data)

    def test_seed_persona_entries_is_callable(self):
        """seed_persona_entries should be importable and callable."""
        assert callable(seed_persona_entries)
