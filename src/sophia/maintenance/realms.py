"""Shared realm/root vocabulary for the positional typing layer.

Single source of truth for the two fixed sets the typing layer keys on, so the
ingest, emergence, rollup, placement, and snapshot-publish paths agree:

- ``GRAFTABLE_REALMS`` -- the domain roots a new type may be grafted under. These
  are the only valid closed-world parents the naming LLM may pick; they must
  always be present in the published catalog, even on a cold graph where they
  have no members yet.
- ``STRUCTURAL_ROOTS`` -- the scaffolding above the domain roots. Never graftable
  parents and never published to the catalog.
"""

from __future__ import annotations

GRAFTABLE_REALMS = frozenset({"entity", "concept", "process"})
STRUCTURAL_ROOTS = frozenset({"node", "root"})
