"""Unit tests for full-scan per-type error isolation (greptile #149).

``_run_full_type_emergence_scan`` runs emergence over every type definition and
must isolate each type so a transient failure on one doesn't abort the rest of
the periodic scan.
"""

from __future__ import annotations

from sophia.api.app import _run_full_type_emergence_scan


def test_full_scan_isolates_failing_type():
    """One failing type must not abort emergence over the remaining types."""

    class FakeHCG:
        def get_all_type_definitions(self):
            return [{"uuid": "type_a"}, {"uuid": "type_b"}, {"uuid": "type_c"}]

    ran = []

    def run_one(type_uuid):
        if type_uuid == "type_b":
            raise RuntimeError("transient HCG write error")
        ran.append(type_uuid)

    _run_full_type_emergence_scan(FakeHCG(), run_one)

    # type_b raised, but type_a and type_c still ran.
    assert ran == ["type_a", "type_c"]


def test_full_scan_skips_blank_uuids():
    """Type definitions without a uuid are skipped, not dispatched."""

    class FakeHCG:
        def get_all_type_definitions(self):
            return [{"uuid": ""}, {"uuid": "type_x"}, {}]

    ran = []
    _run_full_type_emergence_scan(FakeHCG(), ran.append)
    assert ran == ["type_x"]


def test_full_scan_aborts_quietly_when_listing_fails():
    """A failure listing the type definitions aborts without raising."""

    class FakeHCG:
        def get_all_type_definitions(self):
            raise RuntimeError("neo4j down")

    ran = []
    _run_full_type_emergence_scan(FakeHCG(), ran.append)
    assert ran == []
