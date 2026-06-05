"""Residual durability at eviction time (SPEC 5.13 / R8, #175).

The Hermes-flagged outlier filter in `_mint_subtree` must durably park the
members it drops: retype them to the `unsorted` sentinel via the production
update path so the next candidate pull (membership by the authoritative
`type_uuid` property) excludes them. `type_entity` is not a usable home
because emergence clusters the entity junk drawer itself.
"""

from __future__ import annotations

from collections import Counter

from sophia.maintenance.config import MaintenanceConfig
from sophia.maintenance.emergence_handler import EmergenceHandler
from sophia.maintenance.emergence_types import Member, NameResult


class _NoMatchMilvus:
    """Match-before-mint (#504) never finds an existing type."""

    def find_nearest_types(self, centroid, top_k=1):
        return []

    def get_embedding(self, node_type, uuid):
        return None


def _member(uuid, x, current_type="widgetbase"):
    return Member(
        uuid=uuid,
        name=uuid,
        embedding=[x, 0.0],
        signature=Counter({("MOVED_TO", "location"): 1}),
        current_type=current_type,
        hermes_type_hint="object",
        neighbors=[],
    )


def test_evicted_outlier_parked_and_not_repulled():
    """An evicted member is retyped onto the sentinel at eviction time, so
    the very next candidate pull over the seed excludes it and no later
    cluster re-includes it.
    """
    seed = "type_widgetbase"

    members_by_uuid = {f"w{i}": _member(f"w{i}", 0.01 * i) for i in range(4)}
    members_by_uuid.update(
        {f"g{i}": _member(f"g{i}", 9.0 + 0.01 * i) for i in range(4)}
    )

    class FakeHCG:
        """Stateful fake whose candidate selection mirrors production:
        membership is the authoritative `type_uuid` property (#505)."""

        def __init__(self):
            self.nodes = {
                seed: {
                    "uuid": seed,
                    "name": "widgetbase",
                    "properties": {"ancestors": ["root", "node", "entity"]},
                }
            }
            for u in members_by_uuid:
                self.nodes[u] = {
                    "uuid": u,
                    "properties": {"type": "widgetbase", "type_uuid": seed},
                }

        def get_node(self, uuid):
            return self.nodes.get(uuid)

        def update_node(self, uuid, props):
            self.nodes[uuid]["properties"].update(props)

        def get_nodes_by_type_uuid(self, type_uuid):
            return [
                n
                for n in self.nodes.values()
                if (n.get("properties") or {}).get("type_uuid") == type_uuid
            ]

    hcg = FakeHCG()
    pulls = []
    named_clusters = []

    def load_members(type_uuid):
        rows = hcg.get_nodes_by_type_uuid(type_uuid)
        ms = [
            members_by_uuid[r["uuid"]]
            for r in rows
            if r["uuid"] in members_by_uuid
        ]
        pulls.append({m.uuid for m in ms})
        return ms

    def fake_name(cluster, candidates, hermes_url, token):
        uuids = {m.uuid for m in cluster.members}
        named_clusters.append(uuids)
        if any(u.startswith("w") for u in uuids):
            return NameResult(
                label="widget", description="", confidence=0.9, removed=["w3"]
            )
        if any(u.startswith("g") for u in uuids):
            return NameResult(label="gadget", description="", confidence=0.9)
        return NameResult(label="fresh", description="", confidence=0.9)

    def fake_mint(cluster, name, hcg, milvus, source_cluster_id, **kwargs):
        # Mirror the production retype write (type_minting.mint_type).
        type_uuid = f"type_{name.label}"
        if kwargs.get("retype_members", True):
            for m in cluster.members:
                hcg.update_node(
                    m.uuid, {"type": name.label, "type_uuid": type_uuid}
                )
        return type_uuid

    handler = EmergenceHandler(
        config=MaintenanceConfig(),
        hcg=hcg,
        milvus=_NoMatchMilvus(),
        event_bus=None,
        hermes_url="http://h",
        token="t",
        load_members=load_members,
        name_fn=fake_name,
        mint_fn=fake_mint,
        candidates_fn=lambda: [],
    )

    handler.run(seed)

    # Sanity: the cluster minted without the evictee; keepers were retyped.
    for u in ("w0", "w1", "w2"):
        assert hcg.nodes[u]["properties"]["type_uuid"] == "type_widget"
    # THE FIX (SPEC 5.13): eviction durably parks the outlier under the
    # sentinel via the production update path, off the seed it came from.
    assert hcg.nodes["w3"]["properties"]["type_uuid"] == "type_unsorted"
    assert hcg.nodes["w3"]["properties"]["type"] == "unsorted"

    # New arrivals land on the seed between passes, right next to the
    # evictee -- without durable parking it would cluster straight back in.
    for i in range(3):
        u = f"n{i}"
        members_by_uuid[u] = _member(u, 0.01 * i)
        hcg.nodes[u] = {
            "uuid": u,
            "properties": {"type": "widgetbase", "type_uuid": seed},
        }

    named_clusters.clear()
    handler.run(seed)

    # The next candidate pull excludes the parked member entirely...
    assert "w3" not in pulls[-1]
    # ...and no cluster handed to the namer re-includes it.
    assert all("w3" not in c for c in named_clusters)


def test_run_skips_unsorted_sentinel_source():
    """The `unsorted` sentinel is never a clustering source, so parked
    residuals stay parked (SPEC 5.13)."""
    loaded = []
    handler = EmergenceHandler(
        config=MaintenanceConfig(),
        hcg=object(),
        milvus=_NoMatchMilvus(),
        event_bus=None,
        hermes_url="http://h",
        token="t",
        load_members=lambda u: loaded.append(u) or [],
        name_fn=lambda *a: None,
        mint_fn=lambda *a, **k: None,
        candidates_fn=lambda: [],
    )
    handler.run("type_unsorted")
    assert loaded == []
