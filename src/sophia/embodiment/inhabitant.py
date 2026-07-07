from __future__ import annotations

from typing import Any, cast


class Inhabitant:
    """Minimal embodied Sophia: attaches to a body, perceives odom, emits simple cmd_vel.

    Open-loop by design — proves inhabitation + faithful interpretation, NOT navigation
    competence. Routed straight to the embodiment, not through the stub Planner/Executor.
    """

    def __init__(self, embodiment: Any) -> None:
        self._emb = embodiment
        self._emb.describe().actuator("cmd_vel")  # bind; raises if not advertised

    def perceive(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._emb.read().obs)

    def act(self, scripted_step: int) -> dict[str, Any]:
        # Scripted, open-loop: forward for the first stretch, then a steady yaw.
        cmd = {"cmd_vel": [0.8, 0.0]} if scripted_step < 10 else {"cmd_vel": [0.4, 0.6]}
        self._emb.command(cmd)
        return cmd


def run(
    host: str = "127.0.0.1", port: int = 57610, steps: int = 200
) -> None:  # pragma: no cover
    import time

    from talos.embodiment.socket_embodiment import SocketEmbodiment

    emb = SocketEmbodiment(host, port)
    emb.connect()
    inh = Inhabitant(emb)
    try:
        for i in range(steps):
            inh.act(i)
            time.sleep(0.05)
    finally:
        emb.close()
