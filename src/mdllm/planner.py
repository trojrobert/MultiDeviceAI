"""Cluster fit planning helpers.

Turns exo placement previews into human-friendly "will it fit / which split is
best" answers.
"""

from __future__ import annotations

from .client import Placement


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:3.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def valid_placements(placements: list[Placement]) -> list[Placement]:
    return [p for p in placements if p.error is None]


def rank_placements(placements: list[Placement]) -> list[Placement]:
    """Best-first ordering.

    Heuristic for a phone+laptop setup where capacity is the goal:
      1. Prefer placements that actually fit (no error) — filtered before this.
      2. Prefer *fewer* nodes (less network hops => lower latency), because the
         value here is fitting the model, not tensor-parallel speedups.
      3. Break ties by lowest peak memory delta on the most-loaded node.
    """

    def key(p: Placement):
        max_node_mem = max(p.memory_delta_by_node.values(), default=0)
        return (len(p.nodes), max_node_mem)

    return sorted(valid_placements(placements), key=key)


def best_placement(placements: list[Placement]) -> Placement | None:
    ranked = rank_placements(placements)
    return ranked[0] if ranked else None
