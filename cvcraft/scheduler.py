"""Deterministic 3D scheduler — Netron-style vertical DAG layout.

Produces a top-to-bottom topological layout similar to Netron:
- Blocks are placed in rows by their rank (longest path from roots).
- Within each rank, blocks are centered horizontally.
- Forks spread children horizontally; merges reconverge.
- Connections flow downward with minimal crossings.
"""

from __future__ import annotations

from collections import defaultdict

from .scene_v2 import normalize_scene_v2

# Layout tuning constants (Netron-like compact spacing)
ROW_SPACING_Y = 2.0  # Vertical distance between consecutive ranks
COL_SPACING_X = 3.0  # Horizontal distance between nodes in same rank


def _build_adjacency(scene: dict) -> tuple[dict, dict]:
    """Build adjacency maps from edges."""
    children: dict[str, list[str]] = defaultdict(list)
    parents: dict[str, list[str]] = defaultdict(list)
    for edge in scene.get("edges", []):
        children[edge["from"]].append(edge["to"])
        parents[edge["to"]].append(edge["from"])
    return children, parents


def schedule_scene_stages(scene: dict) -> None:
    """Layout blocks in Netron-style top-to-bottom DAG.

    Algorithm:
    1. Compute rank = longest path from any root for each node.
    2. Assign horizontal positions to minimize edge crossings
       using the barycenter heuristic.
    3. Center each rank around x=0.
    """
    normalize_scene_v2(scene)

    ids_by_topo = scene["canonicalGraph"]["topoOrder"]
    blocks = {b["id"]: b for b in scene["blocks"]}
    children, parents = _build_adjacency(scene)

    # --- Step 1: Assign ranks (longest path from roots) ---
    rank: dict[str, int] = {}
    for block_id in ids_by_topo:
        parent_ranks = [rank[p] for p in parents[block_id] if p in rank]
        rank[block_id] = (max(parent_ranks) + 1) if parent_ranks else 0

    # Group blocks by rank
    ranks: dict[int, list[str]] = defaultdict(list)
    for block_id in ids_by_topo:
        ranks[rank[block_id]].append(block_id)

    max_rank = max(ranks.keys()) if ranks else 0

    # --- Step 2: Initial ordering within each rank ---
    # Start with topological order within each rank, then refine
    # using barycenter heuristic (average x of parents).
    order_in_rank: dict[str, int] = {}

    # Initialize: order by appearance in topo sort
    for r in range(max_rank + 1):
        for idx, block_id in enumerate(ranks[r]):
            order_in_rank[block_id] = idx

    # Barycenter heuristic: sweep down then up (2 passes)
    for _sweep in range(4):
        # Forward sweep (top to bottom): order by average parent position
        for r in range(1, max_rank + 1):
            barycenters: list[tuple[float, str]] = []
            for block_id in ranks[r]:
                parent_ids = [p for p in parents[block_id] if p in order_in_rank]
                if parent_ids:
                    avg = sum(order_in_rank[p] for p in parent_ids) / len(parent_ids)
                else:
                    avg = order_in_rank.get(block_id, 0)
                barycenters.append((avg, block_id))
            barycenters.sort(key=lambda x: x[0])
            ranks[r] = [bid for _, bid in barycenters]
            for idx, block_id in enumerate(ranks[r]):
                order_in_rank[block_id] = idx

        # Backward sweep (bottom to top): order by average child position
        for r in range(max_rank - 1, -1, -1):
            barycenters = []
            for block_id in ranks[r]:
                child_ids = [c for c in children[block_id] if c in order_in_rank]
                if child_ids:
                    avg = sum(order_in_rank[c] for c in child_ids) / len(child_ids)
                else:
                    avg = order_in_rank.get(block_id, 0)
                barycenters.append((avg, block_id))
            barycenters.sort(key=lambda x: x[0])
            ranks[r] = [bid for _, bid in barycenters]
            for idx, block_id in enumerate(ranks[r]):
                order_in_rank[block_id] = idx

    # --- Step 3: Assign x,y positions ---
    for r in range(max_rank + 1):
        row_blocks = ranks[r]
        count = len(row_blocks)
        for idx, block_id in enumerate(row_blocks):
            # Center the row around x=0
            x = (idx - (count - 1) / 2.0) * COL_SPACING_X
            # Y goes downward: rank 0 at top (highest y), increasing rank → lower y
            y = -r * ROW_SPACING_Y
            blocks[block_id]["position"] = {"x": x, "y": y, "z": 0}

    normalize_scene_v2(scene)
