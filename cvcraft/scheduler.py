"""Deterministic 3D stage scheduler — vertical tracks layout."""

from __future__ import annotations

from collections import defaultdict

from .constants import STAGE_ORDER
from .scene_v2 import normalize_scene_v2

STAGE_SPACING_Y = 8  # Vertical distance between stages
BLOCK_SPACING_Y = 1.5  # Vertical spacing between blocks within same stage track
TRACK_SPACING_X = 3.5  # Horizontal spacing between parallel tracks (branches)


def _build_adjacency(scene: dict) -> tuple[dict, dict]:
    """Build adjacency maps from edges."""
    children: dict[str, list[str]] = defaultdict(list)
    parents: dict[str, list[str]] = defaultdict(list)
    for edge in scene.get("edges", []):
        children[edge["from"]].append(edge["to"])
        parents[edge["to"]].append(edge["from"])
    return children, parents


def schedule_scene_stages(scene: dict) -> None:
    normalize_scene_v2(scene)

    ids_by_topo = scene["canonicalGraph"]["topoOrder"]
    blocks = {b["id"]: b for b in scene["blocks"]}
    children, parents = _build_adjacency(scene)

    # Assign each block a depth (longest path from any root)
    depth: dict[str, int] = {}
    for block_id in ids_by_topo:
        parent_depths = [depth[p] for p in parents[block_id] if p in depth]
        depth[block_id] = (max(parent_depths) + 1) if parent_depths else 0

    # Assign tracks (x positions) — each branch fork gets a new track
    # Group blocks by stage, then within a stage assign tracks based on connectivity
    stage_blocks: dict[str, list[str]] = defaultdict(list)
    for block_id in ids_by_topo:
        block = blocks[block_id]
        stage = block["meta"]["stage_id"]
        stage_blocks[stage].append(block_id)

    # Determine track assignment per block using a simple rule:
    # - Follow the "main" path (first child) on track 0
    # - Branches get assigned to increasing track indices
    track: dict[str, int] = {}
    max_track = 0

    # For each stage, process blocks in topo order and assign tracks
    for stage in STAGE_ORDER:
        stage_ids = stage_blocks.get(stage, [])
        if not stage_ids:
            continue

        # Track assignment: inherit from parent, or use next available for branches
        next_track = 0
        for block_id in stage_ids:
            parent_ids = [p for p in parents[block_id] if p in track]
            if parent_ids:
                # Inherit track from first parent in same stage or closest parent
                same_stage_parents = [p for p in parent_ids if blocks[p]["meta"]["stage_id"] == stage]
                if same_stage_parents:
                    track[block_id] = track[same_stage_parents[0]]
                else:
                    # Parent in different stage: check if multiple children create a fork
                    parent_id = parent_ids[0]
                    parent_children_in_stage = [c for c in children[parent_id] if c in blocks and blocks[c]["meta"]["stage_id"] == stage]
                    if len(parent_children_in_stage) > 1:
                        idx = parent_children_in_stage.index(block_id)
                        if idx == 0:
                            track[block_id] = track.get(parent_id, 0)
                        else:
                            next_track = max(next_track, max_track + 1)
                            track[block_id] = next_track
                            max_track = next_track
                    else:
                        track[block_id] = track.get(parent_id, 0)
            else:
                track[block_id] = next_track

            max_track = max(max_track, track[block_id])

    # Track the y-position per track column (next available y slot)
    track_y_cursor: dict[int, float] = defaultdict(float)

    # Process in stage order (top-to-bottom: input at top, output at bottom)
    num_stages = len(STAGE_ORDER)
    stage_base_y = {}
    for idx, stage in enumerate(STAGE_ORDER):
        stage_base_y[stage] = (num_stages - 1 - idx) * STAGE_SPACING_Y

    # Initialize track cursors at stage tops
    for stage in STAGE_ORDER:
        stage_ids = stage_blocks.get(stage, [])
        for block_id in stage_ids:
            t = track.get(block_id, 0)
            if t not in track_y_cursor:
                track_y_cursor[t] = stage_base_y[stage]

    # Assign positions: process in topo order for correct vertical ordering
    block_y: dict[str, float] = {}
    current_stage_y: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))

    for stage in STAGE_ORDER:
        base = stage_base_y[stage]
        stage_ids = stage_blocks.get(stage, [])
        for i, block_id in enumerate(stage_ids):
            t = track.get(block_id, 0)
            y = base - current_stage_y[stage][t]
            current_stage_y[stage][t] += BLOCK_SPACING_Y
            block_y[block_id] = y

    # Center tracks around x=0
    all_tracks = set(track.values())
    num_tracks = len(all_tracks)
    track_list = sorted(all_tracks)
    track_x = {t: (i - (num_tracks - 1) / 2.0) * TRACK_SPACING_X for i, t in enumerate(track_list)}

    # Apply positions
    for block_id, block in blocks.items():
        t = track.get(block_id, 0)
        x = track_x.get(t, 0)
        y = block_y.get(block_id, 0)
        z = 0  # Keep flat in z for clean vertical view
        block["position"] = {"x": x, "y": y, "z": z}

    normalize_scene_v2(scene)
