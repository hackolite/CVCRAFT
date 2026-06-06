"""Deterministic 3D stage scheduler."""

from __future__ import annotations

from collections import defaultdict

from .constants import STAGE_ORDER
from .scene_v2 import normalize_scene_v2

DEFAULT_STAGE_OFFSET = 10
BLOCK_SPACING = 2


def schedule_scene_stages(scene: dict) -> None:
    normalize_scene_v2(scene)

    # Output at the bottom (y=0), grow upward: reverse stage order for height
    num_stages = len(STAGE_ORDER)
    # output=0, head=1, neck=2, backbone=3, input=4 (bottom to top)
    stage_height = {stage: (num_stages - 1 - idx) * DEFAULT_STAGE_OFFSET for idx, stage in enumerate(STAGE_ORDER)}

    ids_by_topo = scene["canonicalGraph"]["topoOrder"]
    blocks = {b["id"]: b for b in scene["blocks"]}
    stage_level_counters: dict[tuple[str, int], int] = defaultdict(int)
    stage_local_x: dict[str, int] = defaultdict(int)

    for block_id in ids_by_topo:
        block = blocks[block_id]
        meta = block["meta"]
        stage = meta["stage_id"]
        level = int(meta["resolution_level"])
        branch_index = stage_level_counters[(stage, level)]
        stage_level_counters[(stage, level)] += 1

        # x: spread blocks within same stage horizontally, centered around 0
        x = stage_local_x[stage] * BLOCK_SPACING
        # y: height based on stage (output=0 at bottom, input at top)
        y = stage_height.get(stage, 0)
        # z: branch offset for same level blocks
        z = branch_index
        stage_local_x[stage] += 1
        block["position"] = {"x": x, "y": y, "z": z}

    # Center x positions around 0 for each stage
    for stage in STAGE_ORDER:
        count = stage_local_x[stage]
        if count > 0:
            offset = (count - 1) * BLOCK_SPACING / 2.0
            for block in blocks.values():
                if block["meta"]["stage_id"] == stage:
                    block["position"]["x"] -= offset

    normalize_scene_v2(scene)
