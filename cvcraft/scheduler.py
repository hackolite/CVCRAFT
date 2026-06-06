"""Deterministic 3D stage scheduler."""

from __future__ import annotations

from collections import defaultdict

from .constants import STAGE_ORDER
from .scene_v2 import normalize_scene_v2


def schedule_scene_stages(scene: dict) -> None:
    normalize_scene_v2(scene)

    stage_offset = {stage: idx * 10 for idx, stage in enumerate(STAGE_ORDER)}
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

        x = stage_offset.get(stage, 10) + stage_local_x[stage] * 2
        y = level
        z = branch_index
        stage_local_x[stage] += 1
        block["position"] = {"x": x, "y": y, "z": z}

    normalize_scene_v2(scene)
