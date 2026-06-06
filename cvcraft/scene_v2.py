"""Scene V2 normalization and canonical graph utilities."""

from __future__ import annotations

import math
import re
from collections import defaultdict

from .constants import DETECTION_BLOCKS, NECK_BLOCKS


def infer_stage_id(block: dict) -> str:
    block_type = block.get("type")
    if block_type == "InputBlock":
        return "input"
    if block_type == "OutputBlock":
        return "output"
    if block_type in NECK_BLOCKS:
        return "neck"
    if block_type in DETECTION_BLOCKS:
        return "head"
    return "backbone"


def _extract_level(token: str) -> int | None:
    m = re.search(r"[pPfF](\d+)", token)
    if m:
        return int(m.group(1))
    return None


def infer_resolution_level(block: dict) -> int:
    params = block.get("params", {})
    for key in ("level", "resolution", "scale"):
        val = params.get(key)
        if isinstance(val, str):
            lvl = _extract_level(val)
            if lvl is not None:
                return lvl
        if isinstance(val, int):
            return val

    strides = params.get("strides")
    if isinstance(strides, list) and strides and isinstance(strides[0], int) and strides[0] > 0:
        return max(0, int(math.log2(strides[0])) if strides[0] > 1 else 0)

    io = block.get("io", {})
    for name in list(io.get("in", [])) + list(io.get("out", [])):
        if not isinstance(name, str):
            continue
        lvl = _extract_level(name)
        if lvl is not None:
            return lvl
    return 0


def _topological_order(scene: dict) -> list[str]:
    edges = scene.get("edges", [])
    ids = [b["id"] for b in scene.get("blocks", [])]
    indegree = {bid: 0 for bid in ids}
    outgoing: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        src = edge["from"]
        dst = edge["to"]
        if src in indegree and dst in indegree:
            outgoing[src].append(dst)
            indegree[dst] += 1

    ready = sorted([bid for bid, d in indegree.items() if d == 0])
    ordered: list[str] = []
    while ready:
        current = ready.pop(0)
        ordered.append(current)
        for nxt in sorted(outgoing[current]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
                ready.sort()

    if len(ordered) != len(ids):
        return sorted(ids)
    return ordered


def normalize_scene_v2(scene: dict) -> None:
    scene.setdefault("schemaVersion", "2.0.0")
    metadata = scene.setdefault("metadata", {})
    metadata.setdefault("onnx", {})
    metadata.setdefault("topology", {"deterministic": True})

    block_by_id = {b["id"]: b for b in scene.get("blocks", [])}
    topo_order = _topological_order(scene)
    for topo_index, block_id in enumerate(topo_order):
        block = block_by_id[block_id]
        meta = block.setdefault("meta", {})
        meta.setdefault("stage_id", infer_stage_id(block))
        meta.setdefault("resolution_level", infer_resolution_level(block))
        meta["topo_index"] = topo_index

    canonical_nodes = []
    for block_id in topo_order:
        block = block_by_id[block_id]
        canonical_nodes.append(
            {
                "id": block["id"],
                "type": block["type"],
                "stage_id": block["meta"]["stage_id"],
                "resolution_level": block["meta"]["resolution_level"],
                "topo_index": block["meta"]["topo_index"],
            }
        )

    canonical_edges = sorted(
        [
            {"from": e["from"], "to": e["to"], "tensor": e.get("tensor")}
            for e in scene.get("edges", [])
        ],
        key=lambda e: (e["from"], e["to"], e.get("tensor") or ""),
    )
    scene["canonicalGraph"] = {
        "nodes": canonical_nodes,
        "edges": canonical_edges,
        "topoOrder": topo_order,
    }
