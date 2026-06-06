"""Scene V2 normalization and canonical graph utilities."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from heapq import heapify, heappop, heappush

from .constants import DETECTION_BLOCKS, NECK_BLOCKS, STAGE_COLORS

HEAD_HINT_TOKENS = ("head", "detect", "cls", "reg", "bbox", "dfl", "pred")
NECK_HINT_TOKENS = ("neck", "fpn", "pan", "bifpn", "pafpn", "lateral", "upsample")


def infer_stage_id(block: dict) -> str:
    """Infer high-level stage lane from a block type."""
    block_type = block.get("type")
    if block_type == "InputBlock":
        return "input"
    if block_type == "OutputBlock":
        return "output"
    if block_type in NECK_BLOCKS:
        return "neck"
    if block_type in DETECTION_BLOCKS:
        return "head"
    params = block.get("params", {})
    tokens = []
    for key in ("onnx_name", "onnx_op", "stage_hint"):
        value = params.get(key)
        if isinstance(value, str):
            tokens.append(value.lower())
    io = block.get("io", {})
    for name in io.get("in", []) + io.get("out", []):
        if isinstance(name, str):
            tokens.append(name.lower())
    hint_text = " ".join(tokens)
    if any(token in hint_text for token in HEAD_HINT_TOKENS):
        return "head"
    if any(token in hint_text for token in NECK_HINT_TOKENS):
        return "neck"
    return "backbone"


def _extract_level(token: str) -> int | None:
    m = re.search(r"[pPfF](\d+)", token)
    if m:
        return int(m.group(1))
    return None


def infer_resolution_level(block: dict) -> int:
    """Infer resolution level using params, stride, and IO tensor naming fallbacks."""
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
    if isinstance(strides, list) and strides and all(isinstance(s, int) for s in strides) and strides[0] > 0:
        stride = strides[0]
        # Power-of-two stride maps directly to feature pyramid levels through log2(stride).
        if stride > 1 and (stride & (stride - 1)) == 0:
            return int(math.log2(stride))
        return stride

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

    ready = [bid for bid, d in indegree.items() if d == 0]
    ready.sort()
    heapify(ready)
    ordered: list[str] = []
    while ready:
        current = heappop(ready)
        ordered.append(current)
        for nxt in sorted(outgoing[current]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                heappush(ready, nxt)

    if len(ordered) != len(ids):
        raise ValueError("Graph contains cycles and cannot produce deterministic topological order")
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
        meta["stage_id"] = infer_stage_id(block)
        meta.setdefault("resolution_level", infer_resolution_level(block))
        meta["color"] = STAGE_COLORS.get(meta["stage_id"], STAGE_COLORS["backbone"])
        meta.setdefault("frozen", bool(block.get("params", {}).get("frozen", False)))
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

    canonical_edges = []
    for edge in scene.get("edges", []):
        edge_entry = {"from": edge["from"], "to": edge["to"]}
        tensor = edge.get("tensor")
        if tensor is not None:
            edge_entry["tensor"] = tensor
        canonical_edges.append(edge_entry)
    canonical_edges.sort(key=lambda e: (e["from"], e["to"], e.get("tensor") or ""))
    scene["canonicalGraph"] = {
        "nodes": canonical_nodes,
        "edges": canonical_edges,
        "topoOrder": topo_order,
    }
