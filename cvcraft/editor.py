"""Editing operations with lightweight auto-repair."""

from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy


def _index_blocks(scene: dict) -> dict[str, dict]:
    return {b["id"]: b for b in scene["blocks"]}


def _remove_dead_nodes(scene: dict) -> None:
    blocks = _index_blocks(scene)
    out_edges = defaultdict(list)
    in_degree = defaultdict(int)
    for e in scene["edges"]:
        out_edges[e["from"]].append(e["to"])
        in_degree[e["to"]] += 1

    roots = [b["id"] for b in scene["blocks"] if b["type"] == "InputBlock" or in_degree[b["id"]] == 0]
    q = deque(roots)
    seen = set(roots)
    while q:
        node = q.popleft()
        for nxt in out_edges[node]:
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)

    scene["blocks"] = [b for b in scene["blocks"] if b["id"] in seen]
    valid_ids = {b["id"] for b in scene["blocks"]}
    scene["edges"] = [e for e in scene["edges"] if e["from"] in valid_ids and e["to"] in valid_ids]


def _auto_repair(scene: dict) -> None:
    blocks = _index_blocks(scene)
    block_inputs = {bid: set(blocks[bid]["io"].get("in", [])) for bid in blocks}
    adapters = []
    new_edges = []
    adapter_idx = 0
    for e in scene["edges"]:
        tensor = e.get("tensor")
        consumer_inputs = block_inputs.get(e["to"], set())
        if tensor and consumer_inputs and tensor not in consumer_inputs:
            repaired_tensor = sorted(consumer_inputs)[0]
            adapter_id = f"adapter_{adapter_idx}"
            adapter_idx += 1
            adapters.append(
                {
                    "id": adapter_id,
                    "type": "ShapeAdapterBlock",
                    "position": {"x": 0, "y": 0, "z": 0},
                    "params": {"mode": "auto_repair"},
                    "io": {"in": [tensor], "out": [repaired_tensor]},
                }
            )
            new_edges.append({"from": e["from"], "to": adapter_id, "tensor": tensor})
            new_edges.append({"from": adapter_id, "to": e["to"], "tensor": repaired_tensor})
        else:
            new_edges.append(e)
    scene["blocks"].extend(adapters)
    scene["edges"] = new_edges
    _remove_dead_nodes(scene)


def cut_blocks(scene: dict, block_ids: set[str]) -> dict:
    before = deepcopy(scene)
    scene["blocks"] = [b for b in scene["blocks"] if b["id"] not in block_ids]
    scene["edges"] = [e for e in scene["edges"] if e["from"] not in block_ids and e["to"] not in block_ids]
    _auto_repair(scene)
    return {
        "removed": sorted(block_ids),
        "updated": [],
        "inserted": [b for b in scene["blocks"] if b["id"] not in {x["id"] for x in before["blocks"]}],
    }


def prune_block(scene: dict, block_id: str, param_name: str, new_value: int) -> dict:
    for b in scene["blocks"]:
        if b["id"] == block_id:
            b["params"][param_name] = new_value
            return {"updated": [{"id": block_id, "params": {param_name: new_value}}]}
    raise KeyError(f"Unknown block id: {block_id}")


def replace_block(scene: dict, block_id: str, new_type: str, params: dict | None = None) -> dict:
    for b in scene["blocks"]:
        if b["id"] == block_id:
            b["type"] = new_type
            if params:
                b["params"].update(params)
            _auto_repair(scene)
            return {"updated": [{"id": block_id, "type": new_type, "params": b['params']}]}
    raise KeyError(f"Unknown block id: {block_id}")


def fuse_conv_bn_silu(scene: dict) -> dict:
    blocks = _index_blocks(scene)
    outgoing = defaultdict(list)
    incoming = defaultdict(list)
    for e in scene["edges"]:
        outgoing[e["from"]].append(e)
        incoming[e["to"]].append(e)

    removed = set()
    for block in list(scene["blocks"]):
        if block["type"] != "Conv2dBlock":
            continue
        conv_id = block["id"]
        if len(outgoing[conv_id]) != 1:
            continue
        bn_id = outgoing[conv_id][0]["to"]
        bn = blocks.get(bn_id)
        if not bn or bn["type"] != "BatchNormBlock" or len(outgoing[bn_id]) != 1:
            continue
        act_id = outgoing[bn_id][0]["to"]
        act = blocks.get(act_id)
        if not act or act["type"] != "SiLUBlock":
            continue

        block["type"] = "FusedConvBlock"
        out_after_act = outgoing[act_id]
        scene["edges"] = [
            e
            for e in scene["edges"]
            if e["from"] not in {bn_id, act_id} and e["to"] not in {bn_id, act_id}
        ]
        for e in out_after_act:
            scene["edges"].append({"from": conv_id, "to": e["to"], "tensor": e.get("tensor")})
        removed.update({bn_id, act_id})

    if removed:
        scene["blocks"] = [b for b in scene["blocks"] if b["id"] not in removed]
        _auto_repair(scene)
    return {"removed": sorted(removed)}


def set_blocks_frozen(scene: dict, block_ids: set[str], frozen: bool) -> dict:
    updated = []
    for block in scene["blocks"]:
        if block["id"] in block_ids:
            meta = block.setdefault("meta", {})
            meta["frozen"] = frozen
            updated.append({"id": block["id"], "frozen": frozen})
    if not updated:
        raise ValueError(f"No matching blocks found for ids: {sorted(block_ids)}")
    return {"updated": updated}


# Block types that represent a network output (used by add_block_to_scene)
_OUTPUT_BLOCK_TYPES = frozenset({"OutputBlock", "Output"})


def _unique_block_id(scene: dict, block_type: str) -> str:
    """Return a unique block ID derived from the block type."""
    existing = {b["id"] for b in scene["blocks"]}
    # Strip trailing 'block'/'Block' suffix safely so 'FooBlock' → 'foo', not 'foo_'
    base_raw = block_type.lower()
    if base_raw.endswith("block"):
        base_raw = base_raw[:-5]
    base = base_raw.strip("_") or "block"
    n = len(scene["blocks"]) + 1
    candidate = f"{base}_{n}"
    while candidate in existing:
        n += 1
        candidate = f"{base}_{n}"
    return candidate


def add_block_to_scene(scene: dict, block_type: str, params: dict,
                       anchor_id: str | None = None) -> dict:
    """Append a new block to the scene, optionally after *anchor_id*.

    If *anchor_id* is given the new block is inserted immediately after it in
    the blocks list and the edges that previously left *anchor_id* are
    re-routed through the new block.  When *anchor_id* is ``None`` the block
    is appended before the first ``OutputBlock`` (or at the very end when no
    output block exists).
    """
    block_id = _unique_block_id(scene, block_type)
    new_block: dict = {
        "id": block_id,
        "type": block_type,
        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
        "params": dict(params or {}),
        "io": {"in": [], "out": []},
        "meta": {"stage_id": "backbone", "resolution_level": 0, "topo_index": 0},
    }

    if anchor_id:
        anchor_idx = next(
            (i for i, b in enumerate(scene["blocks"]) if b["id"] == anchor_id),
            None,
        )
        if anchor_idx is None:
            raise KeyError(f"Unknown anchor block id: {anchor_id}")
        scene["blocks"].insert(anchor_idx + 1, new_block)
        # Re-route outgoing edges from anchor through the new block
        old_out = [e for e in scene["edges"] if e["from"] == anchor_id]
        scene["edges"] = [e for e in scene["edges"] if e["from"] != anchor_id]
        scene["edges"].append({"from": anchor_id, "to": block_id})
        for oe in old_out:
            scene["edges"].append({"from": block_id, "to": oe["to"]})
    else:
        # Insert before the first output block, or at the end
        output_ids = {b["id"] for b in scene["blocks"]
                      if b["type"] in _OUTPUT_BLOCK_TYPES}
        non_outputs = [b for b in scene["blocks"] if b["id"] not in output_ids]
        if output_ids and non_outputs:
            last = non_outputs[-1]
            last_idx = next(i for i, b in enumerate(scene["blocks"])
                            if b["id"] == last["id"])
            scene["blocks"].insert(last_idx + 1, new_block)
            # Re-route last→output edges through new block
            rerouted = [e for e in scene["edges"]
                        if e["from"] == last["id"] and e["to"] in output_ids]
            scene["edges"] = [e for e in scene["edges"]
                               if not (e["from"] == last["id"]
                                       and e["to"] in output_ids)]
            scene["edges"].append({"from": last["id"], "to": block_id})
            for oe in rerouted:
                scene["edges"].append({"from": block_id, "to": oe["to"]})
        else:
            scene["blocks"].append(new_block)
            if len(scene["blocks"]) > 1:
                prev = scene["blocks"][-2]
                scene["edges"].append({"from": prev["id"], "to": block_id})

    return {"inserted": [block_id], "updated": []}


def insert_block_on_edge(scene: dict, block_type: str, params: dict,
                         from_id: str, to_id: str) -> dict:
    """Insert a new block between two directly connected blocks.

    The existing edge *from_id* → *to_id* is replaced by two new edges:
    *from_id* → new_block and new_block → *to_id*.  The new block is
    positioned at the midpoint of the two endpoint blocks.
    """
    edge = next(
        (e for e in scene["edges"] if e["from"] == from_id and e["to"] == to_id),
        None,
    )
    if edge is None:
        raise KeyError(f"No edge from '{from_id}' to '{to_id}' found in scene")

    block_id = _unique_block_id(scene, block_type)

    # Place the block at the midpoint between the two endpoints
    blocks_by_id = {b["id"]: b for b in scene["blocks"]}
    from_pos = (blocks_by_id.get(from_id) or {}).get("position", {"x": 0.0, "y": 0.0, "z": 0.0})
    to_pos = (blocks_by_id.get(to_id) or {}).get("position", {"x": 0.0, "y": 0.0, "z": 0.0})
    mid: dict = {
        "x": (from_pos["x"] + to_pos["x"]) / 2.0,
        "y": (from_pos["y"] + to_pos["y"]) / 2.0,
        "z": (from_pos["z"] + to_pos["z"]) / 2.0,
    }

    new_block: dict = {
        "id": block_id,
        "type": block_type,
        "position": mid,
        "params": dict(params or {}),
        "io": {"in": [], "out": []},
        "meta": {"stage_id": "backbone", "resolution_level": 0, "topo_index": 0},
    }

    # Insert the block right after from_id in the blocks list
    from_idx = next(
        (i for i, b in enumerate(scene["blocks"]) if b["id"] == from_id),
        len(scene["blocks"]) - 1,
    )
    scene["blocks"].insert(from_idx + 1, new_block)

    # Swap the old edge with two new edges
    old_tensor = edge.get("tensor")
    scene["edges"] = [e for e in scene["edges"]
                      if not (e["from"] == from_id and e["to"] == to_id)]
    scene["edges"].append({"from": from_id, "to": block_id,
                            **({"tensor": old_tensor} if old_tensor else {})})
    scene["edges"].append({"from": block_id, "to": to_id})

    return {"inserted": [block_id], "updated": []}
