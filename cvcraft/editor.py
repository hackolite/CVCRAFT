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
