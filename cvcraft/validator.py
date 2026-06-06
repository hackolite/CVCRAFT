"""Voxel scene validation."""

from __future__ import annotations

from .constants import SUPPORTED_FAMILIES
from .scene_v2 import normalize_scene_v2

VALID_TASK = "object_detection"
VALID_DETECTION_TYPE = "anchor_free"


class SceneValidationError(ValueError):
    pass


def _require_keys(data: dict, keys: set[str], where: str) -> None:
    missing = keys.difference(data.keys())
    if missing:
        raise SceneValidationError(f"{where} missing keys: {sorted(missing)}")


def validate_scene(scene: dict) -> None:
    normalize_scene_v2(scene)
    _require_keys(scene, {"scene", "model", "blocks", "edges", "metrics"}, "root")
    model = scene["model"]
    _require_keys(model, {"name", "family", "anchorFree", "inputShape"}, "model")

    # Ensure task and detection_type fields exist with correct values
    model.setdefault("task", VALID_TASK)
    model.setdefault("detection_type", VALID_DETECTION_TYPE)

    if model["task"] != VALID_TASK:
        raise SceneValidationError(f"model.task must be '{VALID_TASK}', got '{model['task']}'")
    if model["detection_type"] != VALID_DETECTION_TYPE:
        raise SceneValidationError(f"model.detection_type must be '{VALID_DETECTION_TYPE}', got '{model['detection_type']}'")

    if model["family"] not in SUPPORTED_FAMILIES:
        raise SceneValidationError(f"Unsupported family: {model['family']}")
    if model["anchorFree"] is not True:
        raise SceneValidationError("model.anchorFree must be true")

    blocks = scene["blocks"]
    if not isinstance(blocks, list) or not blocks:
        raise SceneValidationError("blocks must be a non-empty list")
    ids = [b.get("id") for b in blocks]
    if len(ids) != len(set(ids)):
        raise SceneValidationError("Block ids must be unique")

    id_set = set(ids)
    for b in blocks:
        _require_keys(b, {"id", "type", "position", "params", "io"}, f"block {b.get('id')}")
        _require_keys(b["io"], {"in", "out"}, f"block {b.get('id')}.io")
        _require_keys(b, {"meta"}, f"block {b.get('id')}")
        _require_keys(b["meta"], {"stage_id", "resolution_level", "topo_index"}, f"block {b.get('id')}.meta")
        if not isinstance(b["meta"]["stage_id"], str):
            raise SceneValidationError(f"block {b.get('id')}.meta.stage_id must be a string")
        if not isinstance(b["meta"]["resolution_level"], int):
            raise SceneValidationError(f"block {b.get('id')}.meta.resolution_level must be an int")
        if not isinstance(b["meta"]["topo_index"], int):
            raise SceneValidationError(f"block {b.get('id')}.meta.topo_index must be an int")

    edges = scene["edges"]
    if not isinstance(edges, list):
        raise SceneValidationError("edges must be a list")
    for i, e in enumerate(edges):
        _require_keys(e, {"from", "to"}, f"edge[{i}]")
        if e["from"] not in id_set or e["to"] not in id_set:
            raise SceneValidationError(f"edge[{i}] references unknown node")
        tensor = e.get("tensor")
        if tensor is not None and not isinstance(tensor, str):
            raise SceneValidationError(f"edge[{i}].tensor must be string when provided")

    _require_keys(scene["metrics"], {"flops", "parameters", "latencyMs"}, "metrics")
    _require_keys(scene, {"schemaVersion", "metadata", "canonicalGraph"}, "root")
    _require_keys(scene["canonicalGraph"], {"nodes", "edges", "topoOrder"}, "canonicalGraph")
    topo = scene["canonicalGraph"]["topoOrder"]
    if not isinstance(topo, list):
        raise SceneValidationError("canonicalGraph.topoOrder must be a list")
    if len(topo) != len(scene["blocks"]):
        raise SceneValidationError(
            f"canonicalGraph.topoOrder length mismatch: expected {len(scene['blocks'])} blocks, got {len(topo)}"
        )
