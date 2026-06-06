"""Voxel scene validation."""

from __future__ import annotations

from .constants import SUPPORTED_FAMILIES


class SceneValidationError(ValueError):
    pass


def _require_keys(data: dict, keys: set[str], where: str) -> None:
    missing = keys.difference(data.keys())
    if missing:
        raise SceneValidationError(f"{where} missing keys: {sorted(missing)}")


def validate_scene(scene: dict) -> None:
    _require_keys(scene, {"scene", "model", "blocks", "edges", "metrics"}, "root")
    model = scene["model"]
    _require_keys(model, {"name", "family", "anchorFree", "inputShape"}, "model")

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
