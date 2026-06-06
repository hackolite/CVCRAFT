"""YAML export utilities."""

from __future__ import annotations

from .constants import DETECTION_BLOCKS
from .scene_v2 import normalize_scene_v2


def _yaml_scalar(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        if value == "" or any(ch in value for ch in ":#{}[],'\""):
            return f'"{value}"'
        return value
    return str(value)


def _dump_yaml(obj, indent: int = 0) -> list[str]:
    pad = " " * indent
    if isinstance(obj, dict):
        lines = []
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{pad}{k}:")
                lines.extend(_dump_yaml(v, indent + 2))
            else:
                lines.append(f"{pad}{k}: {_yaml_scalar(v)}")
        return lines
    if isinstance(obj, list):
        lines = []
        for item in obj:
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}-")
                lines.extend(_dump_yaml(item, indent + 2))
            else:
                lines.append(f"{pad}- {_yaml_scalar(item)}")
        return lines
    return [f"{pad}{_yaml_scalar(obj)}"]


def export_scene_yaml(scene: dict) -> str:
    normalize_scene_v2(scene)
    model = scene["model"]
    blocks_by_id = {b["id"]: b for b in scene["blocks"]}
    ordered_blocks = [blocks_by_id[bid] for bid in scene["canonicalGraph"]["topoOrder"]]
    backbone = []
    neck = []
    head = []
    for b in ordered_blocks:
        entry = {"type": b["type"], **b.get("params", {})}
        if b["type"] in {"FPNBlock", "PANBlock", "BiFPNBlock", "ShapeAdapterBlock"}:
            neck.append(entry)
        elif b["type"] in DETECTION_BLOCKS:
            head.append(entry)
        elif b["type"] not in {"InputBlock", "OutputBlock", "StageContainer"}:
            backbone.append(entry)

    payload = {
        "model": {
            "name": model["name"],
            "family": model["family"],
            "anchor_free": model["anchorFree"],
            "input_shape": model["inputShape"],
            "classes": model.get("classCount", 1),
        },
        "backbone": backbone,
        "neck": neck,
        "head": head,
        "export": {"targets": ["yaml", "pytorch", "onnx_compatible"]},
        "metrics": {
            "flops": scene["metrics"]["flops"],
            "parameters": scene["metrics"]["parameters"],
            "latency_ms": scene["metrics"]["latencyMs"],
        },
    }
    return "\n".join(_dump_yaml(payload)) + "\n"
