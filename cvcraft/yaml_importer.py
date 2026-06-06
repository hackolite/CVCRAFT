"""YAML architecture import to Scene V2."""

from __future__ import annotations

from pathlib import Path

from .constants import SUPPORTED_FAMILIES
from .scheduler import schedule_scene_stages
from .validator import SceneValidationError, validate_scene

try:
    import yaml as _yaml
except ImportError:  # pragma: no cover
    _yaml = None


def _require_yaml() -> None:
    if _yaml is None:
        raise RuntimeError("YAML support is not installed. Install `PyYAML` to use import-yaml.")


def _validate_stage_blocks(stage_name: str, value) -> list[dict]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise SceneValidationError(f"Invalid YAML format: `{stage_name}` must be a list of blocks")
    parsed: list[dict] = []
    for idx, entry in enumerate(value):
        if not isinstance(entry, dict) or "type" not in entry:
            raise SceneValidationError(f"Invalid YAML format: `{stage_name}[{idx}]` must be a mapping with `type`")
        parsed.append(entry)
    return parsed


def _sanitize_params(raw_params: dict, stage_name: str, index: int) -> dict:
    params = {}
    for key, value in raw_params.items():
        if not isinstance(key, str):
            raise SceneValidationError(f"Invalid YAML format: `{stage_name}[{index}]` has a non-string parameter key")
        if isinstance(value, (str, int, float, bool)) or value is None:
            params[key] = value
            continue
        # Empty lists are accepted to keep parity with YAML configs where optional list params may be unset.
        if isinstance(value, list) and all(isinstance(item, (str, int, float, bool)) for item in value):
            params[key] = value
            continue
        raise SceneValidationError(
            f"Invalid YAML format: `{stage_name}[{index}].{key}` has unsupported parameter type `{type(value).__name__}`"
        )
    return params


def import_yaml_scene(yaml_path: str) -> dict:
    _require_yaml()
    source_path = str(Path(yaml_path).resolve())
    try:
        payload = _yaml.safe_load(Path(source_path).read_text(encoding="utf-8"))
    except (OSError, _yaml.YAMLError) as exc:
        raise SceneValidationError(f"Invalid YAML format: {exc}") from exc
    if not isinstance(payload, dict):
        raise SceneValidationError("Invalid YAML format: root must be a mapping")

    model = payload.get("model")
    if not isinstance(model, dict):
        raise SceneValidationError("Invalid YAML format: missing `model` mapping")

    family = model.get("family", "YOLOX")
    if family not in SUPPORTED_FAMILIES:
        raise SceneValidationError(f"Unsupported family: {family}")
    input_shape = model.get("input_shape", [1, 3, 640, 640])
    if not (isinstance(input_shape, list) and len(input_shape) == 4 and all(isinstance(v, int) for v in input_shape)):
        raise SceneValidationError("Invalid YAML format: model.input_shape must be a list of 4 integers")

    backbone = _validate_stage_blocks("backbone", payload.get("backbone"))
    neck = _validate_stage_blocks("neck", payload.get("neck"))
    head = _validate_stage_blocks("head", payload.get("head"))

    scene = {
        "scene": {"version": "2.0.0", "units": "voxel", "grid": {"chunkSize": 16, "worldSize": [128, 64, 64]}},
        "model": {
            "name": model.get("name", Path(yaml_path).stem),
            "family": family,
            "anchorFree": bool(model.get("anchor_free", True)),
            "inputShape": input_shape,
            "classCount": int(model.get("classes", 80)),
        },
        "blocks": [],
        "edges": [],
        "metrics": {"flops": 0.0, "parameters": 0, "latencyMs": {}},
        "metadata": {"yaml": {"source_path": source_path}},
    }

    scene["blocks"].append(
        {"id": "in_0", "type": "InputBlock", "position": {"x": 0, "y": 0, "z": 0}, "params": {}, "io": {"in": [], "out": ["x"]}}
    )
    previous_id = "in_0"
    previous_tensor = "x"
    counter = 0
    for stage_name, entries in (("backbone", backbone), ("neck", neck), ("head", head)):
        for stage_idx, entry in enumerate(entries):
            block_id = f"{stage_name}_{counter}"
            counter += 1
            out_tensor = f"t_{counter}"
            raw_params = {}
            for key, value in entry.items():
                if key != "type":
                    raw_params[key] = value
            params = _sanitize_params(raw_params, stage_name, stage_idx)
            scene["blocks"].append(
                {
                    "id": block_id,
                    "type": entry["type"],
                    "position": {"x": 0, "y": 0, "z": 0},
                    "params": params,
                    "io": {"in": [previous_tensor], "out": [out_tensor]},
                }
            )
            scene["edges"].append({"from": previous_id, "to": block_id, "tensor": previous_tensor})
            previous_id = block_id
            previous_tensor = out_tensor

    scene["blocks"].append(
        {
            "id": "out_0",
            "type": "OutputBlock",
            "position": {"x": 0, "y": 0, "z": 0},
            "params": {},
            "io": {"in": [previous_tensor], "out": []},
        }
    )
    scene["edges"].append({"from": previous_id, "to": "out_0", "tensor": previous_tensor})

    schedule_scene_stages(scene)
    validate_scene(scene)
    return scene
