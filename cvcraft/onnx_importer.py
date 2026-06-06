"""ONNX import pipeline for Scene V2."""

from __future__ import annotations

from pathlib import Path

from .constants import SUPPORTED_FAMILIES
from .scheduler import schedule_scene_stages
from .validator import SceneValidationError, validate_scene

try:
    import onnx as _onnx
except ImportError:  # pragma: no cover - covered by runtime branch
    _onnx = None


def _require_onnx() -> None:
    if _onnx is None:
        raise RuntimeError("ONNX support is not installed. Install `onnx` to use import-onnx.")


def _attr_to_python(attr) -> int | float | str | list[int] | list[float] | list[str] | None:
    if hasattr(attr, "i") and attr.type == attr.AttributeType.INT:
        return int(attr.i)
    if hasattr(attr, "f") and attr.type == attr.AttributeType.FLOAT:
        return float(attr.f)
    if hasattr(attr, "s") and attr.type == attr.AttributeType.STRING:
        return attr.s.decode("utf-8")
    if hasattr(attr, "ints") and attr.type == attr.AttributeType.INTS:
        return [int(v) for v in attr.ints]
    if hasattr(attr, "floats") and attr.type == attr.AttributeType.FLOATS:
        return [float(v) for v in attr.floats]
    if hasattr(attr, "strings") and attr.type == attr.AttributeType.STRINGS:
        return [v.decode("utf-8") for v in attr.strings]
    return None


def _block_type_for_op(op_type: str) -> str:
    return {
        "Conv": "Conv2dBlock",
        "BatchNormalization": "BatchNormBlock",
        "Relu": "ReLUBlock",
        "Sigmoid": "SigmoidBlock",
        "Add": "AddBlock",
        "Mul": "MulBlock",
        "Concat": "ConcatBlock",
        "Slice": "SliceBlock",
        "Reshape": "ReshapeBlock",
        "Transpose": "TransposeBlock",
        "Resize": "UpsampleBlock",
        "Upsample": "UpsampleBlock",
        "MaxPool": "PoolingBlock",
        "AveragePool": "PoolingBlock",
        "GlobalAveragePool": "PoolingBlock",
        "Identity": "IdentityBlock",
    }.get(op_type, "UnsupportedOpBlock")


def import_onnx_scene(
    onnx_path: str,
    *,
    model_name: str | None = None,
    family: str = "YOLOX",
    class_count: int = 80,
) -> dict:
    _require_onnx()
    if family not in SUPPORTED_FAMILIES:
        raise SceneValidationError(f"Unsupported family: {family}")

    source_path = str(Path(onnx_path).resolve())
    model = _onnx.load(source_path)
    model = _onnx.shape_inference.infer_shapes(model)
    graph = model.graph

    initializer_names = {init.name for init in graph.initializer}
    scene = {
        "scene": {"version": "2.0.0", "units": "voxel", "grid": {"chunkSize": 16, "worldSize": [128, 64, 64]}},
        "model": {
            "name": model_name or Path(onnx_path).stem,
            "family": family,
            "anchorFree": True,
            "inputShape": [1, 3, 640, 640],
            "classCount": class_count,
        },
        "blocks": [],
        "edges": [],
        "metrics": {"flops": 0.0, "parameters": 0, "latencyMs": {}},
        "metadata": {
            "onnx": {
                "source_path": source_path,
                "ir_version": int(model.ir_version),
                "producer_name": model.producer_name,
                "opsets": [{"domain": op.domain, "version": int(op.version)} for op in model.opset_import],
            }
        },
    }

    tensor_producer: dict[str, str] = {}
    input_idx = 0
    for input_value in graph.input:
        if input_value.name in initializer_names:
            continue
        block_id = f"in_{input_idx}"
        input_idx += 1
        scene["blocks"].append(
            {
                "id": block_id,
                "type": "InputBlock",
                "position": {"x": 0, "y": 0, "z": 0},
                "params": {},
                "io": {"in": [], "out": [input_value.name]},
            }
        )
        tensor_producer[input_value.name] = block_id

    for idx, node in enumerate(graph.node):
        block_id = f"n_{idx}"
        attrs = {}
        for attr in node.attribute:
            attrs[attr.name] = _attr_to_python(attr)
        attrs["onnx_op"] = node.op_type
        attrs["onnx_name"] = node.name or block_id
        scene["blocks"].append(
            {
                "id": block_id,
                "type": _block_type_for_op(node.op_type),
                "position": {"x": 0, "y": 0, "z": 0},
                "params": attrs,
                "io": {"in": [x for x in node.input if x], "out": [x for x in node.output if x]},
            }
        )
        for input_tensor in node.input:
            if not input_tensor or input_tensor in initializer_names:
                continue
            producer = tensor_producer.get(input_tensor)
            if producer:
                scene["edges"].append({"from": producer, "to": block_id, "tensor": input_tensor})
        for output_tensor in node.output:
            if output_tensor:
                tensor_producer[output_tensor] = block_id

    output_idx = 0
    for output_value in graph.output:
        producer = tensor_producer.get(output_value.name)
        if not producer:
            continue
        block_id = f"out_{output_idx}"
        output_idx += 1
        scene["blocks"].append(
            {
                "id": block_id,
                "type": "OutputBlock",
                "position": {"x": 0, "y": 0, "z": 0},
                "params": {},
                "io": {"in": [output_value.name], "out": []},
            }
        )
        scene["edges"].append({"from": producer, "to": block_id, "tensor": output_value.name})

    schedule_scene_stages(scene)
    validate_scene(scene)
    return scene
