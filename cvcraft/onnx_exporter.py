"""ONNX export pipeline from Scene V2 IR."""

from __future__ import annotations

import json
from pathlib import Path

from .constants import STAGE_ORDER
from .scene_v2 import normalize_scene_v2

try:
    import onnx as _onnx
    from onnx import TensorProto, helper as _helper
except ImportError:  # pragma: no cover
    _onnx = None
    _helper = None
    TensorProto = None


def _require_onnx() -> None:
    if _onnx is None:
        raise RuntimeError("ONNX support is not installed. Install `onnx` to use export-onnx.")


def _onnx_op_for_block(block_type: str) -> str:
    """Map CVCRAFT block types back to ONNX operation types."""
    return {
        "Conv2dBlock": "Conv",
        "DWConvBlock": "Conv",
        "PWConvBlock": "Conv",
        "FusedConvBlock": "Conv",
        "DeformConvBlock": "Conv",
        "BatchNormBlock": "BatchNormalization",
        "GroupNormBlock": "GroupNormalization",
        "SyncBNBlock": "BatchNormalization",
        "ReLUBlock": "Relu",
        "SiLUBlock": "Relu",  # Approximation: SiLU requires composite op, use Relu as safe fallback
        "HSwishBlock": "HardSwish",
        "SigmoidBlock": "Sigmoid",
        "PoolingBlock": "MaxPool",
        "UpsampleBlock": "Resize",
        "AddBlock": "Add",
        "MulBlock": "Mul",
        "ConcatBlock": "Concat",
        "ReshapeBlock": "Reshape",
        "TransposeBlock": "Transpose",
        "SliceBlock": "Slice",
        "IdentityBlock": "Identity",
        "ShapeAdapterBlock": "Identity",
        "FPNBlock": "Identity",
        "PANBlock": "Identity",
        "BiFPNBlock": "Identity",
        "DecoupledHeadBlock": "Identity",
        "ClsHeadBlock": "Identity",
        "RegHeadBlock": "Identity",
        "CenterHeadBlock": "Identity",
        "DFLBlock": "Identity",
        "DistributionProjectBlock": "Identity",
        "NMSFreeDecodeBlock": "Identity",
        "ClassificationHead": "Identity",
        "RegressionHead": "Identity",
        "CenterHeatmapHead": "Identity",
        "ObjectnessHead": "Identity",
        "CSPBlock": "Identity",
        "SPPBlock": "Identity",
        "SPPFBlock": "Identity",
        "FocusBlock": "Identity",
        "GhostBlock": "Conv",
        "MBConv": "Conv",
        "ResidualBlock": "Add",
        "DropPathBlock": "Identity",
        "QuantStubBlock": "QuantizeLinear",
        "DeQuantStubBlock": "DequantizeLinear",
        "StageContainer": "Identity",
    }.get(block_type, "Identity")


def export_scene_onnx(scene: dict, *, opset_version: int = 13) -> bytes:
    """Export a Scene V2 IR to ONNX binary format.

    Returns the serialized ONNX model as bytes.
    """
    _require_onnx()
    normalize_scene_v2(scene)

    model_info = scene["model"]
    input_shape = model_info.get("inputShape", [1, 3, 640, 640])
    topo_order = scene["canonicalGraph"]["topoOrder"]
    block_by_id = {b["id"]: b for b in scene["blocks"]}

    # Build tensor names for edges
    edge_tensors: dict[tuple[str, str], str] = {}
    for edge in scene["edges"]:
        tensor_name = edge.get("tensor") or f"{edge['from']}_to_{edge['to']}"
        edge_tensors[(edge["from"], edge["to"])] = tensor_name

    # Track output tensor for each block
    block_output_tensor: dict[str, str] = {}
    nodes = []
    tensor_idx = 0

    for block_id in topo_order:
        block = block_by_id[block_id]
        block_type = block["type"]

        if block_type == "InputBlock":
            # Input blocks produce the graph input tensor
            out_tensors = block["io"].get("out", [])
            block_output_tensor[block_id] = out_tensors[0] if out_tensors else "input_0"
            continue

        if block_type == "OutputBlock":
            # Output blocks consume their input directly
            in_tensors = block["io"].get("in", [])
            if in_tensors:
                block_output_tensor[block_id] = in_tensors[0]
            continue

        # Determine input tensors for this node
        incoming_edges = [e for e in scene["edges"] if e["to"] == block_id]
        node_inputs = []
        for e in incoming_edges:
            src = e["from"]
            if src in block_output_tensor:
                node_inputs.append(block_output_tensor[src])

        # Define output tensor
        output_name = f"t_{tensor_idx}"
        tensor_idx += 1
        block_output_tensor[block_id] = output_name

        op_type = _onnx_op_for_block(block_type)
        node = _helper.make_node(
            op_type,
            inputs=node_inputs if node_inputs else ["input_0"],
            outputs=[output_name],
            name=block_id,
        )
        nodes.append(node)

    # Determine graph input
    graph_input = _helper.make_tensor_value_info(
        "input_0", TensorProto.FLOAT, input_shape
    )

    # Determine graph output (last block output)
    output_tensor_name = "input_0"
    for block_id in reversed(topo_order):
        block = block_by_id[block_id]
        if block["type"] == "OutputBlock":
            in_tensors = block["io"].get("in", [])
            if in_tensors and in_tensors[0] in block_output_tensor.values():
                output_tensor_name = block_output_tensor.get(block_id, output_tensor_name)
                break
        if block_id in block_output_tensor and block["type"] != "InputBlock":
            output_tensor_name = block_output_tensor[block_id]
            break

    graph_output = _helper.make_tensor_value_info(
        output_tensor_name, TensorProto.FLOAT, None
    )

    graph = _helper.make_graph(
        nodes,
        name=model_info.get("name", "cvcraft_model"),
        inputs=[graph_input],
        outputs=[graph_output],
    )

    onnx_model = _helper.make_model(graph, opset_imports=[_helper.make_opsetid("", opset_version)])
    onnx_model.producer_name = "cvcraft"
    onnx_model.ir_version = 7

    return onnx_model.SerializeToString()
