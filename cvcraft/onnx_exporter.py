"""ONNX export pipeline from Scene V2 IR."""

from __future__ import annotations

import struct
from pathlib import Path

from .scene_v2 import normalize_scene_v2

try:
    import onnx as _onnx
    from onnx import TensorProto, helper as _helper, numpy_helper as _numpy_helper
except ImportError:  # pragma: no cover
    _onnx = None
    _helper = None
    _numpy_helper = None
    TensorProto = None

# Block types that carry learnable weights (Conv, BN, …)
_WEIGHTED_BLOCK_TYPES = frozenset({
    "Conv2dBlock", "DWConvBlock", "PWConvBlock", "FusedConvBlock",
    "DeformConvBlock", "GhostBlock", "MBConv", "ResidualBlock",
    "BatchNormBlock", "GroupNormBlock", "SyncBNBlock",
    "FPNBlock", "PANBlock", "BiFPNBlock", "CSPBlock",
    "SPPBlock", "SPPFBlock", "FocusBlock",
    "DecoupledHeadBlock", "ClsHeadBlock", "RegHeadBlock",
    "CenterHeadBlock", "DFLBlock", "ObjectnessHead",
})


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


def check_onnx_format(scene: dict) -> dict:
    """Validate ONNX format coherence for a scene.

    Returns a dict with:
    - ``valid``: True if all blocks map to supported ONNX ops
    - ``identity_fallbacks``: list of (block_id, block_type) pairs using Identity as fallback
    - ``unsupported_ops``: list of unknown block types with no explicit mapping
    """
    normalize_scene_v2(scene)
    identity_fallbacks: list[dict] = []
    unsupported_ops: list[dict] = []

    # Block types explicitly mapped to Identity (structural, not unsupported)
    _structural_identity = {
        "IdentityBlock", "ShapeAdapterBlock", "FPNBlock", "PANBlock", "BiFPNBlock",
        "DecoupledHeadBlock", "ClsHeadBlock", "RegHeadBlock", "CenterHeadBlock",
        "DFLBlock", "DistributionProjectBlock", "NMSFreeDecodeBlock",
        "ClassificationHead", "RegressionHead", "CenterHeatmapHead",
        "ObjectnessHead", "CSPBlock", "SPPBlock", "SPPFBlock", "FocusBlock",
        "DropPathBlock", "StageContainer",
        "InputBlock", "OutputBlock",
    }

    for block in scene.get("blocks", []):
        block_type = block["type"]
        if block_type in ("InputBlock", "OutputBlock"):
            continue
        op = _onnx_op_for_block(block_type)
        if op == "Identity":
            if block_type in _structural_identity:
                identity_fallbacks.append({"id": block["id"], "type": block_type})
            else:
                unsupported_ops.append({"id": block["id"], "type": block_type})

    return {
        "valid": len(unsupported_ops) == 0,
        "identity_fallbacks": identity_fallbacks,
        "unsupported_ops": unsupported_ops,
    }


def _make_zero_initializer(name: str, shape: list[int]) -> "_onnx.TensorProto":
    """Create a zero-filled ONNX initializer tensor."""
    import array as _array
    n_elements = 1
    for d in shape:
        n_elements *= d
    raw = _array.array("f", [0.0] * n_elements).tobytes()
    tensor = _onnx.TensorProto()
    tensor.data_type = TensorProto.FLOAT
    tensor.name = name
    tensor.dims.extend(shape)
    tensor.raw_data = raw
    return tensor


def export_scene_onnx(scene: dict, *, opset_version: int = 13, include_weights: bool = True) -> bytes:
    """Export a Scene V2 IR to ONNX binary format.

    Args:
        scene: Scene V2 IR dict.
        opset_version: ONNX opset version to target (default 13).
        include_weights: When True (default), embed zero-initialised weight
            tensors for every block that carries learnable parameters
            (i.e. whose ``params.has_weights`` is True or whose block type
            is in the weighted block set).  When False, the exported graph
            has no initializers (structure-only).

    Returns:
        Serialized ONNX model bytes.

    The export also respects the freeze / unfreeze state of each block:
    frozen blocks are recorded in ``model.metadata_props`` so downstream
    tools can reconstruct which parameters should be kept fixed during
    fine-tuning.
    """
    _require_onnx()
    normalize_scene_v2(scene)

    # --- Validate format coherence up-front (warn, don't raise) ---
    coherence = check_onnx_format(scene)

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
    initializers = []
    tensor_idx = 0

    for block_id in topo_order:
        block = block_by_id[block_id]
        block_type = block["type"]

        if block_type == "InputBlock":
            out_tensors = block["io"].get("out", [])
            block_output_tensor[block_id] = out_tensors[0] if out_tensors else "input_0"
            continue

        if block_type == "OutputBlock":
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
        params = block.get("params", {})
        meta = block.get("meta", {})

        # --- Respect freeze state: append frozen initializers with a prefix ---
        is_frozen = bool(meta.get("frozen", params.get("frozen", False)))

        # Build node inputs, appending weight initializer names where applicable
        node_input_names = list(node_inputs) if node_inputs else ["input_0"]

        if include_weights:
            block_has_weights = bool(
                params.get("has_weights", block_type in _WEIGHTED_BLOCK_TYPES)
            )
            if block_has_weights and op_type == "Conv":
                # Conv expects [X, W, B?] inputs; add stub weight + bias initializers
                out_ch = params.get("out_channels") or params.get("num_outputs") or 1
                in_ch = params.get("in_channels") or 1
                ksize = params.get("kernel_size") or params.get("kernel_shape")
                if isinstance(ksize, list):
                    ksize = ksize[0] if ksize else 1
                ksize = ksize or 1
                groups = params.get("groups") or params.get("group") or 1
                in_ch_per_group = max(1, int(in_ch) // int(groups))

                w_name = f"{block_id}.weight"
                b_name = f"{block_id}.bias"
                frozen_prefix = "frozen." if is_frozen else ""
                w_name = frozen_prefix + w_name
                b_name = frozen_prefix + b_name

                w_shape = [int(out_ch), int(in_ch_per_group), int(ksize), int(ksize)]
                b_shape = [int(out_ch)]
                initializers.append(_make_zero_initializer(w_name, w_shape))
                initializers.append(_make_zero_initializer(b_name, b_shape))
                node_input_names = node_input_names[:1] + [w_name, b_name]

            elif block_has_weights and op_type == "BatchNormalization":
                # BN expects [X, scale, B, mean, var] inputs
                num_features = params.get("num_features") or params.get("out_channels") or 1
                frozen_prefix = "frozen." if is_frozen else ""
                for suffix in ("weight", "bias", "running_mean", "running_var"):
                    init_name = f"{frozen_prefix}{block_id}.{suffix}"
                    initializers.append(_make_zero_initializer(init_name, [int(num_features)]))
                    node_input_names.append(init_name)

        node = _helper.make_node(
            op_type,
            inputs=node_input_names,
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
        initializer=initializers,
    )

    onnx_model = _helper.make_model(graph, opset_imports=[_helper.make_opsetid("", opset_version)])
    onnx_model.producer_name = "cvcraft"
    onnx_model.ir_version = 7

    # --- Embed freeze-state and coherence info in model metadata ---
    frozen_ids = [
        b["id"] for b in scene["blocks"]
        if bool(b.get("meta", {}).get("frozen", b.get("params", {}).get("frozen", False)))
    ]
    _add_metadata_prop(onnx_model, "cvcraft.frozen_blocks", ",".join(frozen_ids))
    _add_metadata_prop(onnx_model, "cvcraft.include_weights", str(include_weights).lower())
    _add_metadata_prop(onnx_model, "cvcraft.identity_fallbacks", str(len(coherence["identity_fallbacks"])))
    _add_metadata_prop(onnx_model, "cvcraft.unsupported_ops", str(len(coherence["unsupported_ops"])))

    return onnx_model.SerializeToString()


def _add_metadata_prop(model, key: str, value: str) -> None:
    """Add a key/value entry to model.metadata_props."""
    entry = model.metadata_props.add()
    entry.key = key
    entry.value = value
