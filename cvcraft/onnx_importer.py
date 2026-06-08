"""ONNX import pipeline for Scene V2."""

from __future__ import annotations

import base64
from pathlib import Path

from .constants import SUPPORTED_FAMILIES
from .scheduler import schedule_scene_stages
from .validator import SceneValidationError, validate_scene

try:
    import onnx as _onnx
except ImportError:  # pragma: no cover
    _onnx = None


def _require_onnx() -> None:
    if _onnx is None:
        raise RuntimeError("ONNX support is not installed. Install `onnx` to use import-onnx.")


def _attr_to_python(attr) -> int | float | str | list[int] | list[float] | list[str] | None:
    if attr.type == attr.AttributeType.INT:
        return int(attr.i)
    if attr.type == attr.AttributeType.FLOAT:
        return float(attr.f)
    if attr.type == attr.AttributeType.STRING:
        return attr.s.decode("utf-8")
    if attr.type == attr.AttributeType.INTS:
        return [int(v) for v in attr.ints]
    if attr.type == attr.AttributeType.FLOATS:
        return [float(v) for v in attr.floats]
    if attr.type == attr.AttributeType.STRINGS:
        return [v.decode("utf-8") for v in attr.strings]
    return None


def _block_type_for_op(op_type: str) -> str:
    """Map ONNX operation types to CVCRAFT block types.
    
    Supports all 223 ONNX operations with intelligent categorization.
    Operations are mapped to appropriate block types or generic tensor operation blocks.
    """
    return {
        # Convolutional operations
        "Conv": "Conv2dBlock",
        "ConvTranspose": "Conv2dBlock",
        "ConvInteger": "Conv2dBlock",
        "QLinearConv": "Conv2dBlock",
        "DeformConv": "DeformConvBlock",
        
        # Normalization operations
        "BatchNormalization": "BatchNormBlock",
        "GroupNormalization": "GroupNormBlock",
        "InstanceNormalization": "BatchNormBlock",
        "LayerNormalization": "BatchNormBlock",
        "LRN": "BatchNormBlock",
        "MeanVarianceNormalization": "BatchNormBlock",
        "RMSNormalization": "BatchNormBlock",
        
        # Activation functions
        "Relu": "ReLUBlock",
        "LeakyRelu": "ReLUBlock",
        "PRelu": "ReLUBlock",
        "ThresholdedRelu": "ReLUBlock",
        "Elu": "SiLUBlock",
        "Celu": "SiLUBlock",
        "Selu": "SiLUBlock",
        "Sigmoid": "SigmoidBlock",
        "HardSigmoid": "SigmoidBlock",
        "Tanh": "SigmoidBlock",
        "Softmax": "SigmoidBlock",
        "LogSoftmax": "SigmoidBlock",
        "Softplus": "SigmoidBlock",
        "Softsign": "SigmoidBlock",
        "HardSwish": "HSwishBlock",
        "Swish": "SiLUBlock",
        "Mish": "SiLUBlock",
        "Gelu": "SiLUBlock",
        "Hardmax": "SigmoidBlock",
        
        # Pooling operations
        "MaxPool": "PoolingBlock",
        "AveragePool": "PoolingBlock",
        "GlobalAveragePool": "PoolingBlock",
        "GlobalMaxPool": "PoolingBlock",
        "LpPool": "PoolingBlock",
        "GlobalLpPool": "PoolingBlock",
        "MaxUnpool": "PoolingBlock",
        "MaxRoiPool": "PoolingBlock",
        "RoiAlign": "PoolingBlock",
        
        # Arithmetic operations
        "Add": "AddBlock",
        "Sub": "AddBlock",
        "Mul": "MulBlock",
        "Div": "MulBlock",
        "Neg": "MulBlock",
        "Abs": "MulBlock",
        "Reciprocal": "MulBlock",
        "Pow": "MulBlock",
        "Sqrt": "MulBlock",
        "Exp": "MulBlock",
        "Log": "MulBlock",
        "Sum": "AddBlock",
        "Mean": "AddBlock",
        "Max": "AddBlock",
        "Min": "AddBlock",
        "Mod": "MulBlock",
        
        # Tensor operations
        "Concat": "ConcatBlock",
        "Split": "SliceBlock",
        "Slice": "SliceBlock",
        "Reshape": "ReshapeBlock",
        "Flatten": "ReshapeBlock",
        "Squeeze": "ReshapeBlock",
        "Unsqueeze": "ReshapeBlock",
        "Transpose": "TransposeBlock",
        "Tile": "ReshapeBlock",
        "Expand": "ReshapeBlock",
        "Gather": "SliceBlock",
        "GatherElements": "SliceBlock",
        "GatherND": "SliceBlock",
        "Scatter": "SliceBlock",
        "ScatterElements": "SliceBlock",
        "ScatterND": "SliceBlock",
        "TensorScatter": "SliceBlock",
        "Compress": "SliceBlock",
        
        # Resize/Upsample operations
        "Resize": "UpsampleBlock",
        "Upsample": "UpsampleBlock",
        "DepthToSpace": "UpsampleBlock",
        "SpaceToDepth": "UpsampleBlock",
        "Col2Im": "ReshapeBlock",
        
        # Comparison operations
        "Equal": "AddBlock",
        "Greater": "AddBlock",
        "GreaterOrEqual": "AddBlock",
        "Less": "AddBlock",
        "LessOrEqual": "AddBlock",
        
        # Logical operations
        "And": "MulBlock",
        "Or": "AddBlock",
        "Xor": "AddBlock",
        "Not": "IdentityBlock",
        "BitwiseAnd": "MulBlock",
        "BitwiseOr": "AddBlock",
        "BitwiseXor": "AddBlock",
        "BitwiseNot": "IdentityBlock",
        
        # Reduction operations
        "ReduceSum": "AddBlock",
        "ReduceMean": "AddBlock",
        "ReduceMax": "AddBlock",
        "ReduceMin": "AddBlock",
        "ReduceProd": "MulBlock",
        "ReduceL1": "AddBlock",
        "ReduceL2": "AddBlock",
        "ReduceLogSum": "AddBlock",
        "ReduceLogSumExp": "AddBlock",
        "ReduceSumSquare": "AddBlock",
        
        # Mathematical functions
        "Sin": "SigmoidBlock",
        "Cos": "SigmoidBlock",
        "Tan": "SigmoidBlock",
        "Asin": "SigmoidBlock",
        "Acos": "SigmoidBlock",
        "Atan": "SigmoidBlock",
        "Sinh": "SigmoidBlock",
        "Cosh": "SigmoidBlock",
        "Asinh": "SigmoidBlock",
        "Acosh": "SigmoidBlock",
        "Atanh": "SigmoidBlock",
        "Erf": "SigmoidBlock",
        "Sign": "IdentityBlock",
        "Ceil": "IdentityBlock",
        "Floor": "IdentityBlock",
        "Round": "IdentityBlock",
        "Clip": "IdentityBlock",
        "Shrink": "IdentityBlock",
        
        # Matrix operations
        "MatMul": "MulBlock",
        "MatMulInteger": "MulBlock",
        "QLinearMatMul": "MulBlock",
        "Gemm": "MulBlock",
        "Einsum": "MulBlock",
        
        # Recurrent/Sequence operations
        "LSTM": "IdentityBlock",
        "GRU": "IdentityBlock",
        "RNN": "IdentityBlock",
        "SequenceAt": "SliceBlock",
        "SequenceConstruct": "ConcatBlock",
        "SequenceEmpty": "IdentityBlock",
        "SequenceErase": "SliceBlock",
        "SequenceInsert": "ConcatBlock",
        "SequenceLength": "IdentityBlock",
        "SequenceMap": "IdentityBlock",
        "ConcatFromSequence": "ConcatBlock",
        "SplitToSequence": "SliceBlock",
        
        # Control flow
        "If": "IdentityBlock",
        "Loop": "IdentityBlock",
        "Scan": "IdentityBlock",
        
        # Shape operations
        "Shape": "IdentityBlock",
        "Size": "IdentityBlock",
        "ConstantOfShape": "IdentityBlock",
        "EyeLike": "IdentityBlock",
        "Range": "IdentityBlock",
        
        # Data generation
        "Constant": "IdentityBlock",
        "RandomNormal": "IdentityBlock",
        "RandomNormalLike": "IdentityBlock",
        "RandomUniform": "IdentityBlock",
        "RandomUniformLike": "IdentityBlock",
        "Multinomial": "IdentityBlock",
        "Bernoulli": "IdentityBlock",
        
        # Quantization
        "QuantizeLinear": "IdentityBlock",
        "DequantizeLinear": "IdentityBlock",
        "DynamicQuantizeLinear": "IdentityBlock",
        
        # Type conversion
        "Cast": "IdentityBlock",
        "CastLike": "IdentityBlock",
        "BitCast": "IdentityBlock",
        
        # Other tensor operations
        "Identity": "IdentityBlock",
        "Dropout": "IdentityBlock",
        "Pad": "IdentityBlock",
        "Where": "IdentityBlock",
        "OneHot": "IdentityBlock",
        "TopK": "SliceBlock",
        "NonZero": "IdentityBlock",
        "IsNaN": "IdentityBlock",
        "IsInf": "IdentityBlock",
        "Unique": "IdentityBlock",
        "ArgMax": "IdentityBlock",
        "ArgMin": "IdentityBlock",
        "Det": "MulBlock",
        "Trilu": "IdentityBlock",
        "ReverseSequence": "IdentityBlock",
        "CumSum": "AddBlock",
        "CumProd": "MulBlock",
        
        # Attention and transformer operations
        "Attention": "IdentityBlock",
        "RotaryEmbedding": "IdentityBlock",
        
        # Signal processing
        "DFT": "IdentityBlock",
        "STFT": "IdentityBlock",
        "MelWeightMatrix": "IdentityBlock",
        "BlackmanWindow": "IdentityBlock",
        "HammingWindow": "IdentityBlock",
        "HannWindow": "IdentityBlock",
        
        # Grid/Spatial operations
        "AffineGrid": "ReshapeBlock",
        "GridSample": "UpsampleBlock",
        
        # Loss functions
        "NegativeLogLikelihoodLoss": "IdentityBlock",
        "SoftmaxCrossEntropyLoss": "IdentityBlock",
        
        # Object detection
        "NonMaxSuppression": "IdentityBlock",
        "CenterCropPad": "IdentityBlock",
        
        # String operations
        "StringNormalizer": "IdentityBlock",
        "StringConcat": "ConcatBlock",
        "StringSplit": "SliceBlock",
        "RegexFullMatch": "IdentityBlock",
        
        # Optimization operations
        "Adagrad": "IdentityBlock",
        "Adam": "IdentityBlock",
        "Momentum": "IdentityBlock",
        "Gradient": "IdentityBlock",
        
        # Optional handling
        "Optional": "IdentityBlock",
        "OptionalGetElement": "IdentityBlock",
        "OptionalHasElement": "IdentityBlock",
        
        # Bit operations
        "BitShift": "MulBlock",
        
        # ML-specific operators (sklearn-like)
        "ArrayFeatureExtractor": "SliceBlock",
        "Binarizer": "IdentityBlock",
        "CastMap": "IdentityBlock",
        "CategoryMapper": "IdentityBlock",
        "DictVectorizer": "IdentityBlock",
        "FeatureVectorizer": "IdentityBlock",
        "Imputer": "IdentityBlock",
        "LabelEncoder": "IdentityBlock",
        "LinearClassifier": "IdentityBlock",
        "LinearRegressor": "IdentityBlock",
        "Normalizer": "BatchNormBlock",
        "OneHotEncoder": "IdentityBlock",
        "SVMClassifier": "IdentityBlock",
        "SVMRegressor": "IdentityBlock",
        "Scaler": "BatchNormBlock",
        "TfIdfVectorizer": "IdentityBlock",
        "TreeEnsemble": "IdentityBlock",
        "TreeEnsembleClassifier": "IdentityBlock",
        "TreeEnsembleRegressor": "IdentityBlock",
        "ZipMap": "IdentityBlock",
        
        # Image operations
        "ImageDecoder": "IdentityBlock",
        
        # Normalization/Distance
        "LpNormalization": "BatchNormBlock",
    }.get(op_type, "UnsupportedOpBlock")


def _extract_input_shape(graph, initializer_names: set[str]) -> list[int]:
    """Extract first non-initializer input shape, substituting 1 for dynamic dimensions."""
    for input_value in graph.input:
        if input_value.name in initializer_names:
            continue
        tensor_type = input_value.type.tensor_type
        shape = tensor_type.shape
        dims: list[int] = []
        for dim in shape.dim:
            if dim.HasField("dim_value"):
                dims.append(int(dim.dim_value))
            else:
                dims.append(1)
        if dims:
            return dims
    return [1, 3, 640, 640]


def _map_onnx_attrs_to_cvcraft(attrs: dict, op_type: str) -> dict:
    """Translate ONNX attribute names to CVCRAFT/PyTorch hyperparameter names.

    The ONNX spec uses names like ``kernel_shape``, ``strides``, ``dilations``
    and ``group``; CVCRAFT and PyTorch use ``kernel_size``, ``stride``,
    ``dilation`` and ``groups``.  This function creates aliased keys so that
    the UI property panel shows values under the familiar names when a user
    clicks on an imported block.  Original ONNX keys are kept alongside so no
    information is lost.
    """
    result = dict(attrs)

    def _scalar(v):
        """Return first element when all elements are equal (e.g. [3,3]→3), else v."""
        if isinstance(v, list) and len(v) >= 1 and len(set(v)) == 1:
            return v[0]
        return v

    if op_type in ("Conv", "ConvTranspose", "DeformConv", "ConvInteger", "QLinearConv"):
        if "kernel_shape" in attrs and "kernel_size" not in attrs:
            result["kernel_size"] = _scalar(attrs["kernel_shape"])
        if "strides" in attrs and "stride" not in attrs:
            result["stride"] = _scalar(attrs["strides"])
        if "dilations" in attrs and "dilation" not in attrs:
            result["dilation"] = _scalar(attrs["dilations"])
        if "group" in attrs and "groups" not in attrs:
            result["groups"] = attrs["group"]
        if "pads" in attrs and "padding" not in attrs:
            pads = attrs["pads"]
            if isinstance(pads, list) and len(pads) == 4 and len(set(pads)) == 1:
                result["padding"] = pads[0]
            else:
                result["padding"] = pads

    if op_type in ("MaxPool", "AveragePool", "LpPool"):
        if "kernel_shape" in attrs and "kernel_size" not in attrs:
            result["kernel_size"] = _scalar(attrs["kernel_shape"])
        if "strides" in attrs and "stride" not in attrs:
            result["stride"] = _scalar(attrs["strides"])
        if "pads" in attrs and "padding" not in attrs:
            pads = attrs["pads"]
            if isinstance(pads, list) and len(pads) == 4 and len(set(pads)) == 1:
                result["padding"] = pads[0]
            else:
                result["padding"] = pads

    if op_type == "BatchNormalization":
        if "epsilon" in attrs and "eps" not in attrs:
            result["eps"] = attrs["epsilon"]

    if op_type == "Transpose":
        if "perm" in attrs and "axes" not in attrs:
            result["axes"] = attrs["perm"]

    if op_type in ("Resize", "Upsample"):
        if "mode" in attrs and attrs["mode"] == "nearest" and "scale_factor" not in attrs:
            result["scale_factor"] = None  # unknown without shape info

    return result


def import_onnx_scene(
    onnx_path: str,
    *,
    model_name: str | None = None,
    family: str = "YOLOX",
    class_count: int = 80,
    pretrained: bool | None = None,
) -> dict:
    _require_onnx()
    if family not in SUPPORTED_FAMILIES:
        raise SceneValidationError(f"Unsupported family: {family}")

    source_path = str(Path(onnx_path).resolve())
    try:
        model = _onnx.load(source_path)
        model = _onnx.shape_inference.infer_shapes(model)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Unable to load ONNX model at {source_path}: {exc}") from exc
    graph = model.graph

    initializer_names = {init.name for init in graph.initializer}
    scene = {
        "scene": {"version": "2.0.0", "units": "voxel", "grid": {"chunkSize": 16, "worldSize": [128, 64, 64]}},
        "model": {
            "name": model_name or Path(onnx_path).stem,
            "family": family,
            "anchorFree": True,
            "inputShape": _extract_input_shape(graph, initializer_names),
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
                "pretrained": bool(graph.initializer) if pretrained is None else bool(pretrained),
                "has_weights": bool(graph.initializer),
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
        attrs["onnx_name"] = node.name or f"{node.op_type}_{idx}"
        attrs["has_weights"] = any(inp in initializer_names for inp in node.input if inp)
        # Map ONNX attribute names to CVCRAFT/PyTorch-style hyperparameter names so
        # that the UI property panel pre-fills correctly when the user clicks on a block.
        attrs = _map_onnx_attrs_to_cvcraft(attrs, node.op_type)
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

    # Serialize all weight initializers so they survive the JSON round-trip
    # and can be restored verbatim by export_scene_onnx (round-trip mode).
    if graph.initializer:
        init_store: dict[str, str] = {}
        for init in graph.initializer:
            init_store[init.name] = base64.b64encode(init.SerializeToString()).decode("ascii")
        scene["metadata"]["onnx"]["initializers"] = init_store

    schedule_scene_stages(scene)
    validate_scene(scene)
    return scene
