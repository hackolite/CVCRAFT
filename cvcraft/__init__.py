"""CVCRAFT core package."""

from .editor import cut_blocks, fuse_conv_bn_silu, prune_block, replace_block, set_blocks_frozen
from .exporter import export_scene_yaml
from .onnx_exporter import export_scene_onnx
from .onnx_importer import import_onnx_scene
from .optimization import (
    dequantize_scene,
    freeze_component,
    quantize_scene,
    set_component_lr_scale,
    set_edge_constraints,
    set_lr_scale,
    set_pretrained_config,
)
from .pytorch_exporter import export_scene_pytorch
from .scheduler import schedule_scene_stages
from .validator import SceneValidationError, validate_scene

__all__ = [
    "SceneValidationError",
    "validate_scene",
    "cut_blocks",
    "prune_block",
    "replace_block",
    "fuse_conv_bn_silu",
    "set_blocks_frozen",
    "export_scene_yaml",
    "export_scene_onnx",
    "export_scene_pytorch",
    "import_onnx_scene",
    "schedule_scene_stages",
    "quantize_scene",
    "dequantize_scene",
    "set_lr_scale",
    "set_component_lr_scale",
    "freeze_component",
    "set_edge_constraints",
    "set_pretrained_config",
]
