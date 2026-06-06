"""Reusable hyperparameter templates for the parameter editor.

Provides a catalog of ready-to-insert sub-graphs (backbones, necks, heads)
with typical hyperparameters for anchor-free object detection workflows.

Currently supported families:

* **VGG-like backbones** — VGG-11/13/16/19 stages built from 3x3 Conv +
  BatchNorm + ReLU and 2x2 MaxPool, suitable to start hand-edited models.
* **CenterNet** — typical DLA-style upsample neck and triple head
  (heatmap / wh / offset) operating on stride-4 features.
* **FCOS** — FPN neck (P3-P7) and decoupled cls/reg/centerness head.
* **NanoDet** — PAN neck and GFL head (cls + DFL regression).
* **PicoDet** — CSP-PAN-like neck and VFL/DFL head.
* **YOLOX** — PAFPN neck and decoupled cls/obj/reg head.
* **PP-YOLOE** — CSP-PAN neck and ET-Head with VFL + DFL projection.

Templates are pure data: each template is a chain (or DAG fragment) of
blocks with sane defaults. `insert_template` splices a template into an
existing scene, optionally connecting it to a user-selected anchor block,
re-running normalization + scheduling and validating the result.
"""

from __future__ import annotations

from copy import deepcopy

from .editor import _auto_repair  # type: ignore[attr-defined]
from .scheduler import schedule_scene_stages
from .validator import SceneValidationError, validate_scene


# ---------------------------------------------------------------------------
# Helper constructors
# ---------------------------------------------------------------------------


def _conv(out_channels: int, kernel_size: int = 3, stride: int = 1,
          padding: int = 1, groups: int = 1) -> dict:
    return {
        "type": "Conv2dBlock",
        "params": {
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "groups": groups,
            "bias": False,
        },
    }


def _bn() -> dict:
    return {"type": "BatchNormBlock", "params": {"momentum": 0.1, "eps": 1e-5}}


def _relu() -> dict:
    return {"type": "ReLUBlock", "params": {"inplace": True}}


def _silu() -> dict:
    return {"type": "SiLUBlock", "params": {"inplace": True}}


def _maxpool(kernel_size: int = 2, stride: int = 2) -> dict:
    return {
        "type": "PoolingBlock",
        "params": {"mode": "max", "kernel_size": kernel_size, "stride": stride, "padding": 0},
    }


def _upsample(scale_factor: int = 2, mode: str = "nearest") -> dict:
    return {
        "type": "UpsampleBlock",
        "params": {"scale_factor": scale_factor, "mode": mode},
    }


def _vgg_conv_relu_bn(out_channels: int) -> list[dict]:
    """Standard VGG-style conv unit: 3x3 conv + BN + ReLU (BN added for stability)."""
    return [_conv(out_channels), _bn(), _relu()]


def _vgg_stage(stage_idx: int, num_convs: int, out_channels: int) -> list[dict]:
    blocks: list[dict] = []
    for conv_idx in range(num_convs):
        blocks.extend(_vgg_conv_relu_bn(out_channels))
    blocks.append(_maxpool())
    return blocks


def _vgg_backbone(stages: list[tuple[int, int]]) -> list[dict]:
    """Build a VGG backbone block chain.

    `stages` is a list of (num_convs, channels) tuples — one per VGG stage.
    """
    chain: list[dict] = []
    for stage_idx, (num_convs, out_channels) in enumerate(stages):
        chain.extend(_vgg_stage(stage_idx, num_convs, out_channels))
    return chain


# ---------------------------------------------------------------------------
# Template catalog
# ---------------------------------------------------------------------------


def _vgg_template(name: str, label: str, stages: list[tuple[int, int]],
                  description: str) -> dict:
    return {
        "id": name,
        "label": label,
        "category": "backbone",
        "family": "VGG",
        "description": description,
        "hyperparameters": {
            "kernel_size": 3,
            "padding": 1,
            "stride": 1,
            "stages": [{"num_convs": nc, "out_channels": c} for nc, c in stages],
            "pooling": {"type": "max", "kernel_size": 2, "stride": 2},
        },
        "blocks": _vgg_backbone(stages),
    }


def _centernet_neck() -> dict:
    """CenterNet DLA-style upsample neck.

    Three upsample + 3x3 conv + BN + ReLU stages bringing the backbone
    output back to stride 4 with progressively decreasing channels.
    """
    chain: list[dict] = []
    for channels in (256, 128, 64):
        chain.append(_upsample(scale_factor=2, mode="bilinear"))
        chain.append(_conv(channels, kernel_size=3, padding=1))
        chain.append(_bn())
        chain.append(_relu())
    return {
        "id": "centernet_neck",
        "label": "CenterNet · Upsample Neck",
        "category": "neck",
        "family": "CenterNet",
        "description": (
            "DLA-style upsample neck producing a stride-4 feature map. "
            "Three (upsample + 3x3 conv + BN + ReLU) stages, channels 256→128→64."
        ),
        "hyperparameters": {
            "channels": [256, 128, 64],
            "upsample_mode": "bilinear",
            "scale_factor": 2,
        },
        "blocks": chain,
    }


def _centernet_head(num_classes: int = 80, head_channels: int = 64) -> dict:
    """CenterNet triple head: heatmap (cls), wh (size), reg (offset)."""
    blocks = [
        # Shared head conv tower
        _conv(head_channels, kernel_size=3, padding=1),
        _relu(),
        # Heatmap head (classification via center heatmap)
        {
            "type": "CenterHeadBlock",
            "params": {
                "in_channels": head_channels,
                "num_classes": num_classes,
                "kernel_size": 1,
                "stride": 1,
                "padding": 0,
                "bias": True,
                "init_bias": -2.19,  # focal-loss style prior
            },
        },
    ]
    return {
        "id": "centernet_head",
        "label": "CenterNet · Heatmap + WH + Offset Head",
        "category": "head",
        "family": "CenterNet",
        "description": (
            "Anchor-free CenterNet head: a shared 3x3 conv tower feeding a "
            "CenterHeadBlock that produces the class heatmap, plus implicit "
            "WH and offset branches handled by the head block configuration."
        ),
        "hyperparameters": {
            "num_classes": num_classes,
            "head_channels": head_channels,
            "branches": ["heatmap", "wh", "offset"],
        },
        "blocks": blocks,
    }


def _fcos_neck() -> dict:
    """FCOS-style FPN neck — 1x1 lateral + 3x3 output convs per pyramid level."""
    blocks = [
        # Single chain placeholder representing P3-P7 pipeline; the FPN block
        # is treated as a composite operator here for simplicity.
        {
            "type": "FPNBlock",
            "params": {
                "in_channels_list": [256, 512, 1024, 2048],
                "out_channels": 256,
                "num_levels": 5,
                "extra_levels": ["P6", "P7"],
                "use_p5": True,
            },
        },
    ]
    return {
        "id": "fcos_neck",
        "label": "FCOS · FPN P3-P7",
        "category": "neck",
        "family": "FCOS",
        "description": "FPN neck producing P3-P7 pyramid features with 256 channels.",
        "hyperparameters": {
            "out_channels": 256,
            "levels": ["P3", "P4", "P5", "P6", "P7"],
        },
        "blocks": blocks,
    }


def _fcos_head(num_classes: int = 80) -> dict:
    """FCOS decoupled head: cls / bbox-reg / centerness."""
    blocks = [
        {
            "type": "DecoupledHeadBlock",
            "params": {
                "in_channels": 256,
                "feat_channels": 256,
                "stacked_convs": 4,
                "norm": "GroupNorm",
                "activation": "ReLU",
            },
        },
        {
            "type": "ClsHeadBlock",
            "params": {
                "in_channels": 256,
                "num_classes": num_classes,
                "kernel_size": 3,
                "padding": 1,
                "prior_prob": 0.01,
            },
        },
        {
            "type": "RegHeadBlock",
            "params": {
                "in_channels": 256,
                "num_outputs": 4,  # ltrb distances
                "kernel_size": 3,
                "padding": 1,
            },
        },
        {
            "type": "ObjectnessHead",
            "params": {
                "in_channels": 256,
                "num_outputs": 1,  # centerness
                "kernel_size": 3,
                "padding": 1,
            },
        },
    ]
    return {
        "id": "fcos_head",
        "label": "FCOS · Decoupled Head",
        "category": "head",
        "family": "FCOS",
        "description": (
            "Anchor-free FCOS head: 4 stacked 3x3 convs with GroupNorm, then "
            "decoupled cls (num_classes), bbox-reg (ltrb) and centerness branches."
        ),
        "hyperparameters": {
            "num_classes": num_classes,
            "feat_channels": 256,
            "stacked_convs": 4,
            "branches": ["cls", "reg(ltrb)", "centerness"],
        },
        "blocks": blocks,
    }


def _nanodet_neck() -> dict:
    blocks = [
        {
            "type": "PANBlock",
            "params": {
                "in_channels_list": [116, 232, 464],
                "out_channels": 96,
                "num_levels": 3,
                "use_depthwise": True,
            },
        },
    ]
    return {
        "id": "nanodet_neck",
        "label": "NanoDet · PAN Neck",
        "category": "neck",
        "family": "NanoDet",
        "description": "Lightweight PAN neck with depthwise separable convs (out=96).",
        "hyperparameters": {"out_channels": 96, "use_depthwise": True},
        "blocks": blocks,
    }


def _nanodet_head(num_classes: int = 80, reg_max: int = 7) -> dict:
    blocks = [
        {
            "type": "DecoupledHeadBlock",
            "params": {
                "in_channels": 96,
                "feat_channels": 96,
                "stacked_convs": 2,
                "use_depthwise": True,
                "norm": "BatchNorm",
                "activation": "LeakyReLU",
            },
        },
        {
            "type": "ClsHeadBlock",
            "params": {
                "in_channels": 96,
                "num_classes": num_classes,
                "kernel_size": 1,
                "prior_prob": 0.01,
            },
        },
        {
            "type": "DFLBlock",
            "params": {
                "in_channels": 96,
                "reg_max": reg_max,
                "num_outputs": 4 * (reg_max + 1),
                "kernel_size": 1,
            },
        },
        {
            "type": "DistributionProjectBlock",
            "params": {"reg_max": reg_max, "num_outputs": 4},
        },
    ]
    return {
        "id": "nanodet_head",
        "label": "NanoDet · GFL Head (DFL)",
        "category": "head",
        "family": "NanoDet",
        "description": (
            "Generalized Focal Loss head: lightweight depthwise tower, cls "
            "branch and DFL regression projected back to 4 ltrb distances."
        ),
        "hyperparameters": {
            "num_classes": num_classes,
            "reg_max": reg_max,
            "feat_channels": 96,
            "use_depthwise": True,
        },
        "blocks": blocks,
    }


def _picodet_neck() -> dict:
    blocks = [
        {
            "type": "BiFPNBlock",
            "params": {
                "in_channels_list": [128, 256, 512],
                "out_channels": 128,
                "num_levels": 3,
                "use_csp": True,
            },
        },
    ]
    return {
        "id": "picodet_neck",
        "label": "PicoDet · CSP-PAN Neck",
        "category": "neck",
        "family": "PicoDet",
        "description": "CSP-PAN-style neck (BiFPN variant) with 128 output channels.",
        "hyperparameters": {"out_channels": 128, "use_csp": True},
        "blocks": blocks,
    }


def _picodet_head(num_classes: int = 80, reg_max: int = 7) -> dict:
    blocks = [
        {
            "type": "DecoupledHeadBlock",
            "params": {
                "in_channels": 128,
                "feat_channels": 128,
                "stacked_convs": 2,
                "use_depthwise": True,
                "norm": "BatchNorm",
                "activation": "HSwish",
            },
        },
        {
            "type": "ClsHeadBlock",
            "params": {
                "in_channels": 128,
                "num_classes": num_classes,
                "kernel_size": 1,
                "use_vfl": True,
                "prior_prob": 0.01,
            },
        },
        {
            "type": "DFLBlock",
            "params": {
                "in_channels": 128,
                "reg_max": reg_max,
                "num_outputs": 4 * (reg_max + 1),
                "kernel_size": 1,
            },
        },
        {
            "type": "DistributionProjectBlock",
            "params": {"reg_max": reg_max, "num_outputs": 4},
        },
    ]
    return {
        "id": "picodet_head",
        "label": "PicoDet · VFL + DFL Head",
        "category": "head",
        "family": "PicoDet",
        "description": (
            "PicoDet head with Varifocal Loss classification branch and "
            "DFL regression projected to 4 ltrb distances."
        ),
        "hyperparameters": {
            "num_classes": num_classes,
            "reg_max": reg_max,
            "feat_channels": 128,
            "use_vfl": True,
        },
        "blocks": blocks,
    }


def _yolox_neck() -> dict:
    blocks = [
        {
            "type": "PANBlock",
            "params": {
                "in_channels_list": [256, 512, 1024],
                "out_channels": 256,
                "num_levels": 3,
                "use_csp": True,
            },
        },
    ]
    return {
        "id": "yolox_neck",
        "label": "YOLOX · PAFPN Neck",
        "category": "neck",
        "family": "YOLOX",
        "description": "YOLOX PAFPN neck with CSP blocks (out=256 per level).",
        "hyperparameters": {"out_channels": 256, "use_csp": True},
        "blocks": blocks,
    }


def _yolox_head(num_classes: int = 80) -> dict:
    blocks = [
        {
            "type": "DecoupledHeadBlock",
            "params": {
                "in_channels": 256,
                "feat_channels": 256,
                "stacked_convs": 2,
                "norm": "BatchNorm",
                "activation": "SiLU",
            },
        },
        {
            "type": "ClsHeadBlock",
            "params": {
                "in_channels": 256,
                "num_classes": num_classes,
                "kernel_size": 1,
                "prior_prob": 0.01,
            },
        },
        {
            "type": "RegHeadBlock",
            "params": {
                "in_channels": 256,
                "num_outputs": 4,  # cx,cy,w,h
                "kernel_size": 1,
            },
        },
        {
            "type": "ObjectnessHead",
            "params": {
                "in_channels": 256,
                "num_outputs": 1,
                "kernel_size": 1,
            },
        },
    ]
    return {
        "id": "yolox_head",
        "label": "YOLOX · Decoupled Head",
        "category": "head",
        "family": "YOLOX",
        "description": (
            "Anchor-free YOLOX decoupled head: cls / reg(xywh) / obj branches "
            "with SiLU activation and BatchNorm."
        ),
        "hyperparameters": {
            "num_classes": num_classes,
            "feat_channels": 256,
            "stacked_convs": 2,
            "branches": ["cls", "reg(xywh)", "obj"],
        },
        "blocks": blocks,
    }


def _pp_yoloe_neck() -> dict:
    blocks = [
        {
            "type": "PANBlock",
            "params": {
                "in_channels_list": [192, 384, 768],
                "out_channels": 192,
                "num_levels": 3,
                "use_csp": True,
                "use_ese": True,
            },
        },
    ]
    return {
        "id": "pp_yoloe_neck",
        "label": "PP-YOLOE · CSP-PAN Neck",
        "category": "neck",
        "family": "PP-YOLOE",
        "description": "CSP-PAN with ESE attention modules (out=192 per level).",
        "hyperparameters": {"out_channels": 192, "use_ese": True},
        "blocks": blocks,
    }


def _pp_yoloe_head(num_classes: int = 80, reg_max: int = 15) -> dict:
    blocks = [
        {
            "type": "DecoupledHeadBlock",
            "params": {
                "in_channels": 192,
                "feat_channels": 192,
                "stacked_convs": 2,
                "use_ese": True,
                "norm": "BatchNorm",
                "activation": "SiLU",
            },
        },
        {
            "type": "ClsHeadBlock",
            "params": {
                "in_channels": 192,
                "num_classes": num_classes,
                "kernel_size": 1,
                "use_vfl": True,
                "prior_prob": 0.01,
            },
        },
        {
            "type": "DFLBlock",
            "params": {
                "in_channels": 192,
                "reg_max": reg_max,
                "num_outputs": 4 * (reg_max + 1),
                "kernel_size": 1,
            },
        },
        {
            "type": "DistributionProjectBlock",
            "params": {"reg_max": reg_max, "num_outputs": 4},
        },
    ]
    return {
        "id": "pp_yoloe_head",
        "label": "PP-YOLOE · ET-Head (VFL + DFL)",
        "category": "head",
        "family": "PP-YOLOE",
        "description": (
            "PP-YOLOE ET-Head: ESE-augmented decoupled tower, VFL classification "
            "branch and high-precision DFL regression (reg_max=15)."
        ),
        "hyperparameters": {
            "num_classes": num_classes,
            "reg_max": reg_max,
            "feat_channels": 192,
            "use_ese": True,
            "use_vfl": True,
        },
        "blocks": blocks,
    }


def _dwconv(out_channels: int, kernel_size: int = 3, stride: int = 1) -> dict:
    return {
        "type": "DWConvBlock",
        "params": {
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": kernel_size // 2,
            "bias": False,
        },
    }


def _pwconv(out_channels: int) -> dict:
    return {
        "type": "PWConvBlock",
        "params": {
            "out_channels": out_channels,
            "kernel_size": 1,
            "stride": 1,
            "padding": 0,
            "bias": False,
        },
    }


def _hswish() -> dict:
    return {"type": "HSwishBlock", "params": {"inplace": True}}


def _mbconv(out_channels: int, expansion: int = 6, kernel_size: int = 3,
            stride: int = 1, use_se: bool = False, activation: str = "ReLU6") -> dict:
    return {
        "type": "MBConv",
        "params": {
            "out_channels": out_channels,
            "expansion": expansion,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": kernel_size // 2,
            "use_se": use_se,
            "activation": activation,
        },
    }


def _ghost(out_channels: int, ratio: int = 2, kernel_size: int = 3,
           stride: int = 1) -> dict:
    return {
        "type": "GhostBlock",
        "params": {
            "out_channels": out_channels,
            "ratio": ratio,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": kernel_size // 2,
            "activation": "ReLU",
        },
    }


def _csp(out_channels: int, num_blocks: int = 3, shortcut: bool = True,
         expansion: float = 0.5) -> dict:
    return {
        "type": "CSPBlock",
        "params": {
            "out_channels": out_channels,
            "num_blocks": num_blocks,
            "shortcut": shortcut,
            "expansion": expansion,
            "activation": "SiLU",
        },
    }


def _sppf(out_channels: int, kernel_size: int = 5) -> dict:
    return {
        "type": "SPPFBlock",
        "params": {
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "activation": "SiLU",
        },
    }


def _mobilenet_v2_backbone() -> dict:
    chain = [
        _conv(32, kernel_size=3, stride=2, padding=1),
        _bn(), _relu(),
        _mbconv(16, expansion=1, kernel_size=3, stride=1),
        _mbconv(24, expansion=6, kernel_size=3, stride=2),
        _mbconv(24, expansion=6, kernel_size=3, stride=1),
        _mbconv(32, expansion=6, kernel_size=3, stride=2),
        _mbconv(32, expansion=6, kernel_size=3, stride=1),
        _mbconv(32, expansion=6, kernel_size=3, stride=1),
        _mbconv(64, expansion=6, kernel_size=3, stride=2),
        _mbconv(64, expansion=6, kernel_size=3, stride=1),
        _mbconv(64, expansion=6, kernel_size=3, stride=1),
        _mbconv(96, expansion=6, kernel_size=3, stride=1),
        _mbconv(160, expansion=6, kernel_size=3, stride=2),
        _mbconv(160, expansion=6, kernel_size=3, stride=1),
        _mbconv(320, expansion=6, kernel_size=3, stride=1),
        _pwconv(1280),
        _bn(), _relu(),
    ]
    return {
        "id": "mobilenet_v2",
        "label": "MobileNetV2 (edge backbone)",
        "category": "backbone",
        "family": "MobileNet",
        "description": (
            "Edge-friendly inverted-residual MobileNetV2 backbone. Pairs well "
            "with anchor-free heads (NanoDet, PicoDet, CenterNet)."
        ),
        "hyperparameters": {
            "width_multiplier": 1.0,
            "expansion": 6,
            "activation": "ReLU6",
            "output_stride": 32,
        },
        "blocks": chain,
    }


def _mobilenet_v3_small_backbone() -> dict:
    chain = [
        _conv(16, kernel_size=3, stride=2, padding=1),
        _bn(), _hswish(),
        _mbconv(16, expansion=1, kernel_size=3, stride=2, use_se=True, activation="ReLU"),
        _mbconv(24, expansion=4, kernel_size=3, stride=2, activation="ReLU"),
        _mbconv(24, expansion=3, kernel_size=3, stride=1, activation="ReLU"),
        _mbconv(40, expansion=4, kernel_size=5, stride=2, use_se=True, activation="HSwish"),
        _mbconv(40, expansion=6, kernel_size=5, stride=1, use_se=True, activation="HSwish"),
        _mbconv(40, expansion=6, kernel_size=5, stride=1, use_se=True, activation="HSwish"),
        _mbconv(48, expansion=3, kernel_size=5, stride=1, use_se=True, activation="HSwish"),
        _mbconv(48, expansion=3, kernel_size=5, stride=1, use_se=True, activation="HSwish"),
        _mbconv(96, expansion=6, kernel_size=5, stride=2, use_se=True, activation="HSwish"),
        _mbconv(96, expansion=6, kernel_size=5, stride=1, use_se=True, activation="HSwish"),
        _mbconv(96, expansion=6, kernel_size=5, stride=1, use_se=True, activation="HSwish"),
        _pwconv(576),
        _bn(), _hswish(),
    ]
    return {
        "id": "mobilenet_v3_small",
        "label": "MobileNetV3-Small (ultra-edge)",
        "category": "backbone",
        "family": "MobileNet",
        "description": (
            "Ultra-lightweight MobileNetV3-Small backbone with SE modules and "
            "HardSwish activations — ideal for ARM / NPU anchor-free detectors."
        ),
        "hyperparameters": {
            "width_multiplier": 1.0,
            "use_se": True,
            "activation": "HSwish",
            "output_stride": 32,
        },
        "blocks": chain,
    }


def _shufflenet_v2_backbone() -> dict:
    chain = [
        _conv(24, kernel_size=3, stride=2, padding=1), _bn(), _relu(),
        _maxpool(kernel_size=3, stride=2),
        # Stage 2 (output 116ch)
        _dwconv(116, kernel_size=3, stride=2), _pwconv(116), _bn(), _relu(),
        _dwconv(116, kernel_size=3, stride=1), _pwconv(116), _bn(), _relu(),
        # Stage 3 (output 232ch)
        _dwconv(232, kernel_size=3, stride=2), _pwconv(232), _bn(), _relu(),
        _dwconv(232, kernel_size=3, stride=1), _pwconv(232), _bn(), _relu(),
        # Stage 4 (output 464ch)
        _dwconv(464, kernel_size=3, stride=2), _pwconv(464), _bn(), _relu(),
        _dwconv(464, kernel_size=3, stride=1), _pwconv(464), _bn(), _relu(),
        _pwconv(1024), _bn(), _relu(),
    ]
    return {
        "id": "shufflenet_v2",
        "label": "ShuffleNetV2 (NanoDet backbone)",
        "category": "backbone",
        "family": "ShuffleNet",
        "description": (
            "ShuffleNetV2 backbone (channels 116/232/464). Reference backbone "
            "for NanoDet on mobile / embedded targets."
        ),
        "hyperparameters": {
            "width_multiplier": 1.0,
            "output_channels": [116, 232, 464],
            "output_stride": 32,
        },
        "blocks": chain,
    }


def _ghostnet_backbone() -> dict:
    chain = [
        _conv(16, kernel_size=3, stride=2, padding=1), _bn(), _relu(),
        _ghost(16, ratio=2, kernel_size=3, stride=1),
        _ghost(24, ratio=2, kernel_size=3, stride=2),
        _ghost(24, ratio=2, kernel_size=3, stride=1),
        _ghost(40, ratio=2, kernel_size=5, stride=2),
        _ghost(40, ratio=2, kernel_size=5, stride=1),
        _ghost(80, ratio=2, kernel_size=3, stride=2),
        _ghost(80, ratio=2, kernel_size=3, stride=1),
        _ghost(112, ratio=2, kernel_size=3, stride=1),
        _ghost(112, ratio=2, kernel_size=3, stride=1),
        _ghost(160, ratio=2, kernel_size=5, stride=2),
        _ghost(160, ratio=2, kernel_size=5, stride=1),
        _pwconv(960), _bn(), _relu(),
    ]
    return {
        "id": "ghostnet",
        "label": "GhostNet (NanoDet-Plus backbone)",
        "category": "backbone",
        "family": "GhostNet",
        "description": (
            "GhostNet backbone with cheap operations producing ghost features. "
            "Used by NanoDet-Plus for anchor-free edge detection."
        ),
        "hyperparameters": {
            "ratio": 2,
            "width_multiplier": 1.0,
            "output_stride": 32,
        },
        "blocks": chain,
    }


def _repvgg_backbone() -> dict:
    chain = [
        _conv(64, kernel_size=3, stride=2, padding=1), _bn(), _relu(),
        # Stage 2
        _conv(64, kernel_size=3, stride=2, padding=1), _bn(), _relu(),
        _conv(64, kernel_size=3, stride=1, padding=1), _bn(), _relu(),
        _conv(64, kernel_size=3, stride=1, padding=1), _bn(), _relu(),
        # Stage 3
        _conv(128, kernel_size=3, stride=2, padding=1), _bn(), _relu(),
        _conv(128, kernel_size=3, stride=1, padding=1), _bn(), _relu(),
        _conv(128, kernel_size=3, stride=1, padding=1), _bn(), _relu(),
        # Stage 4
        _conv(256, kernel_size=3, stride=2, padding=1), _bn(), _relu(),
        _conv(256, kernel_size=3, stride=1, padding=1), _bn(), _relu(),
        _conv(256, kernel_size=3, stride=1, padding=1), _bn(), _relu(),
        # Stage 5
        _conv(512, kernel_size=3, stride=2, padding=1), _bn(), _relu(),
    ]
    return {
        "id": "repvgg_a0",
        "label": "RepVGG-A0 (re-param edge backbone)",
        "category": "backbone",
        "family": "RepVGG",
        "description": (
            "Plain VGG-shaped backbone (3x3 conv + BN + ReLU only) inspired by "
            "RepVGG re-parametrization. Highly ONNX-friendly after fusion."
        ),
        "hyperparameters": {
            "kernel_size": 3,
            "padding": 1,
            "stage_channels": [64, 64, 128, 256, 512],
            "output_stride": 32,
        },
        "blocks": chain,
    }


def _cspdarknet_backbone() -> dict:
    chain = [
        {"type": "FocusBlock", "params": {"out_channels": 64, "kernel_size": 3, "activation": "SiLU"}},
        _conv(128, kernel_size=3, stride=2, padding=1), _bn(), _silu(),
        _csp(128, num_blocks=3),
        _conv(256, kernel_size=3, stride=2, padding=1), _bn(), _silu(),
        _csp(256, num_blocks=9),
        _conv(512, kernel_size=3, stride=2, padding=1), _bn(), _silu(),
        _csp(512, num_blocks=9),
        _conv(1024, kernel_size=3, stride=2, padding=1), _bn(), _silu(),
        _sppf(1024, kernel_size=5),
        _csp(1024, num_blocks=3, shortcut=False),
    ]
    return {
        "id": "cspdarknet",
        "label": "CSPDarknet (YOLOX backbone)",
        "category": "backbone",
        "family": "YOLOX",
        "description": (
            "CSPDarknet backbone with Focus stem and SPPF, channels 64→1024. "
            "Reference backbone for YOLOX / YOLOv5-style anchor-free detectors."
        ),
        "hyperparameters": {
            "depth_multiplier": 1.0,
            "width_multiplier": 1.0,
            "activation": "SiLU",
            "use_focus": True,
            "use_sppf": True,
        },
        "blocks": chain,
    }


def _rtmdet_neck() -> dict:
    blocks = [
        {
            "type": "PANBlock",
            "params": {
                "in_channels_list": [256, 512, 1024],
                "out_channels": 256,
                "num_levels": 3,
                "num_csp_blocks": 3,
                "use_depthwise": True,
                "activation": "SiLU",
                "norm": "BatchNorm",
            },
        },
    ]
    return {
        "id": "rtmdet_neck",
        "label": "RTMDet · CSP-PAFPN Neck",
        "category": "neck",
        "family": "RTMDet",
        "description": "RTMDet CSP-PAFPN neck with depthwise convs and SiLU.",
        "hyperparameters": {"out_channels": 256, "use_depthwise": True, "num_csp_blocks": 3},
        "blocks": blocks,
    }


def _rtmdet_head(num_classes: int = 80) -> dict:
    blocks = [
        {
            "type": "DecoupledHeadBlock",
            "params": {
                "in_channels": 256,
                "feat_channels": 256,
                "stacked_convs": 2,
                "share_conv": True,
                "use_depthwise": True,
                "norm": "BatchNorm",
                "activation": "SiLU",
            },
        },
        {
            "type": "ClsHeadBlock",
            "params": {
                "in_channels": 256,
                "num_classes": num_classes,
                "kernel_size": 1,
                "prior_prob": 0.01,
                "use_qfl": True,
            },
        },
        {
            "type": "RegHeadBlock",
            "params": {
                "in_channels": 256,
                "num_outputs": 4,  # ltrb
                "kernel_size": 1,
            },
        },
    ]
    return {
        "id": "rtmdet_head",
        "label": "RTMDet · QFL Head",
        "category": "head",
        "family": "RTMDet",
        "description": (
            "Real-Time Models for object Detection head: shared depthwise tower, "
            "Quality Focal Loss classification and direct ltrb regression."
        ),
        "hyperparameters": {
            "num_classes": num_classes,
            "feat_channels": 256,
            "stacked_convs": 2,
            "use_qfl": True,
            "share_conv": True,
        },
        "blocks": blocks,
    }


def _ttfnet_head(num_classes: int = 80) -> dict:
    blocks = [
        _conv(128, kernel_size=3, padding=1), _bn(), _relu(),
        _conv(128, kernel_size=3, padding=1), _bn(), _relu(),
        {
            "type": "CenterHeadBlock",
            "params": {
                "in_channels": 128,
                "num_classes": num_classes,
                "kernel_size": 1,
                "init_bias": -2.19,
                "wh_branch": True,
                "wh_planes": 4,
            },
        },
    ]
    return {
        "id": "ttfnet_head",
        "label": "TTFNet · Gaussian Center Head",
        "category": "head",
        "family": "TTFNet",
        "description": (
            "Training-Time-Friendly center head: 2 shared 3x3 convs feeding a "
            "Gaussian center heatmap and a 4-channel WH regression."
        ),
        "hyperparameters": {
            "num_classes": num_classes,
            "feat_channels": 128,
            "wh_planes": 4,
            "branches": ["heatmap", "wh"],
        },
        "blocks": blocks,
    }


def _autoassign_head(num_classes: int = 80) -> dict:
    blocks = [
        {
            "type": "DecoupledHeadBlock",
            "params": {
                "in_channels": 256,
                "feat_channels": 256,
                "stacked_convs": 4,
                "norm": "GroupNorm",
                "activation": "ReLU",
            },
        },
        {
            "type": "ClsHeadBlock",
            "params": {
                "in_channels": 256,
                "num_classes": num_classes,
                "kernel_size": 3,
                "padding": 1,
                "prior_prob": 0.02,
            },
        },
        {
            "type": "RegHeadBlock",
            "params": {
                "in_channels": 256,
                "num_outputs": 4,
                "kernel_size": 3,
                "padding": 1,
            },
        },
        {
            "type": "ObjectnessHead",
            "params": {
                "in_channels": 256,
                "num_outputs": 1,  # implicit-objectness "w" weight
                "kernel_size": 3,
                "padding": 1,
            },
        },
    ]
    return {
        "id": "autoassign_head",
        "label": "AutoAssign · Differentiable Head",
        "category": "head",
        "family": "AutoAssign",
        "description": (
            "AutoAssign anchor-free head with differentiable label assignment: "
            "cls + reg(ltrb) + implicit-objectness weight branches."
        ),
        "hyperparameters": {
            "num_classes": num_classes,
            "feat_channels": 256,
            "stacked_convs": 4,
            "branches": ["cls", "reg(ltrb)", "implicit_obj"],
        },
        "blocks": blocks,
    }


def _onenet_head(num_classes: int = 80) -> dict:
    blocks = [
        {
            "type": "DecoupledHeadBlock",
            "params": {
                "in_channels": 256,
                "feat_channels": 256,
                "stacked_convs": 4,
                "norm": "GroupNorm",
                "activation": "ReLU",
            },
        },
        {
            "type": "ClsHeadBlock",
            "params": {
                "in_channels": 256,
                "num_classes": num_classes,
                "kernel_size": 3,
                "padding": 1,
                "prior_prob": 0.01,
                "min_one_label_per_gt": True,
            },
        },
        {
            "type": "RegHeadBlock",
            "params": {
                "in_channels": 256,
                "num_outputs": 4,
                "kernel_size": 3,
                "padding": 1,
            },
        },
        {
            "type": "NMSFreeDecodeBlock",
            "params": {
                "score_threshold": 0.05,
                "max_detections": 100,
                "top_k": 100,
            },
        },
    ]
    return {
        "id": "onenet_head",
        "label": "OneNet · NMS-Free Head",
        "category": "head",
        "family": "OneNet",
        "description": (
            "OneNet end-to-end anchor-free head with one-to-one Minimum Cost "
            "assignment, removing the NMS post-processing."
        ),
        "hyperparameters": {
            "num_classes": num_classes,
            "feat_channels": 256,
            "stacked_convs": 4,
            "nms_free": True,
        },
        "blocks": blocks,
    }


def _reppoints_head(num_classes: int = 80, num_points: int = 9) -> dict:
    blocks = [
        {
            "type": "DecoupledHeadBlock",
            "params": {
                "in_channels": 256,
                "feat_channels": 256,
                "stacked_convs": 3,
                "norm": "GroupNorm",
                "activation": "ReLU",
            },
        },
        {
            "type": "ClsHeadBlock",
            "params": {
                "in_channels": 256,
                "num_classes": num_classes,
                "kernel_size": 3,
                "padding": 1,
            },
        },
        {
            "type": "RegHeadBlock",
            "params": {
                "in_channels": 256,
                "num_outputs": 2 * num_points,  # x,y offsets per point
                "kernel_size": 3,
                "padding": 1,
            },
        },
        {
            "type": "RegHeadBlock",
            "params": {
                "in_channels": 256,
                "num_outputs": 2 * num_points,  # refinement offsets
                "kernel_size": 3,
                "padding": 1,
            },
        },
    ]
    return {
        "id": "reppoints_head",
        "label": "RepPoints · Anchor-Free Head",
        "category": "head",
        "family": "RepPoints",
        "description": (
            "RepPoints head learning representative points (init + refine) plus "
            "a classification branch — anchor-free without bounding box anchors."
        ),
        "hyperparameters": {
            "num_classes": num_classes,
            "num_points": num_points,
            "feat_channels": 256,
            "stacked_convs": 3,
        },
        "blocks": blocks,
    }


def _foveabox_head(num_classes: int = 80) -> dict:
    blocks = [
        {
            "type": "DecoupledHeadBlock",
            "params": {
                "in_channels": 256,
                "feat_channels": 256,
                "stacked_convs": 4,
                "norm": "GroupNorm",
                "activation": "ReLU",
            },
        },
        {
            "type": "ClsHeadBlock",
            "params": {
                "in_channels": 256,
                "num_classes": num_classes,
                "kernel_size": 3,
                "padding": 1,
                "prior_prob": 0.01,
            },
        },
        {
            "type": "RegHeadBlock",
            "params": {
                "in_channels": 256,
                "num_outputs": 4,  # (t,l,b,r) log-distances
                "kernel_size": 3,
                "padding": 1,
                "activation": "exp",
            },
        },
    ]
    return {
        "id": "foveabox_head",
        "label": "FoveaBox · Fovea Head",
        "category": "head",
        "family": "FoveaBox",
        "description": (
            "FoveaBox anchor-free head with a fovea (object center) cls branch "
            "and an exponential 4-distance regression branch."
        ),
        "hyperparameters": {
            "num_classes": num_classes,
            "feat_channels": 256,
            "stacked_convs": 4,
            "regression_activation": "exp",
        },
        "blocks": blocks,
    }


def _atss_head(num_classes: int = 80) -> dict:
    blocks = [
        {
            "type": "DecoupledHeadBlock",
            "params": {
                "in_channels": 256,
                "feat_channels": 256,
                "stacked_convs": 4,
                "norm": "GroupNorm",
                "activation": "ReLU",
            },
        },
        {
            "type": "ClsHeadBlock",
            "params": {
                "in_channels": 256,
                "num_classes": num_classes,
                "kernel_size": 3,
                "padding": 1,
                "prior_prob": 0.01,
            },
        },
        {
            "type": "RegHeadBlock",
            "params": {
                "in_channels": 256,
                "num_outputs": 4,
                "kernel_size": 3,
                "padding": 1,
            },
        },
        {
            "type": "ObjectnessHead",
            "params": {
                "in_channels": 256,
                "num_outputs": 1,  # centerness
                "kernel_size": 3,
                "padding": 1,
            },
        },
    ]
    return {
        "id": "atss_head",
        "label": "ATSS · Anchor-Free Variant Head",
        "category": "head",
        "family": "ATSS",
        "description": (
            "ATSS-style head used in anchor-free mode (single point per location): "
            "cls + ltrb regression + centerness."
        ),
        "hyperparameters": {
            "num_classes": num_classes,
            "feat_channels": 256,
            "stacked_convs": 4,
            "branches": ["cls", "reg(ltrb)", "centerness"],
        },
        "blocks": blocks,
    }


def _damo_yolo_head(num_classes: int = 80) -> dict:
    blocks = [
        {
            "type": "DecoupledHeadBlock",
            "params": {
                "in_channels": 256,
                "feat_channels": 256,
                "stacked_convs": 1,  # ZeroHead — minimal compute
                "use_depthwise": False,
                "norm": "BatchNorm",
                "activation": "SiLU",
            },
        },
        {
            "type": "ClsHeadBlock",
            "params": {
                "in_channels": 256,
                "num_classes": num_classes,
                "kernel_size": 1,
                "prior_prob": 0.01,
            },
        },
        {
            "type": "DFLBlock",
            "params": {
                "in_channels": 256,
                "reg_max": 16,
                "num_outputs": 4 * 17,
                "kernel_size": 1,
            },
        },
        {
            "type": "DistributionProjectBlock",
            "params": {"reg_max": 16, "num_outputs": 4},
        },
    ]
    return {
        "id": "damo_yolo_head",
        "label": "DAMO-YOLO · ZeroHead",
        "category": "head",
        "family": "DAMO-YOLO",
        "description": (
            "DAMO-YOLO ZeroHead: a single conv tower feeding cls + DFL ltrb "
            "regression — designed for low-latency edge deployment."
        ),
        "hyperparameters": {
            "num_classes": num_classes,
            "feat_channels": 256,
            "stacked_convs": 1,
            "reg_max": 16,
        },
        "blocks": blocks,
    }


def _build_catalog() -> list[dict]:
    return [
        # VGG-like backbones
        _vgg_template(
            "vgg11", "VGG-11 (VGG-A)",
            [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)],
            "VGG-A backbone (8 convs + 5 max-pools). 3x3 conv, pad=1, stride=1, BN+ReLU.",
        ),
        _vgg_template(
            "vgg13", "VGG-13 (VGG-B)",
            [(2, 64), (2, 128), (2, 256), (2, 512), (2, 512)],
            "VGG-B backbone (10 convs + 5 max-pools).",
        ),
        _vgg_template(
            "vgg16", "VGG-16 (VGG-D)",
            [(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)],
            "VGG-D backbone (13 convs + 5 max-pools) — most common VGG variant.",
        ),
        _vgg_template(
            "vgg19", "VGG-19 (VGG-E)",
            [(2, 64), (2, 128), (4, 256), (4, 512), (4, 512)],
            "VGG-E backbone (16 convs + 5 max-pools).",
        ),
        # CenterNet
        _centernet_neck(),
        _centernet_head(),
        # FCOS
        _fcos_neck(),
        _fcos_head(),
        # NanoDet
        _nanodet_neck(),
        _nanodet_head(),
        # PicoDet
        _picodet_neck(),
        _picodet_head(),
        # YOLOX
        _yolox_neck(),
        _yolox_head(),
        # PP-YOLOE
        _pp_yoloe_neck(),
        _pp_yoloe_head(),
        # Edge backbones for anchor-free detection
        _mobilenet_v2_backbone(),
        _mobilenet_v3_small_backbone(),
        _shufflenet_v2_backbone(),
        _ghostnet_backbone(),
        _repvgg_backbone(),
        _cspdarknet_backbone(),
        # RTMDet
        _rtmdet_neck(),
        _rtmdet_head(),
        # Additional anchor-free heads
        _ttfnet_head(),
        _autoassign_head(),
        _onenet_head(),
        _reppoints_head(),
        _foveabox_head(),
        _atss_head(),
        _damo_yolo_head(),
    ]


_CATALOG = _build_catalog()
_CATALOG_INDEX = {t["id"]: t for t in _CATALOG}


def list_templates() -> list[dict]:
    """Return a serializable list of all available templates."""
    return [deepcopy(t) for t in _CATALOG]


def get_template(template_id: str) -> dict:
    """Return a deep copy of a template by id."""
    if template_id not in _CATALOG_INDEX:
        raise KeyError(f"Unknown template id: {template_id}")
    return deepcopy(_CATALOG_INDEX[template_id])


# ---------------------------------------------------------------------------
# Per-block-type default hyperparameter sets
# ---------------------------------------------------------------------------
# These are the recommended editable hyperparameters for every block type that
# is relevant to VGG-like / anchor-free object detection. They are surfaced by
# the parameter editor so users can quickly fill in sane defaults instead of
# typing each key by hand. All defaults map to ONNX-convertible block types
# (see `onnx_exporter._onnx_op_for_block`).

BLOCK_DEFAULT_HYPERPARAMETERS: dict[str, dict] = {
    # --- Convolutional family ---
    "Conv2dBlock": {
        "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1,
        "dilation": 1, "groups": 1, "bias": False,
    },
    "ConvBlock": {
        "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1,
        "groups": 1, "bias": False, "norm": "BatchNorm", "activation": "ReLU",
    },
    "DWConvBlock": {
        "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1,
        "bias": False, "activation": "ReLU",
    },
    "PWConvBlock": {
        "out_channels": 64, "kernel_size": 1, "stride": 1, "padding": 0, "bias": False,
    },
    "DepthwiseConv": {
        "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1, "bias": False,
    },
    "MBConv": {
        "out_channels": 64, "expansion": 6, "kernel_size": 3, "stride": 1,
        "padding": 1, "use_se": False, "activation": "ReLU6",
    },
    "GhostBlock": {
        "out_channels": 64, "ratio": 2, "kernel_size": 3, "stride": 1,
        "padding": 1, "activation": "ReLU",
    },
    "ResidualBlock": {
        "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1,
        "expansion": 1, "shortcut": True, "activation": "ReLU",
    },
    "CSPBlock": {
        "out_channels": 128, "num_blocks": 3, "shortcut": True,
        "expansion": 0.5, "activation": "SiLU",
    },
    "FocusBlock": {"out_channels": 64, "kernel_size": 3, "activation": "SiLU"},
    "SPPBlock": {"out_channels": 256, "kernel_sizes": [5, 9, 13], "activation": "SiLU"},
    "SPPFBlock": {"out_channels": 256, "kernel_size": 5, "activation": "SiLU"},

    # --- Normalization / activation ---
    "BatchNormBlock": {"momentum": 0.1, "eps": 1e-5, "affine": True},
    "BatchNorm2d": {"momentum": 0.1, "eps": 1e-5, "affine": True},
    "GroupNormBlock": {"num_groups": 32, "eps": 1e-5, "affine": True},
    "SyncBNBlock": {"momentum": 0.1, "eps": 1e-5, "affine": True},
    "ReLUBlock": {"inplace": True},
    "ReLU": {"inplace": True},
    "SiLUBlock": {"inplace": True},
    "SiLU": {"inplace": True},
    "HSwishBlock": {"inplace": True},
    "SigmoidBlock": {},
    "Sigmoid": {},

    # --- Pooling / sampling ---
    "PoolingBlock": {"mode": "max", "kernel_size": 2, "stride": 2, "padding": 0},
    "MaxPool2d": {"kernel_size": 2, "stride": 2, "padding": 0, "ceil_mode": False},
    "AvgPool2d": {"kernel_size": 2, "stride": 2, "padding": 0, "ceil_mode": False},
    "UpsampleBlock": {"scale_factor": 2, "mode": "nearest", "align_corners": False},
    "Upsample": {"scale_factor": 2, "mode": "nearest", "align_corners": False},
    "ResizeFeatureMap": {"scale_factor": 2, "mode": "bilinear", "align_corners": False},
    "Downsample": {"factor": 2, "mode": "stride_conv"},

    # --- Tensor ops ---
    "AddBlock": {},
    "Add": {},
    "MulBlock": {},
    "ConcatBlock": {"axis": 1},
    "Concat": {"axis": 1},
    "SliceBlock": {"axis": 1, "start": 0, "end": -1, "step": 1},
    "ReshapeBlock": {"shape": [-1]},
    "Reshape": {"shape": [-1]},
    "TransposeBlock": {"perm": [0, 2, 3, 1]},
    "Transpose": {"perm": [0, 2, 3, 1]},
    "Flatten": {"axis": 1},
    "Permute": {"perm": [0, 2, 3, 1]},

    # --- Neck operators ---
    "FPNBlock": {
        "in_channels_list": [256, 512, 1024], "out_channels": 256,
        "num_levels": 3, "use_p5": True,
    },
    "PANBlock": {
        "in_channels_list": [256, 512, 1024], "out_channels": 256,
        "num_levels": 3, "use_csp": True, "use_depthwise": False,
        "activation": "SiLU",
    },
    "BiFPNBlock": {
        "in_channels_list": [128, 256, 512], "out_channels": 128,
        "num_levels": 3, "num_repeats": 1, "activation": "SiLU",
    },
    "ShapeAdapterBlock": {"mode": "auto"},

    # --- Anchor-free heads ---
    "DecoupledHeadBlock": {
        "in_channels": 256, "feat_channels": 256, "stacked_convs": 2,
        "share_conv": False, "use_depthwise": False,
        "norm": "BatchNorm", "activation": "SiLU",
    },
    "ClsHeadBlock": {
        "in_channels": 256, "num_classes": 80, "kernel_size": 1, "padding": 0,
        "prior_prob": 0.01, "use_vfl": False, "use_qfl": False,
    },
    "RegHeadBlock": {
        "in_channels": 256, "num_outputs": 4, "kernel_size": 1, "padding": 0,
    },
    "CenterHeadBlock": {
        "in_channels": 64, "num_classes": 80, "kernel_size": 1, "padding": 0,
        "init_bias": -2.19, "wh_branch": True, "wh_planes": 2,
    },
    "ClassificationHead": {"in_channels": 256, "num_classes": 80, "kernel_size": 1},
    "RegressionHead": {"in_channels": 256, "num_outputs": 4, "kernel_size": 1},
    "CenterHeatmapHead": {"in_channels": 64, "num_classes": 80, "init_bias": -2.19},
    "ObjectnessHead": {"in_channels": 256, "num_outputs": 1, "kernel_size": 1},
    "DFLBlock": {
        "in_channels": 256, "reg_max": 7, "num_outputs": 4 * 8, "kernel_size": 1,
    },
    "DistributionProjectBlock": {"reg_max": 7, "num_outputs": 4},
    "NMSFreeDecodeBlock": {
        "score_threshold": 0.05, "max_detections": 100, "top_k": 100,
    },
    "DetectHead": {
        "in_channels": 256, "num_classes": 80, "stacked_convs": 2,
        "activation": "SiLU", "norm": "BatchNorm",
    },

    # --- Graph operators ---
    "Split": {"axis": 1, "split": [1, 1]},
    "Merge": {"mode": "concat", "axis": 1},
    "Route": {"axis": 1},
    "IdentityBlock": {},
    "Identity": {},

    # --- Utility / adaptation ---
    "DropPathBlock": {"drop_prob": 0.0},
    "Dropout": {"p": 0.0, "inplace": False},
    "QuantStubBlock": {"dtype": "qint8"},
    "DeQuantStubBlock": {"dtype": "float32"},

    # --- I/O ---
    "InputBlock": {"channels": 3, "height": 640, "width": 640},
    "OutputBlock": {},
    "StageContainer": {"name": "stage", "stride": 1},
}


def get_block_default_hyperparameters(block_type: str) -> dict:
    """Return a deep copy of recommended hyperparameters for `block_type`.

    Returns an empty dict for unknown block types so the editor never crashes.
    """
    return deepcopy(BLOCK_DEFAULT_HYPERPARAMETERS.get(block_type, {}))


def list_block_default_hyperparameters() -> dict:
    """Return all default hyperparameter sets keyed by block type."""
    return {k: deepcopy(v) for k, v in BLOCK_DEFAULT_HYPERPARAMETERS.items()}


# ---------------------------------------------------------------------------
# Insertion into a scene
# ---------------------------------------------------------------------------


def _next_block_index(scene: dict, prefix: str) -> int:
    used = 0
    for block in scene.get("blocks", []):
        bid = block.get("id", "")
        if bid.startswith(prefix):
            suffix = bid[len(prefix):]
            if suffix.isdigit():
                used = max(used, int(suffix) + 1)
    return used


def _find_anchor_block(scene: dict, anchor_id: str | None) -> dict | None:
    """Locate the block we will attach the template to.

    Preference order:
    1. Explicit `anchor_id`.
    2. The last block whose type is not `OutputBlock`.
    3. None — template is appended without a connecting edge.
    """
    blocks = scene.get("blocks", [])
    if anchor_id is not None:
        for block in blocks:
            if block["id"] == anchor_id:
                return block
        raise KeyError(f"Unknown anchor block id: {anchor_id}")

    candidates = [b for b in blocks if b.get("type") != "OutputBlock"]
    if not candidates:
        return None

    canonical = scene.get("canonicalGraph", {}).get("topoOrder")
    if canonical:
        index = {bid: i for i, bid in enumerate(canonical)}
        candidates.sort(key=lambda b: index.get(b["id"], -1))
    else:
        candidates.sort(key=lambda b: b.get("meta", {}).get("topo_index", 0))
    return candidates[-1]


def insert_template(scene: dict, template_id: str, anchor_id: str | None = None,
                    id_prefix: str | None = None) -> dict:
    """Insert a template into `scene` (mutates `scene` in place).

    Returns ``{"inserted": [block_ids], "edges": n, "anchor": anchor_id}``.
    """
    template = get_template(template_id)
    blocks = template["blocks"]
    if not blocks:
        return {"inserted": [], "edges": 0, "anchor": anchor_id}

    prefix = id_prefix or f"{template_id}_"
    start = _next_block_index(scene, prefix)

    inserted_ids: list[str] = []
    new_blocks: list[dict] = []
    new_edges: list[dict] = []

    previous_id: str | None = None
    previous_tensor: str | None = None

    anchor_block = _find_anchor_block(scene, anchor_id)
    output_block = next(
        (b for b in scene.get("blocks", []) if b.get("type") == "OutputBlock"),
        None,
    )

    # Determine the starting input tensor coming from the scene.
    if anchor_block is not None:
        anchor_out = anchor_block.get("io", {}).get("out", [])
        previous_tensor = anchor_out[0] if anchor_out else f"{anchor_block['id']}_out"
        previous_id = anchor_block["id"]

    for idx, block_spec in enumerate(blocks):
        block_id = f"{prefix}{start + idx}"
        in_tensor = previous_tensor if previous_tensor is not None else f"{block_id}_in"
        out_tensor = f"{block_id}_out"
        new_block = {
            "id": block_id,
            "type": block_spec["type"],
            "position": {"x": 0, "y": 0, "z": 0},
            "params": deepcopy(block_spec.get("params", {})),
            "io": {"in": [in_tensor], "out": [out_tensor]},
        }
        new_blocks.append(new_block)
        inserted_ids.append(block_id)
        if previous_id is not None:
            new_edges.append({"from": previous_id, "to": block_id, "tensor": in_tensor})
        previous_id = block_id
        previous_tensor = out_tensor

    # If a head template is inserted and there is an OutputBlock, splice the
    # template's last output into the OutputBlock so the model stays terminated.
    if template["category"] == "head" and output_block is not None and previous_id is not None:
        # Remove edges that previously fed the OutputBlock and reconnect them
        # — the head replaces whatever used to feed the output.
        scene["edges"] = [e for e in scene.get("edges", []) if e["to"] != output_block["id"]]
        output_block["io"]["in"] = [previous_tensor or output_block["io"]["in"][0]]
        new_edges.append(
            {"from": previous_id, "to": output_block["id"], "tensor": previous_tensor}
        )

    scene.setdefault("blocks", []).extend(new_blocks)
    scene.setdefault("edges", []).extend(new_edges)

    # Re-run auto-repair, scheduling and validation so the scene stays canonical.
    _auto_repair(scene)
    schedule_scene_stages(scene)
    try:
        validate_scene(scene)
    except SceneValidationError:
        # Roll back on validation failure.
        ids_to_drop = set(inserted_ids)
        scene["blocks"] = [b for b in scene["blocks"] if b["id"] not in ids_to_drop]
        scene["edges"] = [
            e
            for e in scene["edges"]
            if e["from"] not in ids_to_drop and e["to"] not in ids_to_drop
        ]
        schedule_scene_stages(scene)
        raise

    return {
        "inserted": inserted_ids,
        "edges": len(new_edges),
        "anchor": anchor_block["id"] if anchor_block else None,
        "template": template_id,
    }
