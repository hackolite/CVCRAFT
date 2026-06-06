"""Structured PyTorch export."""

from __future__ import annotations

import json

from .scene_v2 import normalize_scene_v2


def _module_for_block(block: dict) -> str:
    block_type = block["type"]
    if block_type in {"Conv2dBlock", "DWConvBlock", "PWConvBlock", "FusedConvBlock"}:
        return "nn.Conv2d(1, 1, kernel_size=1)"
    if block_type in {"BatchNormBlock", "GroupNormBlock", "SyncBNBlock"}:
        return "nn.BatchNorm2d(1)"
    if block_type == "ReLUBlock":
        return "nn.ReLU()"
    if block_type == "SiLUBlock":
        return "nn.SiLU()"
    if block_type == "SigmoidBlock":
        return "nn.Sigmoid()"
    if block_type == "PoolingBlock":
        return "nn.MaxPool2d(kernel_size=2)"
    if block_type == "UpsampleBlock":
        return "nn.Upsample(scale_factor=2, mode='nearest')"
    return "nn.Identity()"


def export_scene_pytorch(scene: dict, module_name: str = "GeneratedVoxelModel") -> dict[str, str]:
    normalize_scene_v2(scene)
    topo_order = scene["canonicalGraph"]["topoOrder"]
    block_by_id = {b["id"]: b for b in scene["blocks"]}

    stages: dict[str, list[str]] = {"input": [], "backbone": [], "neck": [], "head": [], "output": []}
    for block_id in topo_order:
        stage = block_by_id[block_id]["meta"]["stage_id"]
        stages.setdefault(stage, []).append(block_id)

    lines = [
        "import torch",
        "import torch.nn as nn",
        "",
        f"class {module_name}(nn.Module):",
        "    def __init__(self):",
        "        super().__init__()",
    ]

    for stage_name in ("backbone", "neck", "head"):
        lines.append(f"        self.{stage_name} = nn.ModuleDict({{")
        for block_id in stages.get(stage_name, []):
            module_expr = _module_for_block(block_by_id[block_id])
            lines.append(f"            '{block_id}': {module_expr},")
        lines.append("        })")

    lines.extend(
        [
            "",
            "    def forward(self, x):",
            "        outputs = {'input': x}",
            "        for stage in (self.backbone, self.neck, self.head):",
            "            for block_id, block in stage.items():",
            "                x = block(x)",
            "                outputs[block_id] = x",
            "        return x",
            "",
        ]
    )

    config = {
        "module_name": module_name,
        "model": scene["model"],
        "stages": stages,
        "canonical_graph": scene["canonicalGraph"],
    }
    return {"python": "\n".join(lines), "config": json.dumps(config, indent=2) + "\n"}
