"""Structured PyTorch export."""

from __future__ import annotations

import json

from .constants import STAGE_ORDER
from .scene_v2 import normalize_scene_v2


def _module_for_block(block: dict) -> str:
    """Return a stage-module constructor string for a block type using safe placeholders."""
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
    canonical_edges = scene["canonicalGraph"]["edges"]

    stages: dict[str, list[str]] = {stage: [] for stage in STAGE_ORDER}
    for block_id in topo_order:
        stage = block_by_id[block_id]["meta"]["stage_id"]
        stages.setdefault(stage, []).append(block_id)

    incoming: dict[str, list[str]] = {block_id: [] for block_id in topo_order}
    for edge in canonical_edges:
        incoming.setdefault(edge["to"], []).append(edge["from"])
    for block_id in incoming:
        incoming[block_id] = sorted(set(incoming[block_id]))

    stage_by_block = {block_id: block_by_id[block_id]["meta"]["stage_id"] for block_id in topo_order}
    block_types = {block_id: block_by_id[block_id]["type"] for block_id in topo_order}
    frozen_blocks = sorted(
        block_id
        for block_id in topo_order
        if bool(block_by_id[block_id].get("meta", {}).get("frozen", block_by_id[block_id].get("params", {}).get("frozen", False)))
    )

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
            f"        self.topo_order = {json.dumps(topo_order)}",
            f"        self.incoming = {json.dumps(incoming, sort_keys=True)}",
            f"        self.stage_by_block = {json.dumps(stage_by_block, sort_keys=True)}",
            f"        self.block_types = {json.dumps(block_types, sort_keys=True)}",
            f"        self.frozen_blocks = {json.dumps(frozen_blocks)}",
            "        self._apply_freeze()",
            "",
            "    def _apply_freeze(self):",
            "        for block_id in self.frozen_blocks:",
            "            module = self._get_block_module(block_id)",
            "            for param in module.parameters():",
            "                param.requires_grad = False",
            "",
            "    def _get_block_module(self, block_id):",
            "        stage = self.stage_by_block.get(block_id)",
            "        if stage == 'backbone':",
            "            return self.backbone[block_id] if block_id in self.backbone else nn.Identity()",
            "        if stage == 'neck':",
            "            return self.neck[block_id] if block_id in self.neck else nn.Identity()",
            "        if stage == 'head':",
            "            return self.head[block_id] if block_id in self.head else nn.Identity()",
            "        return nn.Identity()",
        ]
    )

    lines.extend(
        [
            "",
            "    def forward(self, x):",
            "        tensors = {}",
            "        for block_id in self.topo_order:",
            "            block_type = self.block_types[block_id]",
            "            if block_type == 'InputBlock':",
            "                tensors[block_id] = x",
            "                continue",
            "            parents = self.incoming.get(block_id, [])",
            "            if not parents:",
            "                node_in = x",
            "            elif len(parents) == 1:",
            "                node_in = tensors[parents[0]]",
            "            else:",
            "                parent_tensors = [tensors[p] for p in parents]",
            "                if block_type == 'ConcatBlock':",
            "                    node_in = torch.cat(parent_tensors, dim=1)",
            "                elif block_type == 'MulBlock':",
            "                    node_in = parent_tensors[0]",
            "                    for t in parent_tensors[1:]:",
            "                        node_in = node_in * t",
            "                elif block_type == 'AddBlock':",
            "                    node_in = parent_tensors[0]",
            "                    for t in parent_tensors[1:]:",
            "                        node_in = node_in + t",
            "                else:",
            "                    node_in = parent_tensors[0]",
            "            if block_type == 'OutputBlock':",
            "                tensors[block_id] = node_in",
            "                continue",
            "            tensors[block_id] = self._get_block_module(block_id)(node_in)",
            "        return tensors[self.topo_order[-1]] if self.topo_order else x",
            "",
        ]
    )

    config = {
        "module_name": module_name,
        "model": scene["model"],
        "stages": stages,
        "canonical_graph": scene["canonicalGraph"],
        "frozen_blocks": frozen_blocks,
    }
    return {"python": "\n".join(lines), "config": json.dumps(config, indent=2) + "\n"}
