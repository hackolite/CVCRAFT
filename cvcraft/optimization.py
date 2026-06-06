"""Optimization utilities: quantization, LR scaling, edge constraints, component freeze."""

from __future__ import annotations

from .scene_v2 import normalize_scene_v2


# --- Quantization ---

QUANTIZATION_MODES = {"int8", "fp16"}


def quantize_scene(scene: dict, mode: str = "fp16") -> dict:
    """Apply quantization metadata to all eligible blocks.

    Args:
        scene: The Scene V2 IR dict.
        mode: One of 'int8' or 'fp16'.

    Returns:
        Summary dict with quantized block ids.
    """
    if mode not in QUANTIZATION_MODES:
        raise ValueError(f"Unsupported quantization mode: {mode}. Must be one of {sorted(QUANTIZATION_MODES)}")

    normalize_scene_v2(scene)
    quantized = []
    for block in scene["blocks"]:
        block_type = block["type"]
        # Skip structural blocks that don't carry compute
        if block_type in {"InputBlock", "OutputBlock", "IdentityBlock", "StageContainer"}:
            continue
        meta = block.setdefault("meta", {})
        meta["quantization"] = mode
        quantized.append(block["id"])

    # Update model-level metadata
    scene.setdefault("metadata", {})["quantization"] = {
        "mode": mode,
        "quantized_blocks": len(quantized),
        "total_blocks": len(scene["blocks"]),
    }
    return {"mode": mode, "quantized": quantized}


def dequantize_scene(scene: dict) -> dict:
    """Remove quantization metadata from all blocks."""
    removed = []
    for block in scene["blocks"]:
        meta = block.get("meta", {})
        if "quantization" in meta:
            del meta["quantization"]
            removed.append(block["id"])
    scene.get("metadata", {}).pop("quantization", None)
    return {"dequantized": removed}


# --- LR Scaling ---

def set_lr_scale(scene: dict, block_ids: set[str], lr_scale: float) -> dict:
    """Set learning rate scaling factor for specific blocks.

    Args:
        scene: The Scene V2 IR dict.
        block_ids: Set of block IDs to apply LR scaling to.
        lr_scale: Multiplier for the learning rate (e.g. 0.1 for 10x lower LR).

    Returns:
        Summary of updated blocks.
    """
    if lr_scale < 0:
        raise ValueError("lr_scale must be non-negative")

    updated = []
    for block in scene["blocks"]:
        if block["id"] in block_ids:
            meta = block.setdefault("meta", {})
            meta["lr_scale"] = lr_scale
            updated.append({"id": block["id"], "lr_scale": lr_scale})
    if not updated:
        raise ValueError(f"No matching blocks found for ids: {sorted(block_ids)}")
    return {"updated": updated}


def set_component_lr_scale(scene: dict, component: str, lr_scale: float) -> dict:
    """Set learning rate scaling for all blocks in a component (backbone/neck/head).

    Args:
        scene: The Scene V2 IR dict.
        component: One of 'backbone', 'neck', 'head'.
        lr_scale: Multiplier for the learning rate.

    Returns:
        Summary of updated blocks.
    """
    valid_components = {"backbone", "neck", "head"}
    if component not in valid_components:
        raise ValueError(f"Invalid component: {component}. Must be one of {sorted(valid_components)}")
    if lr_scale < 0:
        raise ValueError("lr_scale must be non-negative")

    normalize_scene_v2(scene)
    updated = []
    for block in scene["blocks"]:
        meta = block.get("meta", {})
        if meta.get("stage_id") == component:
            meta["lr_scale"] = lr_scale
            updated.append({"id": block["id"], "lr_scale": lr_scale})
    return {"component": component, "updated": updated}


# --- Component-level Freeze/Unfreeze ---

def freeze_component(scene: dict, component: str, frozen: bool = True) -> dict:
    """Freeze or unfreeze all blocks belonging to a component.

    Args:
        scene: The Scene V2 IR dict.
        component: One of 'backbone', 'neck', 'head'.
        frozen: True to freeze, False to unfreeze.

    Returns:
        Summary of updated blocks.
    """
    valid_components = {"backbone", "neck", "head"}
    if component not in valid_components:
        raise ValueError(f"Invalid component: {component}. Must be one of {sorted(valid_components)}")

    normalize_scene_v2(scene)
    updated = []
    for block in scene["blocks"]:
        meta = block.get("meta", {})
        if meta.get("stage_id") == component:
            meta["frozen"] = frozen
            updated.append({"id": block["id"], "frozen": frozen})
    return {"component": component, "frozen": frozen, "updated": updated}


# --- Edge Deployment Constraints ---

def set_edge_constraints(scene: dict, *, max_size_mb: float | None = None, latency_target_ms: float | None = None) -> dict:
    """Set edge deployment constraints on the scene.

    Args:
        scene: The Scene V2 IR dict.
        max_size_mb: Maximum model size in megabytes (e.g. 2.0).
        latency_target_ms: Target inference latency in milliseconds.

    Returns:
        The deployment constraints and validation result.
    """
    deployment = scene.setdefault("deployment", {})
    constraints = deployment.setdefault("edge_constraints", {})

    if max_size_mb is not None:
        if max_size_mb <= 0:
            raise ValueError("max_size_mb must be positive")
        constraints["max_size_mb"] = max_size_mb

    if latency_target_ms is not None:
        if latency_target_ms <= 0:
            raise ValueError("latency_target_ms must be positive")
        constraints["latency_target_ms"] = latency_target_ms

    # Estimate current model size from parameters
    params = scene.get("metrics", {}).get("parameters", 0)
    # Rough estimate: 4 bytes per parameter (fp32), halved for fp16
    quant_mode = scene.get("metadata", {}).get("quantization", {}).get("mode")
    bytes_per_param = 1 if quant_mode == "int8" else (2 if quant_mode == "fp16" else 4)
    estimated_size_mb = (params * bytes_per_param) / (1024 * 1024)

    deployment["estimated_size_mb"] = round(estimated_size_mb, 4)

    violations = []
    if max_size_mb is not None and estimated_size_mb > max_size_mb:
        violations.append(f"Estimated size {estimated_size_mb:.4f}MB exceeds limit {max_size_mb}MB")

    deployment["constraints_met"] = len(violations) == 0
    deployment["violations"] = violations

    return {
        "edge_constraints": constraints,
        "estimated_size_mb": deployment["estimated_size_mb"],
        "constraints_met": deployment["constraints_met"],
        "violations": violations,
    }


# --- Pretrained Weight Inheritance ---

def set_pretrained_config(scene: dict, *, source: str | None = None, inherit_weights: bool = True, strict: bool = False) -> dict:
    """Configure pretrained weight inheritance for the model.

    Args:
        scene: The Scene V2 IR dict.
        source: Path or identifier of the pretrained model source.
        inherit_weights: Whether to inherit weights from source.
        strict: Whether to require strict shape matching.

    Returns:
        The pretrained configuration.
    """
    training = scene.setdefault("training", {})
    pretrained = training.setdefault("pretrained", {})

    pretrained["inherit_weights"] = inherit_weights
    pretrained["strict"] = strict
    if source is not None:
        pretrained["source"] = source

    # Mark which blocks can inherit weights
    inheritable = []
    for block in scene["blocks"]:
        params = block.get("params", {})
        has_weights = params.get("has_weights", False)
        meta = block.setdefault("meta", {})
        meta["weight_inheritable"] = has_weights
        if has_weights:
            inheritable.append(block["id"])

    pretrained["inheritable_blocks"] = inheritable
    pretrained["inheritable_count"] = len(inheritable)

    return {
        "pretrained": pretrained,
    }
