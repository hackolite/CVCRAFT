"""Tests for new IR features: quantization, LR scaling, component freeze, edge constraints, ONNX export."""

import json
import tempfile
import unittest
from pathlib import Path

from cvcraft.optimization import (
    dequantize_scene,
    freeze_component,
    quantize_scene,
    set_component_lr_scale,
    set_edge_constraints,
    set_lr_scale,
    set_pretrained_config,
)
from cvcraft.validator import SceneValidationError, validate_scene


def sample_scene():
    return {
        "scene": {"version": "2.0.0", "units": "voxel", "grid": {"chunkSize": 16, "worldSize": [64, 64, 64]}},
        "model": {"name": "yolox_s", "family": "YOLOX", "anchorFree": True, "inputShape": [1, 3, 640, 640], "classCount": 80},
        "blocks": [
            {"id": "b0", "type": "InputBlock", "position": {"x": 0, "y": 0, "z": 0}, "params": {}, "io": {"in": [], "out": ["x"]}},
            {"id": "b1", "type": "Conv2dBlock", "position": {"x": 1, "y": 0, "z": 0}, "params": {"has_weights": True}, "io": {"in": ["x"], "out": ["p3"]}},
            {"id": "b2", "type": "PANBlock", "position": {"x": 2, "y": 0, "z": 0}, "params": {}, "io": {"in": ["p3"], "out": ["neck_out"]}},
            {"id": "b3", "type": "DecoupledHeadBlock", "position": {"x": 3, "y": 0, "z": 0}, "params": {"head_channels": 256}, "io": {"in": ["neck_out"], "out": ["pred"]}},
        ],
        "edges": [
            {"from": "b0", "to": "b1", "tensor": "x"},
            {"from": "b1", "to": "b2", "tensor": "p3"},
            {"from": "b2", "to": "b3", "tensor": "neck_out"},
        ],
        "metrics": {"flops": 1.0, "parameters": 500000, "latencyMs": {"cpu": 5.0}},
    }


class TestQuantization(unittest.TestCase):
    def test_quantize_fp16(self):
        scene = sample_scene()
        validate_scene(scene)
        result = quantize_scene(scene, mode="fp16")
        self.assertEqual(result["mode"], "fp16")
        self.assertIn("b1", result["quantized"])
        self.assertIn("b2", result["quantized"])
        # InputBlock should not be quantized
        self.assertNotIn("b0", result["quantized"])
        # Check metadata
        self.assertEqual(scene["metadata"]["quantization"]["mode"], "fp16")

    def test_quantize_int8(self):
        scene = sample_scene()
        validate_scene(scene)
        result = quantize_scene(scene, mode="int8")
        self.assertEqual(result["mode"], "int8")
        for block in scene["blocks"]:
            if block["type"] not in {"InputBlock", "OutputBlock", "IdentityBlock", "StageContainer"}:
                self.assertEqual(block["meta"]["quantization"], "int8")

    def test_quantize_invalid_mode(self):
        scene = sample_scene()
        validate_scene(scene)
        with self.assertRaises(ValueError):
            quantize_scene(scene, mode="invalid")

    def test_dequantize(self):
        scene = sample_scene()
        validate_scene(scene)
        quantize_scene(scene, mode="fp16")
        result = dequantize_scene(scene)
        self.assertTrue(len(result["dequantized"]) > 0)
        for block in scene["blocks"]:
            self.assertNotIn("quantization", block.get("meta", {}))


class TestLRScaling(unittest.TestCase):
    def test_set_lr_scale_per_block(self):
        scene = sample_scene()
        validate_scene(scene)
        result = set_lr_scale(scene, {"b1", "b2"}, 0.1)
        self.assertEqual(len(result["updated"]), 2)
        for block in scene["blocks"]:
            if block["id"] in {"b1", "b2"}:
                self.assertEqual(block["meta"]["lr_scale"], 0.1)

    def test_set_lr_scale_invalid_ids(self):
        scene = sample_scene()
        validate_scene(scene)
        with self.assertRaises(ValueError):
            set_lr_scale(scene, {"nonexistent"}, 0.5)

    def test_set_lr_scale_negative(self):
        scene = sample_scene()
        validate_scene(scene)
        with self.assertRaises(ValueError):
            set_lr_scale(scene, {"b1"}, -0.1)

    def test_set_component_lr_scale(self):
        scene = sample_scene()
        validate_scene(scene)
        result = set_component_lr_scale(scene, "backbone", 0.01)
        self.assertEqual(result["component"], "backbone")
        # b1 is a Conv2dBlock in backbone
        backbone_blocks = [u for u in result["updated"]]
        self.assertTrue(len(backbone_blocks) > 0)

    def test_set_component_lr_scale_invalid(self):
        scene = sample_scene()
        validate_scene(scene)
        with self.assertRaises(ValueError):
            set_component_lr_scale(scene, "invalid_component", 0.1)


class TestComponentFreeze(unittest.TestCase):
    def test_freeze_backbone(self):
        scene = sample_scene()
        validate_scene(scene)
        result = freeze_component(scene, "backbone", frozen=True)
        self.assertEqual(result["component"], "backbone")
        self.assertTrue(result["frozen"])
        for entry in result["updated"]:
            block = next(b for b in scene["blocks"] if b["id"] == entry["id"])
            self.assertTrue(block["meta"]["frozen"])

    def test_unfreeze_component(self):
        scene = sample_scene()
        validate_scene(scene)
        freeze_component(scene, "head", frozen=True)
        result = freeze_component(scene, "head", frozen=False)
        for entry in result["updated"]:
            block = next(b for b in scene["blocks"] if b["id"] == entry["id"])
            self.assertFalse(block["meta"]["frozen"])

    def test_freeze_invalid_component(self):
        scene = sample_scene()
        validate_scene(scene)
        with self.assertRaises(ValueError):
            freeze_component(scene, "invalid")


class TestEdgeConstraints(unittest.TestCase):
    def test_set_max_size(self):
        scene = sample_scene()
        validate_scene(scene)
        result = set_edge_constraints(scene, max_size_mb=2.0)
        self.assertEqual(result["edge_constraints"]["max_size_mb"], 2.0)
        self.assertIn("estimated_size_mb", result)
        self.assertIn("constraints_met", result)

    def test_set_latency_target(self):
        scene = sample_scene()
        validate_scene(scene)
        result = set_edge_constraints(scene, latency_target_ms=10.0)
        self.assertEqual(result["edge_constraints"]["latency_target_ms"], 10.0)

    def test_size_violation(self):
        scene = sample_scene()
        # Set a very large parameter count
        scene["metrics"]["parameters"] = 100_000_000  # 100M params ~= 400MB
        validate_scene(scene)
        result = set_edge_constraints(scene, max_size_mb=2.0)
        self.assertFalse(result["constraints_met"])
        self.assertTrue(len(result["violations"]) > 0)

    def test_invalid_constraints(self):
        scene = sample_scene()
        validate_scene(scene)
        with self.assertRaises(ValueError):
            set_edge_constraints(scene, max_size_mb=-1.0)
        with self.assertRaises(ValueError):
            set_edge_constraints(scene, latency_target_ms=0)


class TestPretrainedConfig(unittest.TestCase):
    def test_set_pretrained_config(self):
        scene = sample_scene()
        validate_scene(scene)
        result = set_pretrained_config(scene, source="yolox_s.onnx", inherit_weights=True)
        self.assertEqual(result["pretrained"]["source"], "yolox_s.onnx")
        self.assertTrue(result["pretrained"]["inherit_weights"])
        # b1 has has_weights=True
        self.assertIn("b1", result["pretrained"]["inheritable_blocks"])

    def test_pretrained_strict_mode(self):
        scene = sample_scene()
        validate_scene(scene)
        result = set_pretrained_config(scene, strict=True)
        self.assertTrue(result["pretrained"]["strict"])


class TestModelFields(unittest.TestCase):
    def test_task_and_detection_type_set_by_validation(self):
        scene = sample_scene()
        validate_scene(scene)
        self.assertEqual(scene["model"]["task"], "object_detection")
        self.assertEqual(scene["model"]["detection_type"], "anchor_free")

    def test_invalid_task_rejected(self):
        scene = sample_scene()
        scene["model"]["task"] = "classification"
        with self.assertRaises(SceneValidationError):
            validate_scene(scene)

    def test_invalid_detection_type_rejected(self):
        scene = sample_scene()
        scene["model"]["detection_type"] = "anchor_based"
        with self.assertRaises(SceneValidationError):
            validate_scene(scene)


class TestOnnxExport(unittest.TestCase):
    def test_export_onnx_scene(self):
        try:
            import onnx
        except ImportError:
            self.skipTest("onnx is not installed")
        from cvcraft.onnx_exporter import export_scene_onnx

        scene = sample_scene()
        validate_scene(scene)
        onnx_bytes = export_scene_onnx(scene)
        self.assertIsInstance(onnx_bytes, bytes)
        self.assertTrue(len(onnx_bytes) > 0)

        # Verify it's a valid ONNX model
        model = onnx.load_from_string(onnx_bytes)
        self.assertEqual(model.producer_name, "cvcraft")
        self.assertTrue(len(model.graph.node) > 0)

    def test_export_onnx_roundtrip(self):
        try:
            import onnx
        except ImportError:
            self.skipTest("onnx is not installed")
        from cvcraft.onnx_exporter import export_scene_onnx
        from cvcraft.onnx_importer import import_onnx_scene

        scene = sample_scene()
        validate_scene(scene)
        onnx_bytes = export_scene_onnx(scene)

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "exported.onnx"
            p.write_bytes(onnx_bytes)
            reimported = import_onnx_scene(str(p), model_name="roundtrip")

        validate_scene(reimported)
        self.assertEqual(reimported["model"]["name"], "roundtrip")


if __name__ == "__main__":
    unittest.main()
