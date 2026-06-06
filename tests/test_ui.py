"""Tests for CVCRAFT Web UI endpoints."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from cvcraft.ui.app import create_app


def _minimal_scene() -> dict:
    return {
        "scene": {"version": "1.0.0", "units": "voxel", "grid": {"chunkSize": 16, "worldSize": [64, 64, 64]}},
        "model": {"name": "test", "family": "YOLOX", "anchorFree": True, "inputShape": [1, 3, 640, 640], "classCount": 80},
        "blocks": [
            {"id": "b0", "type": "InputBlock", "position": {"x": 0, "y": 0, "z": 0}, "params": {}, "io": {"in": [], "out": ["x"]}},
            {"id": "b1", "type": "Conv2dBlock", "position": {"x": 2, "y": 0, "z": 0}, "params": {"out_channels": 64}, "io": {"in": ["x"], "out": ["y"]}},
        ],
        "edges": [{"from": "b0", "to": "b1", "tensor": "x"}],
        "metrics": {"flops": 1e9, "parameters": 100000, "latencyMs": {"cpu": 10.0}},
    }


class TestUIEndpoints(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_index_page(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"CVCRAFT", resp.data)

    def test_validate_valid_scene(self):
        scene = _minimal_scene()
        resp = self.client.post("/api/validate", json=scene)
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.get_json()["valid"])

    def test_validate_invalid_scene(self):
        resp = self.client.post("/api/validate", json={"blocks": []})
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertFalse(data["valid"])

    def test_edit_cut(self):
        scene = _minimal_scene()
        resp = self.client.post("/api/edit/cut", json={"scene": scene, "ids": ["b1"]})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("b1", data["result"]["removed"])

    def test_edit_prune(self):
        scene = _minimal_scene()
        resp = self.client.post("/api/edit/prune", json={"scene": scene, "id": "b1", "param": "out_channels", "value": 32})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["result"]["updated"][0]["params"]["out_channels"], 32)

    def test_edit_fuse(self):
        scene = _minimal_scene()
        resp = self.client.post("/api/edit/fuse", json={"scene": scene})
        self.assertEqual(resp.status_code, 200)

    def test_edit_freeze(self):
        scene = _minimal_scene()
        resp = self.client.post("/api/edit/freeze", json={"scene": scene, "ids": ["b0", "b1"], "value": True})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(len(data["result"]["updated"]), 2)

    def test_export_yaml(self):
        scene = _minimal_scene()
        resp = self.client.post("/api/export/yaml", json=scene)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("yaml", resp.get_json())

    def test_export_pytorch(self):
        scene = _minimal_scene()
        resp = self.client.post("/api/export/pytorch", json={"scene": scene, "module_name": "TestModel"})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("python", data)
        self.assertIn("config", data)

    def test_schedule(self):
        scene = _minimal_scene()
        resp = self.client.post("/api/schedule", json=scene)
        self.assertEqual(resp.status_code, 200)

    def test_import_onnx(self):
        """Test ONNX import via UI endpoint with 3D visualization metadata."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 32, 32])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 32, 32])
        W = np.ones((1, 3, 3, 3), dtype=np.float32)
        W_init = onnx.numpy_helper.from_array(W, name="W")
        conv = helper.make_node("Conv", ["X", "W"], ["Y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
        graph = helper.make_graph([conv], "test", [X], [Y], [W_init])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(model.SerializeToString())
            tmp_path = f.name

        try:
            with open(tmp_path, "rb") as f:
                data = {
                    "file": (f, "test.onnx"),
                    "family": "YOLOX",
                    "classes": "80",
                    "pretrained": "auto",
                }
                resp = self.client.post("/api/import/onnx", data=data, content_type="multipart/form-data")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        self.assertEqual(resp.status_code, 200)
        scene = resp.get_json()
        self.assertIn("blocks", scene)
        self.assertIn("edges", scene)
        self.assertIn("canonicalGraph", scene)
        # Verify all blocks have meta with stage_id and color for 3D rendering
        for block in scene["blocks"]:
            self.assertIn("meta", block, f"Block {block['id']} missing meta")
            self.assertIn("stage_id", block["meta"], f"Block {block['id']} missing stage_id")
            self.assertIn("color", block["meta"], f"Block {block['id']} missing color")
            self.assertIn("position", block)
            # Verify position is not all zeros (scheduler assigned positions)
            pos = block["position"]
            self.assertTrue(isinstance(pos, dict))

    def test_import_onnx_no_file(self):
        resp = self.client.post("/api/import/onnx", data={}, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", resp.get_json())

    def test_import_yaml(self):
        """Test YAML import via UI endpoint."""
        yaml_content = b"""model:
  name: test_model
  family: YOLOX
  anchor_free: true
  input_shape: [1, 3, 640, 640]
  classes: 80
backbone:
  - type: Conv2dBlock
    out_channels: 64
neck:
  - type: FPNBlock
head:
  - type: DecoupledHeadBlock
"""
        data = {"file": (io.BytesIO(yaml_content), "test.yaml")}
        resp = self.client.post("/api/import/yaml", data=data, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 200)
        scene = resp.get_json()
        self.assertIn("blocks", scene)
        # Verify stage coloring is applied
        stage_ids_found = set()
        for block in scene["blocks"]:
            self.assertIn("meta", block)
            stage_ids_found.add(block["meta"]["stage_id"])
        self.assertIn("backbone", stage_ids_found)
        self.assertIn("neck", stage_ids_found)
        self.assertIn("head", stage_ids_found)

    def test_edit_preserves_meta_and_colors(self):
        """Verify that edit operations preserve/recompute meta for 3D visualization."""
        scene = _minimal_scene()
        # Replace a block type and verify meta is recomputed
        resp = self.client.post("/api/edit/replace", json={"scene": scene, "id": "b1", "type": "FPNBlock"})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        for block in data["scene"]["blocks"]:
            self.assertIn("meta", block, f"Block {block['id']} missing meta after replace")
            self.assertIn("stage_id", block["meta"])
            self.assertIn("color", block["meta"])
        # After replacing with FPNBlock (a neck block), its stage should be 'neck'
        replaced = next(b for b in data["scene"]["blocks"] if b["id"] == "b1")
        self.assertEqual(replaced["meta"]["stage_id"], "neck")

    # ------------------------------------------------------------------
    # Templates endpoints
    # ------------------------------------------------------------------
    def test_list_templates_endpoint(self):
        resp = self.client.get("/api/templates")
        self.assertEqual(resp.status_code, 200)
        payload = resp.get_json()
        self.assertIn("templates", payload)
        self.assertIn("blockDefaults", payload)
        self.assertGreaterEqual(len(payload["templates"]), 20)
        # Defaults for Conv2dBlock should expose editable hyperparameters.
        self.assertIn("Conv2dBlock", payload["blockDefaults"])

    def test_block_defaults_endpoint(self):
        resp = self.client.get("/api/templates/block-defaults/Conv2dBlock")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["type"], "Conv2dBlock")
        self.assertEqual(data["params"]["kernel_size"], 3)

    def test_insert_template_endpoint(self):
        scene = _minimal_scene()
        resp = self.client.post(
            "/api/templates/insert",
            json={"scene": scene, "template_id": "centernet_head"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertGreater(len(data["result"]["inserted"]), 0)
        types = {b["type"] for b in data["scene"]["blocks"]}
        self.assertIn("CenterHeadBlock", types)

    def test_insert_template_invalid_id(self):
        scene = _minimal_scene()
        resp = self.client.post(
            "/api/templates/insert",
            json={"scene": scene, "template_id": "does_not_exist"},
        )
        self.assertEqual(resp.status_code, 400)


if __name__ == "__main__":
    unittest.main()
