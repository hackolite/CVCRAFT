"""Tests for CVCRAFT Web UI endpoints."""

from __future__ import annotations

import json
import unittest

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


if __name__ == "__main__":
    unittest.main()
