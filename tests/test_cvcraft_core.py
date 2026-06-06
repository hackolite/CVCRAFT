import json
import tempfile
import unittest
from pathlib import Path

from cvcraft.editor import cut_blocks, prune_block
from cvcraft.exporter import export_scene_yaml
from cvcraft.validator import SceneValidationError, validate_scene


def sample_scene():
    return {
        "scene": {"version": "1.0.0", "units": "voxel", "grid": {"chunkSize": 16, "worldSize": [64, 64, 64]}},
        "model": {"name": "yolox_s", "family": "YOLOX", "anchorFree": True, "inputShape": [1, 3, 640, 640], "classCount": 80},
        "blocks": [
            {"id": "b0", "type": "InputBlock", "position": {"x": 0, "y": 0, "z": 0}, "params": {}, "io": {"in": [], "out": ["x"]}},
            {"id": "b1", "type": "StageContainer", "position": {"x": 1, "y": 0, "z": 0}, "params": {}, "io": {"in": ["x"], "out": ["p3"]}},
            {"id": "b2", "type": "DecoupledHeadBlock", "position": {"x": 2, "y": 0, "z": 0}, "params": {"head_channels": 256}, "io": {"in": ["p3"], "out": ["pred"]}},
        ],
        "edges": [{"from": "b0", "to": "b1", "tensor": "x"}, {"from": "b1", "to": "b2", "tensor": "p3"}],
        "metrics": {"flops": 1.0, "parameters": 10, "latencyMs": {"cpu": 1.1}},
    }


class TestCVCraftCore(unittest.TestCase):
    def test_validate_scene(self):
        validate_scene(sample_scene())

    def test_validate_scene_rejects_non_anchor_free(self):
        scene = sample_scene()
        scene["model"]["anchorFree"] = False
        with self.assertRaises(SceneValidationError):
            validate_scene(scene)

    def test_cut_and_prune(self):
        scene = sample_scene()
        cut_blocks(scene, {"b1"})
        prune_block(scene, "b2", "head_channels", 192)
        self.assertEqual(scene["blocks"][-1]["params"]["head_channels"], 192)

    def test_yaml_export(self):
        text = export_scene_yaml(sample_scene())
        self.assertIn("anchor_free: true", text)
        self.assertIn("head:", text)

    def test_scene_json_roundtrip(self):
        scene = sample_scene()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "scene.json"
            p.write_text(json.dumps(scene), encoding="utf-8")
            loaded = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(loaded["model"]["family"], "YOLOX")


if __name__ == "__main__":
    unittest.main()
