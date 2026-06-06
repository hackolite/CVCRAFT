import json
import tempfile
import unittest
from pathlib import Path

from cvcraft.editor import cut_blocks, prune_block, set_blocks_frozen
from cvcraft.exporter import export_scene_yaml
from cvcraft.onnx_importer import _block_type_for_op, import_onnx_scene
from cvcraft.pytorch_exporter import export_scene_pytorch
from cvcraft.scheduler import schedule_scene_stages
from cvcraft.validator import SceneValidationError, validate_scene
from cvcraft.yaml_importer import import_yaml_scene


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
        scene = sample_scene()
        validate_scene(scene)
        self.assertEqual(scene["schemaVersion"], "2.0.0")
        self.assertIn("canonicalGraph", scene)
        self.assertIn("metadata", scene)
        self.assertIn("topoOrder", scene["canonicalGraph"])

    def test_validate_scene_rejects_non_anchor_free(self):
        scene = sample_scene()
        scene["model"]["anchorFree"] = False
        with self.assertRaises(SceneValidationError):
            validate_scene(scene)

    def test_cut_and_prune(self):
        scene = sample_scene()
        cut_blocks(scene, {"b1"})
        prune_block(scene, "b2", "head_channels", 192)
        validate_scene(scene)
        self.assertEqual(scene["blocks"][-1]["params"]["head_channels"], 192)

    def test_yaml_export(self):
        text = export_scene_yaml(sample_scene())
        self.assertIn("anchor_free: true", text)
        self.assertIn("head:", text)

    def test_schedule_is_deterministic(self):
        scene_a = sample_scene()
        scene_b = sample_scene()
        schedule_scene_stages(scene_a)
        schedule_scene_stages(scene_b)
        pos_a = {b["id"]: b["position"] for b in scene_a["blocks"]}
        pos_b = {b["id"]: b["position"] for b in scene_b["blocks"]}
        self.assertEqual(pos_a, pos_b)

    def test_export_pytorch_structured(self):
        exported = export_scene_pytorch(sample_scene(), module_name="MyModel")
        self.assertIn("class MyModel(nn.Module)", exported["python"])
        self.assertIn("self.backbone = nn.ModuleDict", exported["python"])
        config = json.loads(exported["config"])
        self.assertIn("stages", config)
        self.assertIn("canonical_graph", config)

    def test_scene_json_roundtrip(self):
        scene = sample_scene()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "scene.json"
            p.write_text(json.dumps(scene), encoding="utf-8")
            loaded = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(loaded, scene)

    def test_onnx_op_mapping(self):
        self.assertEqual(_block_type_for_op("Conv"), "Conv2dBlock")
        self.assertEqual(_block_type_for_op("Relu"), "ReLUBlock")
        self.assertEqual(_block_type_for_op("CustomOp"), "UnsupportedOpBlock")

    def test_import_onnx_scene(self):
        try:
            import onnx
        except ImportError:
            self.skipTest("onnx is not installed")
        helper = onnx.helper
        tensor_type = onnx.TensorProto.FLOAT
        graph = helper.make_graph(
            nodes=[
                helper.make_node("Relu", ["x"], ["y"], name="relu_0"),
            ],
            name="g",
            inputs=[helper.make_tensor_value_info("x", tensor_type, [1, 3, 8, 8])],
            outputs=[helper.make_tensor_value_info("y", tensor_type, [1, 3, 8, 8])],
        )
        model = helper.make_model(graph)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "toy.onnx"
            onnx.save(model, str(p))
            scene = import_onnx_scene(str(p), model_name="toy")
        validate_scene(scene)
        self.assertEqual(scene["model"]["name"], "toy")
        self.assertTrue(any(b["type"] == "ReLUBlock" for b in scene["blocks"]))
        self.assertEqual(Path(scene["metadata"]["onnx"]["source_path"]).name, "toy.onnx")
        self.assertIn("pretrained", scene["metadata"]["onnx"])

    def test_set_blocks_frozen_in_export_pytorch(self):
        scene = sample_scene()
        set_blocks_frozen(scene, {"b2"}, True)
        exported = export_scene_pytorch(scene, module_name="FrozenModel")
        self.assertIn("self._apply_freeze()", exported["python"])
        config = json.loads(exported["config"])
        self.assertIn("b2", config["frozen_blocks"])

    def test_import_yaml_and_export_pytorch(self):
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML is not installed")
        yaml_content = """
model:
  name: tiny_model
  family: YOLOX
  anchor_free: true
  input_shape: [1, 3, 32, 32]
  classes: 2
backbone:
  - {type: Conv2dBlock}
neck:
  - {type: PANBlock}
head:
  - {type: DecoupledHeadBlock}
"""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "model.yaml"
            p.write_text(yaml_content, encoding="utf-8")
            scene = import_yaml_scene(str(p))
        validate_scene(scene)
        exported = export_scene_pytorch(scene, module_name="FromYaml")
        self.assertIn("class FromYaml(nn.Module)", exported["python"])

    def test_import_yaml_invalid_format(self):
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML is not installed")
        invalid_yaml = """
model:
  name: bad
  family: YOLOX
backbone:
  type: Conv2dBlock
"""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.yaml"
            p.write_text(invalid_yaml, encoding="utf-8")
            with self.assertRaises(SceneValidationError):
                import_yaml_scene(str(p))


if __name__ == "__main__":
    unittest.main()
