"""Tests for the templates catalog and insertion logic."""

from __future__ import annotations

import unittest

from cvcraft.scheduler import schedule_scene_stages
from cvcraft.templates import (
    BLOCK_DEFAULT_HYPERPARAMETERS,
    get_block_default_hyperparameters,
    get_template,
    insert_template,
    list_templates,
)
from cvcraft.validator import validate_scene
from cvcraft.constants import ANCHOR_FREE_RELEVANT_BLOCKS


def _seed_scene() -> dict:
    scene = {
        "scene": {"version": "2.0.0", "units": "voxel", "grid": {"chunkSize": 16, "worldSize": [64, 64, 64]}},
        "model": {
            "name": "seed", "family": "YOLOX", "anchorFree": True,
            "inputShape": [1, 3, 640, 640], "classCount": 80,
        },
        "blocks": [
            {"id": "in_0", "type": "InputBlock", "position": {"x": 0, "y": 0, "z": 0},
             "params": {}, "io": {"in": [], "out": ["x"]}},
            {"id": "out_0", "type": "OutputBlock", "position": {"x": 0, "y": 0, "z": 0},
             "params": {}, "io": {"in": ["x"], "out": []}},
        ],
        "edges": [{"from": "in_0", "to": "out_0", "tensor": "x"}],
        "metrics": {"flops": 0.0, "parameters": 0, "latencyMs": {}},
    }
    schedule_scene_stages(scene)
    return scene


class TestTemplatesCatalog(unittest.TestCase):
    def test_catalog_has_many_templates(self):
        catalog = list_templates()
        # New requirement: ~20 edge anchor-free templates (+ VGG-like backbones).
        self.assertGreaterEqual(len(catalog), 20)

    def test_catalog_includes_required_families(self):
        families = {t["family"] for t in list_templates()}
        for required in ("VGG", "CenterNet", "FCOS", "NanoDet", "PicoDet",
                         "YOLOX", "PP-YOLOE", "MobileNet", "RepVGG", "RTMDet"):
            self.assertIn(required, families)

    def test_catalog_entries_are_well_formed(self):
        for t in list_templates():
            self.assertIn(t["category"], {"backbone", "neck", "head"})
            self.assertIn("label", t)
            self.assertIn("description", t)
            self.assertIn("hyperparameters", t)
            self.assertIsInstance(t["blocks"], list)
            self.assertGreater(len(t["blocks"]), 0)
            for block in t["blocks"]:
                self.assertIn("type", block)
                self.assertIn("params", block)

    def test_all_template_block_types_have_defaults(self):
        for t in list_templates():
            for block in t["blocks"]:
                self.assertIn(
                    block["type"], BLOCK_DEFAULT_HYPERPARAMETERS,
                    f"{t['id']}: missing defaults for {block['type']}",
                )

    def test_all_template_block_types_are_anchor_free_relevant(self):
        for t in list_templates():
            for block in t["blocks"]:
                self.assertIn(
                    block["type"], ANCHOR_FREE_RELEVANT_BLOCKS,
                    f"{t['id']}: {block['type']} not in ANCHOR_FREE_RELEVANT_BLOCKS",
                )

    def test_get_block_default_hyperparameters_known_unknown(self):
        defaults = get_block_default_hyperparameters("Conv2dBlock")
        self.assertEqual(defaults["kernel_size"], 3)
        self.assertEqual(get_block_default_hyperparameters("__missing__"), {})

    def test_get_template_unknown_raises(self):
        with self.assertRaises(KeyError):
            get_template("nope")


class TestInsertTemplate(unittest.TestCase):
    def test_insert_backbone_keeps_scene_valid(self):
        scene = _seed_scene()
        result = insert_template(scene, "vgg16")
        self.assertGreater(len(result["inserted"]), 0)
        validate_scene(scene)

    def test_insert_head_reconnects_output_block(self):
        scene = _seed_scene()
        insert_template(scene, "centernet_head")
        # OutputBlock must now consume the head's last output, not the raw input.
        output_block = next(b for b in scene["blocks"] if b["type"] == "OutputBlock")
        self.assertNotEqual(output_block["io"]["in"], ["x"])
        validate_scene(scene)

    def test_insert_unknown_template_raises(self):
        scene = _seed_scene()
        with self.assertRaises(KeyError):
            insert_template(scene, "does_not_exist")

    def test_insert_with_explicit_anchor(self):
        scene = _seed_scene()
        result = insert_template(scene, "yolox_neck", anchor_id="in_0")
        self.assertEqual(result["anchor"], "in_0")
        validate_scene(scene)

    def test_full_anchor_free_pipeline(self):
        """Chain backbone + neck + head and ensure scene stays valid + ONNX-exportable."""
        scene = _seed_scene()
        insert_template(scene, "mobilenet_v2")
        insert_template(scene, "nanodet_neck")
        insert_template(scene, "nanodet_head")
        validate_scene(scene)

        # ONNX export should accept all block types used by these templates.
        try:
            from cvcraft.onnx_exporter import export_scene_onnx
            payload = export_scene_onnx(scene)
            self.assertIsInstance(payload, (bytes, bytearray))
            self.assertGreater(len(payload), 0)
        except RuntimeError:  # onnx not installed in this env
            self.skipTest("onnx not installed")


if __name__ == "__main__":
    unittest.main()
