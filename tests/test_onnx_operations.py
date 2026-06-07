"""Test comprehensive ONNX operation support."""

import unittest

try:
    import onnx
    from onnx.defs import get_all_schemas_with_history
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from cvcraft.onnx_importer import _block_type_for_op


class TestONNXOperations(unittest.TestCase):
    """Test that all ONNX operations are supported."""

    @unittest.skipIf(not ONNX_AVAILABLE, "ONNX not installed")
    def test_all_onnx_ops_supported(self):
        """Verify all ONNX operations map to block types."""
        schemas = get_all_schemas_with_history()
        all_ops = set(s.name for s in schemas)
        
        unsupported = []
        for op in all_ops:
            block_type = _block_type_for_op(op)
            if block_type == "UnsupportedOpBlock":
                unsupported.append(op)
        
        self.assertEqual(
            len(unsupported),
            0,
            f"Found {len(unsupported)} unsupported operations: {unsupported}"
        )
    
    def test_common_ops_map_correctly(self):
        """Test that common operations map to expected block types."""
        expected_mappings = {
            "Conv": "Conv2dBlock",
            "BatchNormalization": "BatchNormBlock",
            "Relu": "ReLUBlock",
            "Sigmoid": "SigmoidBlock",
            "Add": "AddBlock",
            "Mul": "MulBlock",
            "Concat": "ConcatBlock",
            "Slice": "SliceBlock",
            "Reshape": "ReshapeBlock",
            "Transpose": "TransposeBlock",
            "Resize": "UpsampleBlock",
            "MaxPool": "PoolingBlock",
            "AveragePool": "PoolingBlock",
            "GlobalAveragePool": "PoolingBlock",
            "Identity": "IdentityBlock",
            "MatMul": "MulBlock",
            "Gemm": "MulBlock",
            "LeakyRelu": "ReLUBlock",
            "Softmax": "SigmoidBlock",
            "Dropout": "IdentityBlock",
            "Flatten": "ReshapeBlock",
            "Pad": "IdentityBlock",
        }
        
        for op_type, expected_block in expected_mappings.items():
            with self.subTest(op_type=op_type):
                actual_block = _block_type_for_op(op_type)
                self.assertEqual(
                    actual_block,
                    expected_block,
                    f"Operation {op_type} should map to {expected_block}, got {actual_block}"
                )
    
    def test_activation_functions(self):
        """Test that activation functions are mapped."""
        activation_ops = [
            "Relu", "LeakyRelu", "PRelu", "ThresholdedRelu",
            "Elu", "Selu", "Sigmoid", "Tanh", "Softmax",
            "HardSwish", "Mish", "Gelu"
        ]
        
        for op in activation_ops:
            with self.subTest(op=op):
                block_type = _block_type_for_op(op)
                self.assertNotEqual(
                    block_type,
                    "UnsupportedOpBlock",
                    f"Activation {op} should be supported"
                )
    
    def test_pooling_operations(self):
        """Test that pooling operations are mapped correctly."""
        pooling_ops = [
            "MaxPool", "AveragePool", "GlobalAveragePool",
            "GlobalMaxPool", "LpPool"
        ]
        
        for op in pooling_ops:
            with self.subTest(op=op):
                block_type = _block_type_for_op(op)
                self.assertEqual(
                    block_type,
                    "PoolingBlock",
                    f"Pooling operation {op} should map to PoolingBlock"
                )
    
    def test_normalization_operations(self):
        """Test that normalization operations are mapped correctly."""
        norm_ops = [
            "BatchNormalization", "GroupNormalization",
            "InstanceNormalization", "LayerNormalization"
        ]
        
        for op in norm_ops:
            with self.subTest(op=op):
                block_type = _block_type_for_op(op)
                self.assertIn(
                    block_type,
                    ["BatchNormBlock", "GroupNormBlock"],
                    f"Normalization operation {op} should map to a normalization block"
                )
    
    def test_tensor_shape_operations(self):
        """Test that tensor shape operations are mapped."""
        shape_ops = [
            "Reshape", "Flatten", "Squeeze", "Unsqueeze",
            "Transpose", "Tile", "Expand"
        ]
        
        for op in shape_ops:
            with self.subTest(op=op):
                block_type = _block_type_for_op(op)
                self.assertNotEqual(
                    block_type,
                    "UnsupportedOpBlock",
                    f"Shape operation {op} should be supported"
                )
    
    def test_arithmetic_operations(self):
        """Test that arithmetic operations are mapped."""
        arithmetic_ops = [
            "Add", "Sub", "Mul", "Div", "Pow", "Sqrt",
            "Exp", "Log", "Abs", "Neg"
        ]
        
        for op in arithmetic_ops:
            with self.subTest(op=op):
                block_type = _block_type_for_op(op)
                self.assertNotEqual(
                    block_type,
                    "UnsupportedOpBlock",
                    f"Arithmetic operation {op} should be supported"
                )
    
    def test_reduction_operations(self):
        """Test that reduction operations are mapped."""
        reduction_ops = [
            "ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin",
            "ReduceProd", "ReduceL1", "ReduceL2"
        ]
        
        for op in reduction_ops:
            with self.subTest(op=op):
                block_type = _block_type_for_op(op)
                self.assertNotEqual(
                    block_type,
                    "UnsupportedOpBlock",
                    f"Reduction operation {op} should be supported"
                )
    
    @unittest.skipIf(not ONNX_AVAILABLE, "ONNX not installed")
    def test_operation_count(self):
        """Test that we support all 223 ONNX operations."""
        schemas = get_all_schemas_with_history()
        all_ops = set(s.name for s in schemas)
        
        # Count supported operations
        supported_count = sum(
            1 for op in all_ops
            if _block_type_for_op(op) != "UnsupportedOpBlock"
        )
        
        self.assertEqual(
            supported_count,
            len(all_ops),
            f"Expected all {len(all_ops)} operations to be supported, got {supported_count}"
        )
        
        # Verify it's 223 operations as expected
        self.assertEqual(
            len(all_ops),
            223,
            f"Expected 223 ONNX operations, found {len(all_ops)}"
        )


class TestONNXExporter(unittest.TestCase):
    """Tests for the ONNX export pipeline."""

    def _make_minimal_scene(self, frozen_conv=False):
        blocks = [
            {
                "id": "inp",
                "type": "InputBlock",
                "position": {"x": 0, "y": 0, "z": 0},
                "params": {},
                "io": {"in": [], "out": ["x"]},
            },
            {
                "id": "conv",
                "type": "Conv2dBlock",
                "position": {"x": 1, "y": 0, "z": 0},
                "params": {"in_channels": 3, "out_channels": 16, "kernel_size": 3, "has_weights": True},
                "io": {"in": ["x"], "out": ["y"]},
            },
            {
                "id": "out",
                "type": "OutputBlock",
                "position": {"x": 2, "y": 0, "z": 0},
                "params": {},
                "io": {"in": ["y"], "out": []},
            },
        ]
        if frozen_conv:
            blocks[1]["meta"] = {"frozen": True}
        return {
            "model": {
                "name": "test_model",
                "family": "YOLOX",
                "anchorFree": True,
                "inputShape": [1, 3, 64, 64],
                "classCount": 80,
            },
            "blocks": blocks,
            "edges": [
                {"from": "inp", "to": "conv", "tensor": "x"},
                {"from": "conv", "to": "out", "tensor": "y"},
            ],
            "metrics": {"flops": 0.0, "parameters": 0, "latencyMs": {}},
        }

    @unittest.skipIf(not ONNX_AVAILABLE, "ONNX not installed")
    def test_export_returns_bytes(self):
        from cvcraft.onnx_exporter import export_scene_onnx
        import copy
        data = export_scene_onnx(copy.deepcopy(self._make_minimal_scene()))
        self.assertIsInstance(data, bytes)
        self.assertGreater(len(data), 0)

    @unittest.skipIf(not ONNX_AVAILABLE, "ONNX not installed")
    def test_export_include_weights(self):
        from cvcraft.onnx_exporter import export_scene_onnx
        import copy
        data = export_scene_onnx(copy.deepcopy(self._make_minimal_scene()), include_weights=True)
        model = onnx.load_from_string(data)
        init_names = [i.name for i in model.graph.initializer]
        self.assertTrue(any("conv" in n for n in init_names), f"Expected conv weight in {init_names}")

    @unittest.skipIf(not ONNX_AVAILABLE, "ONNX not installed")
    def test_export_no_weights(self):
        from cvcraft.onnx_exporter import export_scene_onnx
        import copy
        data = export_scene_onnx(copy.deepcopy(self._make_minimal_scene()), include_weights=False)
        model = onnx.load_from_string(data)
        self.assertEqual(len(model.graph.initializer), 0)

    @unittest.skipIf(not ONNX_AVAILABLE, "ONNX not installed")
    def test_frozen_blocks_tagged_in_initializers(self):
        from cvcraft.onnx_exporter import export_scene_onnx
        import copy
        data = export_scene_onnx(copy.deepcopy(self._make_minimal_scene(frozen_conv=True)), include_weights=True)
        model = onnx.load_from_string(data)
        init_names = [i.name for i in model.graph.initializer]
        self.assertTrue(
            any(n.startswith("frozen.") for n in init_names),
            f"Expected frozen. prefix in {init_names}",
        )

    @unittest.skipIf(not ONNX_AVAILABLE, "ONNX not installed")
    def test_frozen_metadata_in_model(self):
        from cvcraft.onnx_exporter import export_scene_onnx
        import copy
        data = export_scene_onnx(copy.deepcopy(self._make_minimal_scene(frozen_conv=True)), include_weights=True)
        model = onnx.load_from_string(data)
        meta = {p.key: p.value for p in model.metadata_props}
        self.assertIn("cvcraft.frozen_blocks", meta)
        self.assertIn("conv", meta["cvcraft.frozen_blocks"])

    @unittest.skipIf(not ONNX_AVAILABLE, "ONNX not installed")
    def test_check_onnx_format_valid(self):
        from cvcraft.onnx_exporter import check_onnx_format
        import copy
        result = check_onnx_format(copy.deepcopy(self._make_minimal_scene()))
        self.assertTrue(result["valid"])
        self.assertEqual(result["unsupported_ops"], [])


class TestONNXAttrMapping(unittest.TestCase):
    """Tests for the ONNX attribute-to-CVCRAFT name mapping."""

    def test_conv_attrs_mapped(self):
        from cvcraft.onnx_importer import _map_onnx_attrs_to_cvcraft
        attrs = {
            "kernel_shape": [3, 3],
            "strides": [1, 1],
            "dilations": [1, 1],
            "group": 1,
            "pads": [1, 1, 1, 1],
        }
        result = _map_onnx_attrs_to_cvcraft(attrs, "Conv")
        self.assertEqual(result["kernel_size"], 3)
        self.assertEqual(result["stride"], 1)
        self.assertEqual(result["dilation"], 1)
        self.assertEqual(result["groups"], 1)
        self.assertEqual(result["padding"], 1)

    def test_asymmetric_pads_kept_as_list(self):
        from cvcraft.onnx_importer import _map_onnx_attrs_to_cvcraft
        attrs = {"kernel_shape": [3, 5], "pads": [0, 1, 0, 2]}
        result = _map_onnx_attrs_to_cvcraft(attrs, "Conv")
        self.assertIsInstance(result["padding"], list)

    def test_batchnorm_epsilon_aliased(self):
        from cvcraft.onnx_importer import _map_onnx_attrs_to_cvcraft
        attrs = {"epsilon": 1e-5, "momentum": 0.1}
        result = _map_onnx_attrs_to_cvcraft(attrs, "BatchNormalization")
        self.assertAlmostEqual(result["eps"], 1e-5)

    def test_existing_keys_not_overwritten(self):
        from cvcraft.onnx_importer import _map_onnx_attrs_to_cvcraft
        attrs = {"kernel_shape": [5, 5], "kernel_size": 3}
        result = _map_onnx_attrs_to_cvcraft(attrs, "Conv")
        # Should NOT overwrite already-present kernel_size
        self.assertEqual(result["kernel_size"], 3)
        # Original ONNX key should still be present
        self.assertIn("kernel_shape", result)

    def test_original_attrs_preserved(self):
        from cvcraft.onnx_importer import _map_onnx_attrs_to_cvcraft
        attrs = {"kernel_shape": [3, 3], "strides": [2, 2]}
        result = _map_onnx_attrs_to_cvcraft(attrs, "Conv")
        self.assertIn("kernel_shape", result)
        self.assertIn("strides", result)


if __name__ == "__main__":
    unittest.main()
