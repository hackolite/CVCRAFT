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


if __name__ == "__main__":
    unittest.main()
