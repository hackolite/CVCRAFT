# ONNX Operation Support

CVCRAFT now supports **all 223 ONNX operations** for model import and conversion to voxel scenes.

## Overview

When importing ONNX models using `cvcraft import-onnx`, all operations are automatically mapped to appropriate CVCRAFT block types. Operations that don't have a direct equivalent are mapped to generic block types that preserve the computational graph structure.

## Operation Mapping Categories

### Convolutional Operations (5 ops)
- `Conv`, `ConvTranspose`, `ConvInteger`, `QLinearConv` → `Conv2dBlock`
- `DeformConv` → `DeformConvBlock`

### Normalization Operations (7 ops)
- `BatchNormalization`, `InstanceNormalization`, `LayerNormalization`, `LRN`, `MeanVarianceNormalization`, `RMSNormalization` → `BatchNormBlock`
- `GroupNormalization` → `GroupNormBlock`

### Activation Functions (16 ops)
- `Relu`, `LeakyRelu`, `PRelu`, `ThresholdedRelu` → `ReLUBlock`
- `Elu`, `Celu`, `Selu`, `Swish`, `Mish`, `Gelu` → `SiLUBlock`
- `Sigmoid`, `HardSigmoid`, `Tanh`, `Softmax`, `LogSoftmax`, `Softplus`, `Softsign`, `Hardmax` → `SigmoidBlock`
- `HardSwish` → `HSwishBlock`

### Pooling Operations (9 ops)
All map to `PoolingBlock`:
- `MaxPool`, `AveragePool`, `GlobalAveragePool`, `GlobalMaxPool`
- `LpPool`, `GlobalLpPool`, `MaxUnpool`, `MaxRoiPool`, `RoiAlign`

### Arithmetic Operations (16 ops)
- Addition-like: `Add`, `Sub`, `Sum`, `Mean`, `Max`, `Min`, `CumSum` → `AddBlock`
- Multiplication-like: `Mul`, `Div`, `Neg`, `Abs`, `Reciprocal`, `Pow`, `Sqrt`, `Exp`, `Log`, `Mod`, `CumProd` → `MulBlock`

### Tensor Operations (17 ops)
- `Concat`, `ConcatFromSequence`, `SequenceConstruct`, `StringConcat` → `ConcatBlock`
- `Split`, `Slice`, `Gather`, `GatherElements`, `GatherND`, `Scatter`, `ScatterElements`, `ScatterND`, `TensorScatter`, `Compress`, `SequenceAt`, `SequenceErase`, `TopK`, `StringSplit` → `SliceBlock`
- `Reshape`, `Flatten`, `Squeeze`, `Unsqueeze`, `Tile`, `Expand`, `Col2Im` → `ReshapeBlock`
- `Transpose` → `TransposeBlock`

### Resize/Upsample Operations (4 ops)
- `Resize`, `Upsample` → `UpsampleBlock`
- `DepthToSpace`, `SpaceToDepth` → `UpsampleBlock`
- `GridSample` → `UpsampleBlock`

### Comparison Operations (5 ops)
All map to `AddBlock`:
- `Equal`, `Greater`, `GreaterOrEqual`, `Less`, `LessOrEqual`

### Logical Operations (8 ops)
- `And`, `BitwiseAnd` → `MulBlock`
- `Or`, `Xor`, `BitwiseOr`, `BitwiseXor` → `AddBlock`
- `Not`, `BitwiseNot` → `IdentityBlock`

### Reduction Operations (10 ops)
- `ReduceSum`, `ReduceMean`, `ReduceMax`, `ReduceMin`, `ReduceL1`, `ReduceL2`, `ReduceLogSum`, `ReduceLogSumExp`, `ReduceSumSquare` → `AddBlock`
- `ReduceProd` → `MulBlock`

### Mathematical Functions (20 ops)
Trigonometric and transcendental functions map to `SigmoidBlock`:
- `Sin`, `Cos`, `Tan`, `Asin`, `Acos`, `Atan`
- `Sinh`, `Cosh`, `Asinh`, `Acosh`, `Atanh`
- `Erf`

Rounding functions map to `IdentityBlock`:
- `Sign`, `Ceil`, `Floor`, `Round`, `Clip`, `Shrink`

### Matrix Operations (5 ops)
All map to `MulBlock`:
- `MatMul`, `MatMulInteger`, `QLinearMatMul`, `Gemm`, `Einsum`

### Recurrent/Sequence Operations (13 ops)
Most map to `IdentityBlock`:
- `LSTM`, `GRU`, `RNN`, `SequenceEmpty`, `SequenceLength`, `SequenceMap`

Some map to their tensor operation equivalents:
- `SequenceInsert` → `ConcatBlock`
- `SplitToSequence` → `SliceBlock`

### Control Flow (3 ops)
All map to `IdentityBlock`:
- `If`, `Loop`, `Scan`

### Shape Operations (5 ops)
All map to `IdentityBlock`:
- `Shape`, `Size`, `ConstantOfShape`, `EyeLike`, `Range`

### Data Generation (7 ops)
All map to `IdentityBlock`:
- `Constant`, `RandomNormal`, `RandomNormalLike`, `RandomUniform`, `RandomUniformLike`, `Multinomial`, `Bernoulli`

### Quantization (3 ops)
All map to `IdentityBlock`:
- `QuantizeLinear`, `DequantizeLinear`, `DynamicQuantizeLinear`

### Type Conversion (3 ops)
All map to `IdentityBlock`:
- `Cast`, `CastLike`, `BitCast`

### Signal Processing (6 ops)
All map to `IdentityBlock`:
- `DFT`, `STFT`, `MelWeightMatrix`, `BlackmanWindow`, `HammingWindow`, `HannWindow`

### Attention and Transformer (2 ops)
All map to `IdentityBlock`:
- `Attention`, `RotaryEmbedding`

### Grid/Spatial Operations (2 ops)
- `AffineGrid` → `ReshapeBlock`
- `GridSample` → `UpsampleBlock`

### Loss Functions (2 ops)
All map to `IdentityBlock`:
- `NegativeLogLikelihoodLoss`, `SoftmaxCrossEntropyLoss`

### Object Detection (2 ops)
All map to `IdentityBlock`:
- `NonMaxSuppression`, `CenterCropPad`

### String Operations (4 ops)
- `StringConcat` → `ConcatBlock`
- `StringSplit` → `SliceBlock`
- `StringNormalizer`, `RegexFullMatch` → `IdentityBlock`

### Optimization Operations (4 ops)
All map to `IdentityBlock`:
- `Adagrad`, `Adam`, `Momentum`, `Gradient`

### ML-Specific Operators (20 ops)
Sklearn-like operators:
- `ArrayFeatureExtractor` → `SliceBlock`
- `Normalizer`, `Scaler` → `BatchNormBlock`
- All classifiers, regressors, encoders, vectorizers → `IdentityBlock`

### Miscellaneous (10+ ops)
- `Identity`, `Dropout`, `Pad`, `Where`, `OneHot`, `NonZero`, `IsNaN`, `IsInf`, `Unique`, `ArgMax`, `ArgMin`, `Det`, `Trilu`, `ReverseSequence` → `IdentityBlock`
- `Optional`, `OptionalGetElement`, `OptionalHasElement` → `IdentityBlock`
- `BitShift` → `MulBlock`
- `ImageDecoder` → `IdentityBlock`
- `LpNormalization` → `BatchNormBlock`

## Testing

All 223 ONNX operations are covered by comprehensive tests. Run:

```bash
python -m unittest tests.test_onnx_operations -v
```

## Usage Example

```bash
# Import any ONNX model with full operation support
cvcraft import-onnx model.onnx output.json --name my_model --family YOLOX

# Use diagnostic script to verify no unsupported blocks
python diagnostic_unsupported.py output.json
```

## Notes

- Operations are mapped to the most semantically appropriate CVCRAFT block type
- Generic operations use `IdentityBlock` to preserve graph structure while maintaining flow
- All mappings preserve the computational graph topology for proper scene generation
- The mapping is optimized for anchor-free object detection models (YOLOX, NanoDet, PicoDet, FCOS, CenterNet, PP-YOLOE)
