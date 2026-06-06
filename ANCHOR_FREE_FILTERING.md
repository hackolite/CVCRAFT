# Anchor-Free Operation Filtering

CVCRAFT is specifically designed for **anchor-free object detection models**. To help users identify operations that are not typically used in anchor-free architectures, the system now includes visual filtering.

## Visual Indicators

### In the 3D Viewport

Non-relevant operations are visually distinguished in the 3D editor:

- **Opacity**: Non-relevant blocks are rendered with 30% opacity (very translucent)
- **Color**: Non-relevant blocks have a desaturated (grayish) color
- **Legend**: A "⚠ Non-relevant" indicator appears in the stage legend

### In Block Information

When selecting a block, the details panel shows:

- **Relevance status**: `✓ Relevant` or `⚠ Non-relevant`
- **Warning message**: For non-relevant blocks, a note explains that the operation is not typically used in anchor-free object detection models

### In Tooltips

Hovering over blocks shows:
- A `⚠` warning icon for non-relevant operations

## What Operations are Considered Relevant?

Based on the IR specification for edge anchor-free object detection, the following operation categories are considered relevant:

### Structural
- InputBlock, OutputBlock, StageContainer, SkipRoute, ConcatRoute

### Backbone Operators
- **Convolutions**: Conv2dBlock, Conv2d, ConvBlock, DWConvBlock, PWConvBlock, DepthwiseConv, MBConv, GhostBlock, ResidualBlock
- **Pooling**: PoolingBlock, MaxPool2d, AvgPool2d, UpsampleBlock, Upsample
- **Spatial**: SPPBlock, SPPFBlock, CSPBlock, FocusBlock

### Normalization & Activation
- **Normalization**: BatchNormBlock, BatchNorm2d, GroupNormBlock, SyncBNBlock
- **Activation**: ReLUBlock, ReLU, SiLUBlock, SiLU, HSwishBlock

### Tensor Operations
- AddBlock, Add, MulBlock, ConcatBlock, Concat, SliceBlock
- ReshapeBlock, Reshape, TransposeBlock, Transpose
- SigmoidBlock, Sigmoid, Flatten, Permute

### Neck Operators (Multi-scale Fusion)
- FPNBlock, PANBlock, BiFPNBlock, ShapeAdapterBlock, ResizeFeatureMap, Downsample

### Head Operators (Anchor-Free Detection Only)
- DecoupledHeadBlock, ClsHeadBlock, RegHeadBlock, CenterHeadBlock
- ClassificationHead, RegressionHead, CenterHeatmapHead, ObjectnessHead
- DFLBlock, DistributionProjectBlock, NMSFreeDecodeBlock, DetectHead

### Graph & Utility Operators
- Split, Merge, Route, IdentityBlock, Identity
- DropPathBlock, Dropout, QuantStubBlock, DeQuantStubBlock

## What Operations are Considered Non-Relevant?

Operations that are **NOT** typically used in anchor-free object detection architectures include:

### Recurrent Layers
- **LSTM, GRU, RNN**: Not used in anchor-free detection (designed for sequence modeling)

### Control Flow
- **If, Loop, Scan**: Not typical in feed-forward detection models

### Signal Processing
- **DFT, STFT, FFT operations**: Not used in standard object detection

### ML-Specific Operators
- **Sklearn-like operators**: Classifiers, regressors, vectorizers, encoders

### Attention/Transformer Operations
- **Attention, RotaryEmbedding**: Unless explicitly used in a specific anchor-free architecture variant (not in the core set)

### String & Loss Operations
- **String operations, loss functions**: Typically not part of inference models

### Optimization Operations
- **Gradient, Adam, Momentum**: Training-time only, not in inference graphs

## Technical Implementation

The filtering system works through:

1. **Constants Definition** (`cvcraft/constants.py`):
   - `ANCHOR_FREE_RELEVANT_BLOCKS` set defines all relevant block types

2. **Scene Normalization** (`cvcraft/scene_v2.py`):
   - `normalize_scene_v2()` function adds `anchor_free_relevant` metadata to each block

3. **Frontend Rendering** (`cvcraft/ui/static/app.js`):
   - `renderScene()` function applies visual styling based on relevance
   - Non-relevant blocks are desaturated and rendered with low opacity

4. **User Interface** (`cvcraft/ui/templates/index.html`):
   - Legend includes non-relevant indicator
   - Tooltips and selection panels show relevance status

## Use Cases

This feature helps users:

1. **Import ONNX models** and quickly identify operations that may not be optimized for anchor-free detection
2. **Debug architectures** by spotting unexpected operations (e.g., recurrent layers in a CNN-based detector)
3. **Understand compatibility** when converting models from other frameworks
4. **Optimize for edge deployment** by focusing on relevant operations

## Testing

Run the test script to verify the filtering:

```bash
python test_anchor_free_filtering.py
```

This will create a test scene with both relevant and non-relevant operations and verify that they are correctly classified.

## Supported Anchor-Free Families

- NanoDet
- PicoDet
- YOLOX
- FCOS
- CenterNet
- PP-YOLOE

All operations in imported models from these families will be analyzed and marked for relevance to anchor-free architectures.
