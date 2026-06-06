SUPPORTED_FAMILIES = {"NanoDet", "PicoDet", "YOLOX", "FCOS", "CenterNet", "PP-YOLOE"}

DETECTION_BLOCKS = {
    "FPNBlock",
    "PANBlock",
    "BiFPNBlock",
    "DecoupledHeadBlock",
    "ClsHeadBlock",
    "RegHeadBlock",
    "CenterHeadBlock",
    "DFLBlock",
    "DistributionProjectBlock",
    "NMSFreeDecodeBlock",
}

NECK_BLOCKS = {"FPNBlock", "PANBlock", "BiFPNBlock", "ShapeAdapterBlock"}

STAGE_ORDER = ("input", "backbone", "neck", "head", "output")

STAGE_COLORS = {
    "input": "#9CA3AF",
    "backbone": "#2563EB",
    "neck": "#D97706",
    "head": "#DC2626",
    "output": "#059669",
}

# Anchor-free relevant block types for edge object detection
# Based on the IR specification for anchor-free models
ANCHOR_FREE_RELEVANT_BLOCKS = {
    # Structural
    "InputBlock",
    "OutputBlock",
    "StageContainer",
    "SkipRoute",
    "ConcatRoute",
    
    # Backbone operators (convolutional / spatial)
    "Conv2dBlock",
    "Conv2d",
    "ConvBlock",
    "DWConvBlock",
    "PWConvBlock",
    "DepthwiseConv",
    "MBConv",
    "GhostBlock",
    "ResidualBlock",
    "PoolingBlock",
    "MaxPool2d",
    "AvgPool2d",
    "UpsampleBlock",
    "Upsample",
    "SPPBlock",
    "SPPFBlock",
    "CSPBlock",
    "FocusBlock",
    
    # Normalization / activation
    "BatchNormBlock",
    "BatchNorm2d",
    "GroupNormBlock",
    "SyncBNBlock",
    "ReLUBlock",
    "ReLU",
    "SiLUBlock",
    "SiLU",
    "HSwishBlock",
    
    # Tensor ops
    "AddBlock",
    "Add",
    "MulBlock",
    "ConcatBlock",
    "Concat",
    "SliceBlock",
    "ReshapeBlock",
    "Reshape",
    "TransposeBlock",
    "Transpose",
    "SigmoidBlock",
    "Sigmoid",
    "Flatten",
    "Permute",
    
    # Neck operators (detection-specific)
    "FPNBlock",
    "PANBlock",
    "BiFPNBlock",
    "ShapeAdapterBlock",
    "ResizeFeatureMap",
    "Downsample",
    
    # Head operators (anchor-free only)
    "DecoupledHeadBlock",
    "ClsHeadBlock",
    "RegHeadBlock",
    "CenterHeadBlock",
    "ClassificationHead",
    "RegressionHead",
    "CenterHeatmapHead",
    "ObjectnessHead",
    "DFLBlock",
    "DistributionProjectBlock",
    "NMSFreeDecodeBlock",
    "DetectHead",
    
    # Graph operators
    "Split",
    "Merge",
    "Route",
    "IdentityBlock",
    "Identity",
    
    # Utility / adaptation
    "DropPathBlock",
    "Dropout",
    "QuantStubBlock",
    "DeQuantStubBlock",
}
