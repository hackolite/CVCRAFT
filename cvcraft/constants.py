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
