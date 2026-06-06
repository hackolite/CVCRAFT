"""Test script to verify anchor-free operation filtering."""

import json
from cvcraft.constants import ANCHOR_FREE_RELEVANT_BLOCKS
from cvcraft.scene_v2 import normalize_scene_v2

# Create a test scene with both relevant and non-relevant blocks
test_scene = {
    "scene": {
        "version": "2.0.0",
        "units": "voxel",
        "grid": {"chunkSize": 16, "worldSize": [100, 100, 100]},
    },
    "model": {
        "name": "test_model",
        "family": "YOLOX",
        "anchorFree": True,
        "inputShape": [1, 3, 640, 640],
        "classCount": 80,
    },
    "blocks": [
        {"id": "input", "type": "InputBlock", "position": {"x": 0, "y": 0, "z": 0}, "params": {}, "io": {"in": [], "out": ["x"]}},
        {"id": "conv1", "type": "Conv2dBlock", "position": {"x": 2, "y": 0, "z": 0}, "params": {}, "io": {"in": ["x"], "out": ["c1"]}},
        {"id": "lstm1", "type": "LSTMBlock", "position": {"x": 4, "y": 0, "z": 0}, "params": {}, "io": {"in": ["c1"], "out": ["l1"]}},
        {"id": "relu1", "type": "ReLUBlock", "position": {"x": 6, "y": 0, "z": 0}, "params": {}, "io": {"in": ["l1"], "out": ["r1"]}},
        {"id": "gru1", "type": "GRUBlock", "position": {"x": 8, "y": 0, "z": 0}, "params": {}, "io": {"in": ["r1"], "out": ["g1"]}},
        {"id": "fpn1", "type": "FPNBlock", "position": {"x": 10, "y": 0, "z": 0}, "params": {}, "io": {"in": ["g1"], "out": ["f1"]}},
        {"id": "output", "type": "OutputBlock", "position": {"x": 12, "y": 0, "z": 0}, "params": {}, "io": {"in": ["f1"], "out": []}},
    ],
    "edges": [
        {"from": "input", "to": "conv1", "tensor": "x"},
        {"from": "conv1", "to": "lstm1", "tensor": "c1"},
        {"from": "lstm1", "to": "relu1", "tensor": "l1"},
        {"from": "relu1", "to": "gru1", "tensor": "r1"},
        {"from": "gru1", "to": "fpn1", "tensor": "g1"},
        {"from": "fpn1", "to": "output", "tensor": "f1"},
    ],
    "metrics": {"flops": 1000000, "parameters": 10000, "latencyMs": {}},
}

# Normalize the scene to add metadata
normalize_scene_v2(test_scene)

# Check which blocks are marked as relevant/non-relevant
print("=" * 60)
print("Anchor-Free Operation Filtering Test")
print("=" * 60)
print()

relevant_count = 0
non_relevant_count = 0

for block in test_scene["blocks"]:
    block_id = block["id"]
    block_type = block["type"]
    is_relevant = block.get("meta", {}).get("anchor_free_relevant", False)
    
    status = "✓ RELEVANT" if is_relevant else "⚠ NON-RELEVANT"
    
    print(f"{block_id:12} [{block_type:20}] → {status}")
    
    if is_relevant:
        relevant_count += 1
    else:
        non_relevant_count += 1

print()
print("=" * 60)
print(f"Total blocks: {len(test_scene['blocks'])}")
print(f"Relevant blocks: {relevant_count}")
print(f"Non-relevant blocks: {non_relevant_count}")
print("=" * 60)
print()

# Verify expected behavior
expected_relevant = {"InputBlock", "Conv2dBlock", "ReLUBlock", "FPNBlock", "OutputBlock"}
expected_non_relevant = {"LSTMBlock", "GRUBlock"}

print("Verification:")
for block in test_scene["blocks"]:
    block_type = block["type"]
    is_relevant = block.get("meta", {}).get("anchor_free_relevant", False)
    
    should_be_relevant = block_type in expected_relevant
    should_be_non_relevant = block_type in expected_non_relevant
    
    if should_be_relevant and not is_relevant:
        print(f"❌ ERROR: {block_type} should be marked as RELEVANT but is NOT")
    elif should_be_non_relevant and is_relevant:
        print(f"❌ ERROR: {block_type} should be marked as NON-RELEVANT but IS")
    else:
        print(f"✓ {block_type} correctly marked")

print()
print("=" * 60)
print("Total anchor-free relevant block types defined:", len(ANCHOR_FREE_RELEVANT_BLOCKS))
print("=" * 60)
print()
print("Test completed!")
