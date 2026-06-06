# CVCRAFT — VoxelDet

VoxelDet is a Minecraft-like 3D voxel editor for **anchor-free** object detection models only.

Supported families:
- NanoDet
- PicoDet
- YOLOX
- FCOS
- CenterNet
- PP-YOLOE

---

## Installation

### Prerequisites
- Python 3.10 or newer
- `pip`

### Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows
```

### Install from source

```bash
git clone https://github.com/hackolite/CVCRAFT.git
cd CVCRAFT
pip install -e .
```

### Verify the installation

```bash
cvcraft --help
cvcraft-ui --help
```

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `cvcraft validate <scene>` | Validate a scene JSON file |
| `cvcraft cut <scene> <output> --ids ID [ID ...]` | Remove selected blocks with auto-repair |
| `cvcraft prune <scene> <output> --id ID --param PARAM --value N` | Reduce a parameter value |
| `cvcraft replace <scene> <output> --id ID --type TYPE` | Swap a block type |
| `cvcraft fuse <scene> <output>` | Merge Conv+BN+SiLU chains |
| `cvcraft freeze <scene> <output> --ids ID [ID ...] --value true/false` | Toggle gradient freezing |
| `cvcraft export-yaml <scene> <output>` | Export scene to YAML |
| `cvcraft export-pytorch <scene> <output> [--config-output] [--module-name]` | Export scene to PyTorch module |
| `cvcraft import-onnx <onnx> <output> [--name] [--family] [--classes] [--pretrained]` | Import ONNX model |
| `cvcraft import-yaml <yaml> <output>` | Import YAML architecture |

---

## Web UI

CVCRAFT includes a browser-based 3D editor built with Flask and Three.js.

### Launch the UI

```bash
cvcraft-ui
```

By default the server listens on `http://127.0.0.1:5000`.

Options:

| Flag | Description | Default |
|------|-------------|---------|
| `--host` | Bind address | `127.0.0.1` |
| `--port` | Port number | `5000` |
| `--debug` | Enable Flask debug/reload mode | off |

Example — expose on LAN with live reload:

```bash
cvcraft-ui --host 0.0.0.0 --port 8080 --debug
```

### UI Features

The web editor provides a Minecraft-like 3D viewport for anchor-free detection models:

- **Import**: load ONNX models (with pretrained metadata detection) or YAML architecture files.
- **3D Viewport**: blocks rendered as colored voxels per stage (backbone=blue, neck=orange, head=red). Edges shown as conduits. Frozen blocks appear translucent.
- **Selection**: click a block to select; Ctrl+click for multi-select.
- **Edit Tools**:
  - ⛏ Cut — remove selected blocks with auto-repair
  - ✂ Prune — reduce a channel parameter
  - 🔧 Replace — swap block type
  - 🧪 Fuse — merge Conv+BN+SiLU chains
  - ❄ Freeze / 🔥 Unfreeze — toggle gradient freezing
- **Export**: generate YAML or PyTorch module + config from the current scene.
- **Validate**: check scene schema and anchor-free constraints.
- **Re-layout**: deterministic 3D stage scheduling.
- **Live metrics**: FLOPs, parameters, and latency displayed in the header bar.

---

## 1) System architecture design

### A. Core subsystems
1. **Voxel UX Client (3D Editor)**
   - Block placement/removal in a chunked 3D grid
   - Selection tools: cut, prune, replace, fuse
   - Live overlays: FLOPs/params/latency, tensor shapes, graph health
2. **Model Graph Engine**
   - Internal typed DAG (operator graph)
   - Round-trip mapping: Graph ⇄ Voxel scene
   - Constraint checker for anchor-free heads only
3. **Import Service (ONNX → Voxel)**
   - ONNX parser + shape inference
   - Operator canonicalization to VoxelDet taxonomy
   - Layout planner (stage lanes, neck bridge, head tower)
4. **Edit/Repair Service**
   - Executes cut/prune/replace/fuse transforms
   - Auto graph repair (rewire, shape adapters, dead-node cleanup)
5. **Export Service (Voxel → YAML / PyTorch / ONNX-def)**
   - Emits architecture YAML for training frameworks
   - Emits PyTorch module skeleton/config
   - Emits ONNX-compatible graph definition metadata
6. **Metrics Service (real-time)**
   - FLOPs, parameters, estimated latency (backend profiles)
   - Incremental recomputation only for impacted subgraphs

### B. Processing flow
1. Import ONNX or start template (NanoDet/PicoDet/YOLOX/FCOS/CenterNet/PP-YOLOE)
2. Build canonical DAG
3. Materialize DAG as voxel blocks in 3D world
4. User edits blocks
5. Apply transform + auto-repair + validation
6. Recompute metrics in real-time
7. Export YAML / PyTorch / ONNX-compatible definitions

### C. 3D Minecraft-like UX design
- **World metaphor**: each layer/stage is a floating platform; tensor flow shown as conduits.
- **Blocks**: operator blocks have color + icon + size proportional to compute.
- **Tools**:
  - Pickaxe = delete/cut
  - Shears = prune channels
  - Wrench = replace op
  - Glue gun = fuse eligible chains
- **Modes**:
  - Build mode (free place)
  - Graph-safe mode (only valid connections)
  - Perf mode (heatmap by latency/FLOPs)
- **Live warnings**: incompatible tensor shapes, unsupported ops, anchor-based head detection.

---

## Full voxel block taxonomy

### Structural
- `InputBlock`
- `OutputBlock`
- `StageContainer`
- `SkipRoute`
- `ConcatRoute`

### Convolutional / spatial
- `Conv2dBlock`
- `DWConvBlock`
- `PWConvBlock`
- `DeformConvBlock`
- `PoolingBlock` (max/avg)
- `UpsampleBlock`
- `SPPBlock` / `SPPFBlock`

### Normalization / activation
- `BatchNormBlock`
- `GroupNormBlock`
- `SyncBNBlock`
- `ReLUBlock`
- `SiLUBlock`
- `HSwishBlock`

### Tensor ops
- `AddBlock`
- `MulBlock`
- `ConcatBlock`
- `SliceBlock`
- `ReshapeBlock`
- `TransposeBlock`
- `SigmoidBlock`

### Detection-specific (anchor-free only)
- `FPNBlock`
- `PANBlock`
- `BiFPNBlock`
- `DecoupledHeadBlock`
- `ClsHeadBlock`
- `RegHeadBlock`
- `CenterHeadBlock`
- `DFLBlock`
- `DistributionProjectBlock`
- `NMSFreeDecodeBlock`

### Utility / adaptation
- `IdentityBlock`
- `DropPathBlock`
- `QuantStubBlock`
- `DeQuantStubBlock`
- `ShapeAdapterBlock` (auto inserted by repair)

---

## 2) Voxel data model (JSON schema)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://cvcraft.io/schemas/voxeldet/v1.0.0/scene.json",
  "title": "VoxelDetScene",
  "type": "object",
  "required": ["scene", "model", "blocks", "edges", "metrics"],
  "properties": {
    "scene": {
      "type": "object",
      "required": ["version", "units", "grid"],
      "properties": {
        "version": { "type": "string" },
        "units": { "type": "string", "enum": ["voxel"] },
        "grid": {
          "type": "object",
          "required": ["chunkSize", "worldSize"],
          "properties": {
            "chunkSize": { "type": "integer", "minimum": 8 },
            "worldSize": {
              "type": "array",
              "items": { "type": "integer", "minimum": 1 },
              "minItems": 3,
              "maxItems": 3
            }
          }
        }
      }
    },
    "model": {
      "type": "object",
      "required": ["name", "family", "anchorFree", "inputShape"],
      "properties": {
        "name": { "type": "string" },
        "family": {
          "type": "string",
          "enum": ["NanoDet", "PicoDet", "YOLOX", "FCOS", "CenterNet", "PP-YOLOE"]
        },
        "anchorFree": { "type": "boolean", "const": true },
        "inputShape": {
          "type": "array",
          "items": { "type": "integer", "minimum": 1 },
          "minItems": 4,
          "maxItems": 4
        },
        "classCount": { "type": "integer", "minimum": 1 }
      }
    },
    "blocks": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["id", "type", "position", "params", "io"],
        "properties": {
          "id": { "type": "string" },
          "type": { "type": "string" },
          "position": {
            "type": "object",
            "required": ["x", "y", "z"],
            "properties": {
              "x": { "type": "integer" },
              "y": { "type": "integer" },
              "z": { "type": "integer" }
            }
          },
          "params": { "type": "object", "additionalProperties": true },
          "io": {
            "type": "object",
            "required": ["in", "out"],
            "properties": {
              "in": { "type": "array", "items": { "type": "string" } },
              "out": { "type": "array", "items": { "type": "string" } }
            }
          },
          "tags": { "type": "array", "items": { "type": "string" } }
        }
      }
    },
    "edges": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["from", "to"],
        "properties": {
          "from": { "type": "string" },
          "to": { "type": "string" },
          "tensor": { "type": "string" }
        }
      }
    },
    "metrics": {
      "type": "object",
      "required": ["flops", "parameters", "latencyMs"],
      "properties": {
        "flops": { "type": "number", "minimum": 0 },
        "parameters": { "type": "integer", "minimum": 0 },
        "latencyMs": {
          "type": "object",
          "additionalProperties": { "type": "number", "minimum": 0 }
        }
      }
    }
  }
}
```

Canonical multi-branch wiring uses **one edge object per tensor** (not tensor arrays inside one edge).
`$id` is versioned and illustrative; in deployment it should point to a hosted schema endpoint.
`anchorFree` is explicitly retained to enforce and validate the invariant at scene level.

---

## ONNX → voxel conversion pipeline

1. Parse ONNX graph + initializers.
2. Run shape/type inference.
3. Canonicalize nodes (`Conv+BN+Act` pattern labels retained for optional fuse).
4. Validate architecture family is in supported anchor-free set.
5. Map each ONNX op to taxonomy block type.
6. Place blocks in 3D lanes by stage and resolution scale.
7. Generate edges/tensors.
8. Compute initial metrics and attach scene metadata.

---

## 3) Example import of YOLOX model

Input ONNX: `yolox_s.onnx`

Voxel scene excerpt:

```json
{
  "model": { "name": "yolox_s", "family": "YOLOX", "anchorFree": true, "inputShape": [1, 3, 640, 640], "classCount": 80 },
  "blocks": [
    { "id": "b0", "type": "InputBlock", "position": { "x": 0, "y": 0, "z": 0 }, "params": {}, "io": { "in": [], "out": ["x"] } },
    { "id": "b11", "type": "StageContainer", "position": { "x": 4, "y": 0, "z": 0 }, "params": { "name": "Backbone-P3P4P5" }, "io": { "in": ["x"], "out": ["p3", "p4", "p5"] } },
    { "id": "b72", "type": "PANBlock", "position": { "x": 18, "y": 2, "z": 1 }, "params": { "levels": ["P3", "P4", "P5"] }, "io": { "in": ["p3", "p4", "p5"], "out": ["f3", "f4", "f5"] } },
    { "id": "b96", "type": "DecoupledHeadBlock", "position": { "x": 24, "y": 3, "z": 1 }, "params": { "strides": [8, 16, 32] }, "io": { "in": ["f3", "f4", "f5"], "out": ["pred"] } },
    { "id": "b120", "type": "NMSFreeDecodeBlock", "position": { "x": 28, "y": 3, "z": 1 }, "params": {}, "io": { "in": ["pred"], "out": ["detections"] } }
  ],
  "edges": [
    { "from": "b0", "to": "b11", "tensor": "x" },
    { "from": "b11", "to": "b72", "tensor": "p3" },
    { "from": "b11", "to": "b72", "tensor": "p4" },
    { "from": "b11", "to": "b72", "tensor": "p5" },
    { "from": "b72", "to": "b96", "tensor": "f3" },
    { "from": "b72", "to": "b96", "tensor": "f4" },
    { "from": "b72", "to": "b96", "tensor": "f5" },
    { "from": "b96", "to": "b120", "tensor": "pred" }
  ],
  "metrics": { "flops": 26.8e9, "parameters": 9050000, "latencyMs": { "t4_fp16": 2.9, "cpu_onnx": 24.1 } }
}
```

---

## Pruning/cutting logic with automatic graph repair

### Operations
- **Cut**: remove contiguous selected subgraph.
- **Prune**: reduce channels/branches while retaining node identity.
- **Replace**: swap block type/params (e.g., Conv2d → DWConv+PWConv).
- **Fuse**: merge eligible sequences (Conv+BN+SiLU).

### Auto-repair algorithm
1. Detect broken producer/consumer pairs.
2. Rewire direct if tensor shapes match.
3. Insert `ShapeAdapterBlock` where rank/shape mismatch is legal and resolvable.
4. Remove unreachable/dead nodes.
5. Recompute topological order.
6. Revalidate anchor-free head invariants and stride consistency.

---

## 4) Example cut/prune operation

Action:
- Cut block `b72` (`PANBlock`)
- Prune head channels in `b96` from 256 → 192

Repair result:
- Insert `ShapeAdapterBlock` blocks `b72r3`, `b72r4`, `b72r5` to map each backbone feature directly to head input scale.

Patch excerpt:

```json
{
  "removed": ["b72"],
  "updated": [
    { "id": "b96", "params": { "head_channels": 192, "strides": [8, 16, 32] } }
  ],
  "inserted": [
    { "id": "b72r3", "type": "ShapeAdapterBlock", "params": { "mode": "1x1_conv_align", "out_channels": 192 } },
    { "id": "b72r4", "type": "ShapeAdapterBlock", "params": { "mode": "1x1_conv_align", "out_channels": 192 } },
    { "id": "b72r5", "type": "ShapeAdapterBlock", "params": { "mode": "1x1_conv_align", "out_channels": 192 } }
  ],
  "rewiredEdges": [
    { "from": "b11", "to": "b72r3", "tensor": "p3" },
    { "from": "b11", "to": "b72r4", "tensor": "p4" },
    { "from": "b11", "to": "b72r5", "tensor": "p5" },
    { "from": "b72r3", "to": "b96", "tensor": "f3_aligned" },
    { "from": "b72r4", "to": "b96", "tensor": "f4_aligned" },
    { "from": "b72r5", "to": "b96", "tensor": "f5_aligned" }
  ],
  "metricsDelta": { "flops": -3.2e9, "parameters": -1100000, "latencyMs": { "t4_fp16": -0.5, "cpu_onnx": -3.3 } }
}
```

---

## Voxel → YAML export pipeline

1. Validate scene schema.
2. Convert voxel blocks to canonical DAG.
3. Run final repair/normalization pass.
4. Emit deterministic topological YAML with stage grouping (JSON camelCase fields are transformed to framework-style snake_case).
5. Attach metrics and metadata.

---

## 5) Exported YAML result

```yaml
model:
  name: yolox_s_cut_pruned
  family: YOLOX
  anchor_free: true
  input_shape: [1, 3, 640, 640]
  classes: 80

backbone:
  - {type: Conv, out_channels: 64, k: 3, s: 2}

neck:
  - {type: ShapeAdapter, mode: 1x1_conv_align, out_channels: 192, level: P3}
  - {type: ShapeAdapter, mode: 1x1_conv_align, out_channels: 192, level: P4}
  - {type: ShapeAdapter, mode: 1x1_conv_align, out_channels: 192, level: P5}

head:
  - {type: DecoupledHead, head_channels: 192, strides: [8, 16, 32]}
  - {type: NMSFreeDecode}

export:
  targets: [yaml, pytorch, onnx_compatible]

metrics:
  flops: 23.6e9
  parameters: 7950000
  latency_ms:
    t4_fp16: 2.4
    cpu_onnx: 20.8
```
