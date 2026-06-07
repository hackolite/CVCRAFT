"""CVCRAFT CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .editor import cut_blocks, fuse_conv_bn_silu, prune_block, replace_block, set_blocks_frozen
from .exporter import export_scene_yaml
from .onnx_exporter import export_scene_onnx
from .onnx_importer import import_onnx_scene
from .optimization import (
    dequantize_scene,
    freeze_component,
    quantize_scene,
    set_component_lr_scale,
    set_edge_constraints,
    set_lr_scale,
    set_pretrained_config,
)
from .pytorch_exporter import export_scene_pytorch
from .templates import insert_template, list_templates
from .yaml_importer import import_yaml_scene
from .validator import SceneValidationError, validate_scene


def _read_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str, content: dict) -> None:
    Path(path).write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(prog="cvcraft")
    sub = parser.add_subparsers(dest="cmd", required=True)

    validate_cmd = sub.add_parser("validate")
    validate_cmd.add_argument("scene")

    cut_cmd = sub.add_parser("cut")
    cut_cmd.add_argument("scene")
    cut_cmd.add_argument("output")
    cut_cmd.add_argument("--ids", nargs="+", required=True)

    prune_cmd = sub.add_parser("prune")
    prune_cmd.add_argument("scene")
    prune_cmd.add_argument("output")
    prune_cmd.add_argument("--id", required=True)
    prune_cmd.add_argument("--param", required=True)
    prune_cmd.add_argument("--value", type=int, required=True)

    replace_cmd = sub.add_parser("replace")
    replace_cmd.add_argument("scene")
    replace_cmd.add_argument("output")
    replace_cmd.add_argument("--id", required=True)
    replace_cmd.add_argument("--type", required=True)

    fuse_cmd = sub.add_parser("fuse")
    fuse_cmd.add_argument("scene")
    fuse_cmd.add_argument("output")

    export_cmd = sub.add_parser("export-yaml")
    export_cmd.add_argument("scene")
    export_cmd.add_argument("output")

    export_pt_cmd = sub.add_parser("export-pytorch")
    export_pt_cmd.add_argument("scene")
    export_pt_cmd.add_argument("output")
    export_pt_cmd.add_argument("--config-output")
    export_pt_cmd.add_argument("--module-name", default="GeneratedVoxelModel")

    import_cmd = sub.add_parser("import-onnx")
    import_cmd.add_argument("onnx")
    import_cmd.add_argument("output")
    import_cmd.add_argument("--name")
    import_cmd.add_argument("--family", default="YOLOX")
    import_cmd.add_argument("--classes", type=int, default=80)
    import_cmd.add_argument("--pretrained", choices=("auto", "yes", "no"), default="auto")

    import_yaml_cmd = sub.add_parser("import-yaml")
    import_yaml_cmd.add_argument("yaml")
    import_yaml_cmd.add_argument("output")

    freeze_cmd = sub.add_parser("freeze")
    freeze_cmd.add_argument("scene")
    freeze_cmd.add_argument("output")
    freeze_cmd.add_argument("--ids", nargs="+", required=True)
    freeze_cmd.add_argument("--value", choices=("true", "false"), required=True)

    # Export ONNX
    export_onnx_cmd = sub.add_parser("export-onnx")
    export_onnx_cmd.add_argument("scene")
    export_onnx_cmd.add_argument("output")
    export_onnx_cmd.add_argument("--opset", type=int, default=13)
    export_onnx_cmd.add_argument(
        "--no-weights", action="store_true",
        help="Omit weight initializers from the exported ONNX (structure-only export)"
    )

    # Quantization
    quantize_cmd = sub.add_parser("quantize")
    quantize_cmd.add_argument("scene")
    quantize_cmd.add_argument("output")
    quantize_cmd.add_argument("--mode", choices=("int8", "fp16"), required=True)

    dequantize_cmd = sub.add_parser("dequantize")
    dequantize_cmd.add_argument("scene")
    dequantize_cmd.add_argument("output")

    # LR scaling
    lr_scale_cmd = sub.add_parser("lr-scale")
    lr_scale_cmd.add_argument("scene")
    lr_scale_cmd.add_argument("output")
    lr_scale_cmd.add_argument("--ids", nargs="+")
    lr_scale_cmd.add_argument("--component", choices=("backbone", "neck", "head"))
    lr_scale_cmd.add_argument("--value", type=float, required=True)

    # Component freeze
    freeze_component_cmd = sub.add_parser("freeze-component")
    freeze_component_cmd.add_argument("scene")
    freeze_component_cmd.add_argument("output")
    freeze_component_cmd.add_argument("--component", choices=("backbone", "neck", "head"), required=True)
    freeze_component_cmd.add_argument("--value", choices=("true", "false"), required=True)

    # Edge constraints
    edge_cmd = sub.add_parser("edge-constraints")
    edge_cmd.add_argument("scene")
    edge_cmd.add_argument("output")
    edge_cmd.add_argument("--max-size-mb", type=float)
    edge_cmd.add_argument("--latency-target-ms", type=float)

    # Pretrained config
    pretrained_cmd = sub.add_parser("pretrained-config")
    pretrained_cmd.add_argument("scene")
    pretrained_cmd.add_argument("output")
    pretrained_cmd.add_argument("--source")
    pretrained_cmd.add_argument("--inherit", choices=("true", "false"), default="true")
    pretrained_cmd.add_argument("--strict", choices=("true", "false"), default="false")

    # Templates
    insert_tpl_cmd = sub.add_parser("insert-template")
    insert_tpl_cmd.add_argument("scene")
    insert_tpl_cmd.add_argument("output")
    insert_tpl_cmd.add_argument("--template", required=True, help="Template id (use list-templates to discover)")
    insert_tpl_cmd.add_argument("--anchor", help="Anchor block id (defaults to last non-output block)")

    list_tpl_cmd = sub.add_parser("list-templates")
    list_tpl_cmd.add_argument("--category", choices=("backbone", "neck", "head"))

    args = parser.parse_args()
    try:
        if args.cmd == "list-templates":
            for t in list_templates():
                if args.category and t["category"] != args.category:
                    continue
                print(f"{t['id']:24s} [{t['category']:8s}] {t['family']:12s} — {t['label']}")
            return 0
        if args.cmd == "import-onnx":
            pretrained = None
            if args.pretrained == "yes":
                pretrained = True
            elif args.pretrained == "no":
                pretrained = False
            scene = import_onnx_scene(
                args.onnx, model_name=args.name, family=args.family, class_count=args.classes, pretrained=pretrained
            )
            _write_json(args.output, scene)
            return 0
        if args.cmd == "import-yaml":
            scene = import_yaml_scene(args.yaml)
            _write_json(args.output, scene)
            return 0

        if args.cmd == "validate":
            validate_scene(_read_json(args.scene))
            print("Scene valid")
            return 0

        scene = _read_json(args.scene)
        validate_scene(scene)
        if args.cmd == "cut":
            cut_blocks(scene, set(args.ids))
            _write_json(args.output, scene)
            return 0
        if args.cmd == "prune":
            prune_block(scene, args.id, args.param, args.value)
            _write_json(args.output, scene)
            return 0
        if args.cmd == "replace":
            replace_block(scene, args.id, args.type)
            _write_json(args.output, scene)
            return 0
        if args.cmd == "fuse":
            fuse_conv_bn_silu(scene)
            _write_json(args.output, scene)
            return 0
        if args.cmd == "freeze":
            set_blocks_frozen(scene, set(args.ids), args.value == "true")
            _write_json(args.output, scene)
            return 0
        if args.cmd == "insert-template":
            insert_template(scene, args.template, anchor_id=args.anchor)
            _write_json(args.output, scene)
            return 0
        if args.cmd == "export-yaml":
            Path(args.output).write_text(export_scene_yaml(scene), encoding="utf-8")
            return 0
        if args.cmd == "export-pytorch":
            exported = export_scene_pytorch(scene, module_name=args.module_name)
            Path(args.output).write_text(exported["python"], encoding="utf-8")
            config_path = args.config_output or str(Path(args.output).with_suffix(".json"))
            Path(config_path).write_text(exported["config"], encoding="utf-8")
            return 0
        if args.cmd == "export-onnx":
            onnx_bytes = export_scene_onnx(scene, opset_version=args.opset, include_weights=not args.no_weights)
            Path(args.output).write_bytes(onnx_bytes)
            return 0
        if args.cmd == "quantize":
            quantize_scene(scene, mode=args.mode)
            _write_json(args.output, scene)
            return 0
        if args.cmd == "dequantize":
            dequantize_scene(scene)
            _write_json(args.output, scene)
            return 0
        if args.cmd == "lr-scale":
            if args.component:
                set_component_lr_scale(scene, args.component, args.value)
            elif args.ids:
                set_lr_scale(scene, set(args.ids), args.value)
            else:
                print("Error: --ids or --component required")
                return 2
            _write_json(args.output, scene)
            return 0
        if args.cmd == "freeze-component":
            freeze_component(scene, args.component, args.value == "true")
            _write_json(args.output, scene)
            return 0
        if args.cmd == "edge-constraints":
            set_edge_constraints(scene, max_size_mb=args.max_size_mb, latency_target_ms=args.latency_target_ms)
            _write_json(args.output, scene)
            return 0
        if args.cmd == "pretrained-config":
            set_pretrained_config(
                scene,
                source=args.source,
                inherit_weights=args.inherit == "true",
                strict=args.strict == "true",
            )
            _write_json(args.output, scene)
            return 0
        return 1
    except (SceneValidationError, KeyError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
