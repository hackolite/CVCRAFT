"""CVCRAFT CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .editor import cut_blocks, fuse_conv_bn_silu, prune_block, replace_block
from .exporter import export_scene_yaml
from .onnx_importer import import_onnx_scene
from .pytorch_exporter import export_scene_pytorch
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

    args = parser.parse_args()
    try:
        if args.cmd == "import-onnx":
            scene = import_onnx_scene(args.onnx, model_name=args.name, family=args.family, class_count=args.classes)
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
        if args.cmd == "export-yaml":
            Path(args.output).write_text(export_scene_yaml(scene), encoding="utf-8")
            return 0
        if args.cmd == "export-pytorch":
            exported = export_scene_pytorch(scene, module_name=args.module_name)
            Path(args.output).write_text(exported["python"], encoding="utf-8")
            config_path = args.config_output or str(Path(args.output).with_suffix(".json"))
            Path(config_path).write_text(exported["config"], encoding="utf-8")
            return 0
        return 1
    except (SceneValidationError, KeyError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
