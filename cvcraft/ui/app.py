"""CVCRAFT Flask Web UI."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from ..editor import cut_blocks, fuse_conv_bn_silu, prune_block, replace_block, set_blocks_frozen
from ..exporter import export_scene_yaml
from ..onnx_importer import import_onnx_scene
from ..pytorch_exporter import export_scene_pytorch
from ..scheduler import schedule_scene_stages
from ..validator import SceneValidationError, validate_scene
from ..yaml_importer import import_yaml_scene

logger = logging.getLogger(__name__)


def _safe_error_message(exc: Exception) -> str:
    """Return a user-safe error message without exposing internal stack details."""
    if isinstance(exc, SceneValidationError):
        return f"Validation error: {exc}"
    if isinstance(exc, KeyError):
        return f"Missing key: {exc}"
    if isinstance(exc, ValueError):
        return f"Invalid value: {exc}"
    if isinstance(exc, RuntimeError):
        return f"Runtime error: {exc}"
    # Generic fallback — do not expose internal details
    logger.exception("Unexpected error in API handler")
    return "An internal error occurred"


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB

    @app.route("/")
    def index():
        return render_template("index.html")

    # ------------------------------------------------------------------
    # Scene validation
    # ------------------------------------------------------------------
    @app.route("/api/validate", methods=["POST"])
    def api_validate():
        scene = request.get_json(force=True)
        try:
            validate_scene(scene)
            return jsonify({"valid": True})
        except SceneValidationError as exc:
            return jsonify({"valid": False, "error": _safe_error_message(exc)}), 400

    # ------------------------------------------------------------------
    # Import ONNX
    # ------------------------------------------------------------------
    @app.route("/api/import/onnx", methods=["POST"])
    def api_import_onnx():
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        model_name = request.form.get("name") or None
        family = request.form.get("family", "YOLOX")
        class_count = int(request.form.get("classes", "80"))
        pretrained_raw = request.form.get("pretrained", "auto")
        pretrained: bool | None = None
        if pretrained_raw == "yes":
            pretrained = True
        elif pretrained_raw == "no":
            pretrained = False

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            file.save(tmp)
            tmp_path = tmp.name
        try:
            scene = import_onnx_scene(
                tmp_path,
                model_name=model_name,
                family=family,
                class_count=class_count,
                pretrained=pretrained,
            )
            return jsonify(scene)
        except (SceneValidationError, RuntimeError, ValueError) as exc:
            return jsonify({"error": _safe_error_message(exc)}), 400
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Import YAML
    # ------------------------------------------------------------------
    @app.route("/api/import/yaml", methods=["POST"])
    def api_import_yaml():
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="wb") as tmp:
            file.save(tmp)
            tmp_path = tmp.name
        try:
            scene = import_yaml_scene(tmp_path)
            return jsonify(scene)
        except (SceneValidationError, RuntimeError, ValueError) as exc:
            return jsonify({"error": _safe_error_message(exc)}), 400
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Edit operations
    # ------------------------------------------------------------------
    @app.route("/api/edit/cut", methods=["POST"])
    def api_cut():
        data = request.get_json(force=True)
        scene = data["scene"]
        block_ids = set(data["ids"])
        try:
            validate_scene(scene)
            result = cut_blocks(scene, block_ids)
            return jsonify({"scene": scene, "result": result})
        except (SceneValidationError, KeyError, ValueError) as exc:
            return jsonify({"error": _safe_error_message(exc)}), 400

    @app.route("/api/edit/prune", methods=["POST"])
    def api_prune():
        data = request.get_json(force=True)
        scene = data["scene"]
        block_id = data["id"]
        param = data["param"]
        value = int(data["value"])
        try:
            validate_scene(scene)
            result = prune_block(scene, block_id, param, value)
            return jsonify({"scene": scene, "result": result})
        except (SceneValidationError, KeyError, ValueError) as exc:
            return jsonify({"error": _safe_error_message(exc)}), 400

    @app.route("/api/edit/replace", methods=["POST"])
    def api_replace():
        data = request.get_json(force=True)
        scene = data["scene"]
        block_id = data["id"]
        new_type = data["type"]
        try:
            validate_scene(scene)
            result = replace_block(scene, block_id, new_type)
            return jsonify({"scene": scene, "result": result})
        except (SceneValidationError, KeyError, ValueError) as exc:
            return jsonify({"error": _safe_error_message(exc)}), 400

    @app.route("/api/edit/fuse", methods=["POST"])
    def api_fuse():
        data = request.get_json(force=True)
        scene = data["scene"]
        try:
            validate_scene(scene)
            result = fuse_conv_bn_silu(scene)
            return jsonify({"scene": scene, "result": result})
        except (SceneValidationError, KeyError, ValueError) as exc:
            return jsonify({"error": _safe_error_message(exc)}), 400

    @app.route("/api/edit/freeze", methods=["POST"])
    def api_freeze():
        data = request.get_json(force=True)
        scene = data["scene"]
        block_ids = set(data["ids"])
        frozen = data.get("value", True)
        try:
            validate_scene(scene)
            result = set_blocks_frozen(scene, block_ids, frozen)
            return jsonify({"scene": scene, "result": result})
        except (SceneValidationError, KeyError, ValueError) as exc:
            return jsonify({"error": _safe_error_message(exc)}), 400

    # ------------------------------------------------------------------
    # Export operations
    # ------------------------------------------------------------------
    @app.route("/api/export/yaml", methods=["POST"])
    def api_export_yaml():
        scene = request.get_json(force=True)
        try:
            validate_scene(scene)
            yaml_output = export_scene_yaml(scene)
            return jsonify({"yaml": yaml_output})
        except (SceneValidationError, KeyError, ValueError) as exc:
            return jsonify({"error": _safe_error_message(exc)}), 400

    @app.route("/api/export/pytorch", methods=["POST"])
    def api_export_pytorch():
        data = request.get_json(force=True)
        scene = data.get("scene", data)
        module_name = data.get("module_name", "GeneratedVoxelModel")
        try:
            validate_scene(scene)
            exported = export_scene_pytorch(scene, module_name=module_name)
            return jsonify({"python": exported["python"], "config": exported["config"]})
        except (SceneValidationError, KeyError, ValueError) as exc:
            return jsonify({"error": _safe_error_message(exc)}), 400

    # ------------------------------------------------------------------
    # Schedule (re-layout 3D positions)
    # ------------------------------------------------------------------
    @app.route("/api/schedule", methods=["POST"])
    def api_schedule():
        scene = request.get_json(force=True)
        try:
            validate_scene(scene)
            schedule_scene_stages(scene)
            return jsonify(scene)
        except (SceneValidationError, KeyError, ValueError) as exc:
            return jsonify({"error": _safe_error_message(exc)}), 400

    return app


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="cvcraft-ui", description="CVCRAFT Web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
