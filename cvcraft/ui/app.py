"""CVCRAFT Flask Web UI."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from ..editor import (add_block_to_scene, cut_blocks, fuse_conv_bn_silu,
                       insert_block_on_edge, prune_block, replace_block,
                       set_blocks_frozen)
from ..exporter import export_scene_yaml
from ..onnx_importer import import_onnx_scene
from ..pytorch_exporter import export_scene_pytorch
from ..scene_v2 import normalize_scene_v2
from ..scheduler import schedule_scene_stages
from ..templates import (
    get_block_default_hyperparameters,
    insert_template,
    list_block_default_hyperparameters,
    list_templates,
)
from ..validator import SceneValidationError, validate_scene
from ..yaml_importer import import_yaml_scene

logger = logging.getLogger(__name__)


def _safe_error_message(exc: Exception) -> str:
    """Return a user-safe error message without exposing internal stack details.

    Only SceneValidationError messages are designed to be user-facing.
    All other exceptions return generic descriptions.
    """
    if isinstance(exc, SceneValidationError):
        # SceneValidationError messages are explicitly constructed strings
        # that describe validation issues — safe to expose.
        msg = exc.args[0] if exc.args else "Scene validation failed"
        return msg
    if isinstance(exc, KeyError):
        return "A required key is missing from the input"
    if isinstance(exc, ValueError):
        return "An invalid value was provided"
    if isinstance(exc, RuntimeError):
        return "A required dependency is not available"
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
            schedule_scene_stages(scene)
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
            schedule_scene_stages(scene)
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
            schedule_scene_stages(scene)
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
            schedule_scene_stages(scene)
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
            schedule_scene_stages(scene)
            return jsonify({"scene": scene, "result": result})
        except (SceneValidationError, KeyError, ValueError) as exc:
            return jsonify({"error": _safe_error_message(exc)}), 400

    @app.route("/api/edit/add-block", methods=["POST"])
    def api_add_block():
        """Add a new block to the scene, optionally after an anchor block.

        Body: ``{scene, type, params, anchor_id?}``
        """
        data = request.get_json(force=True)
        scene = data.get("scene")
        block_type = data.get("type")
        params = data.get("params", {})
        anchor_id = data.get("anchor_id") or None
        if not isinstance(scene, dict) or not block_type:
            return jsonify({"error": "scene and type are required"}), 400
        try:
            validate_scene(scene)
            result = add_block_to_scene(scene, block_type, params, anchor_id=anchor_id)
            schedule_scene_stages(scene)
            return jsonify({"scene": scene, "result": result})
        except (SceneValidationError, KeyError, ValueError) as exc:
            return jsonify({"error": _safe_error_message(exc)}), 400

    @app.route("/api/edit/insert-on-edge", methods=["POST"])
    def api_insert_on_edge():
        """Insert a new block between two directly connected blocks.

        Body: ``{scene, type, params, from, to}``
        """
        data = request.get_json(force=True)
        scene = data.get("scene")
        block_type = data.get("type")
        params = data.get("params", {})
        from_id = data.get("from")
        to_id = data.get("to")
        if not isinstance(scene, dict) or not block_type or not from_id or not to_id:
            return jsonify({"error": "scene, type, from and to are required"}), 400
        try:
            validate_scene(scene)
            result = insert_block_on_edge(scene, block_type, params, from_id, to_id)
            schedule_scene_stages(scene)
            return jsonify({"scene": scene, "result": result})
        except (SceneValidationError, KeyError, ValueError) as exc:
            return jsonify({"error": _safe_error_message(exc)}), 400

    # ------------------------------------------------------------------
    # Compatibility check (channel dimensions)
    # ------------------------------------------------------------------
    @app.route("/api/validate/compat", methods=["POST"])
    def api_validate_compat():
        """Return edges with channel-dimension mismatches.

        Body: scene JSON.
        Response: ``{issues: [{from, to, fromOut, toIn}]}``
        """
        scene = request.get_json(force=True)
        if not scene or "blocks" not in scene or "edges" not in scene:
            return jsonify({"issues": []})
        blocks_by_id = {b["id"]: b for b in scene["blocks"]}
        issues = []
        for edge in scene.get("edges", []):
            fid = edge.get("from")
            tid = edge.get("to")
            fb = blocks_by_id.get(fid)
            tb = blocks_by_id.get(tid)
            if not fb or not tb:
                continue
            from_out = (fb.get("params") or {}).get("out_channels") or \
                       (fb.get("meta") or {}).get("out_channels")
            to_in = (tb.get("params") or {}).get("in_channels") or \
                    (tb.get("meta") or {}).get("in_channels")
            if from_out is not None and to_in is not None:
                try:
                    if int(from_out) != int(to_in):
                        issues.append({
                            "from": fid, "to": tid,
                            "fromOut": int(from_out), "toIn": int(to_in),
                        })
                except (TypeError, ValueError):
                    pass
        return jsonify({"issues": issues})

    @app.route("/api/edit/update-block", methods=["POST"])
    def api_update_block():
        """Update block properties inline (Netron-style).

        Validates the scene after applying changes. If validation fails,
        returns the error without applying the change (dry-run approach).
        """
        data = request.get_json(force=True)
        scene = data["scene"]
        block_id = data["id"]
        updates = data.get("updates", {})  # e.g. {"params": {...}, "meta": {...}}
        try:
            validate_scene(scene)
            # Find target block
            target = None
            for b in scene["blocks"]:
                if b["id"] == block_id:
                    target = b
                    break
            if target is None:
                raise KeyError(f"Unknown block id: {block_id}")

            # Save original values for rollback
            import copy
            original = copy.deepcopy(target)

            # Apply updates
            applied = {}
            if "params" in updates:
                for k, v in updates["params"].items():
                    target.setdefault("params", {})[k] = v
                    applied.setdefault("params", {})[k] = v
            if "meta" in updates:
                for k, v in updates["meta"].items():
                    target.setdefault("meta", {})[k] = v
                    applied.setdefault("meta", {})[k] = v
            if "type" in updates:
                target["type"] = updates["type"]
                applied["type"] = updates["type"]

            # Validate after modification — rollback if broken
            try:
                validate_scene(scene)
            except SceneValidationError as ve:
                # Rollback: restore original block state
                for b in scene["blocks"]:
                    if b["id"] == block_id:
                        b.update(original)
                        break
                return jsonify({
                    "error": f"Modification rejected: {_safe_error_message(ve)}",
                    "rejected": True,
                    "reason": _safe_error_message(ve),
                }), 400

            schedule_scene_stages(scene)
            return jsonify({"scene": scene, "result": {"updated": [{"id": block_id, "applied": applied}]}})
        except (SceneValidationError, KeyError, ValueError) as exc:
            return jsonify({"error": _safe_error_message(exc)}), 400

    # ------------------------------------------------------------------
    # Templates (VGG-like + anchor-free detection presets)
    # ------------------------------------------------------------------
    @app.route("/api/templates", methods=["GET"])
    def api_templates():
        return jsonify(
            {
                "templates": list_templates(),
                "blockDefaults": list_block_default_hyperparameters(),
            }
        )

    @app.route("/api/templates/block-defaults/<block_type>", methods=["GET"])
    def api_template_block_defaults(block_type: str):
        return jsonify({"type": block_type, "params": get_block_default_hyperparameters(block_type)})

    @app.route("/api/templates/insert", methods=["POST"])
    def api_template_insert():
        data = request.get_json(force=True)
        scene = data.get("scene")
        template_id = data.get("template_id")
        anchor_id = data.get("anchor_id")
        if not isinstance(scene, dict) or not template_id:
            return jsonify({"error": "scene and template_id are required"}), 400
        try:
            validate_scene(scene)
            result = insert_template(scene, template_id, anchor_id=anchor_id)
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

    # ------------------------------------------------------------------
    # Normalize (recompute metadata without changing positions)
    # ------------------------------------------------------------------
    @app.route("/api/normalize", methods=["POST"])
    def api_normalize():
        scene = request.get_json(force=True)
        try:
            normalize_scene_v2(scene)
            return jsonify(scene)
        except (KeyError, ValueError) as exc:
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
