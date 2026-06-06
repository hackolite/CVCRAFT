#!/usr/bin/env python3
"""
Script de diagnostic pour identifier les blocs ONNX non supportés dans CVCRAFT.

Usage:
    python diagnostic_unsupported.py <path_to_onnx_file>

Ce script analyse un fichier ONNX et identifie tous les blocs qui seraient
importés comme 'UnsupportedOpBlock', avec des détails sur le type d'opération
ONNX original et leur position dans le graphe.
"""

from __future__ import annotations

import sys
from pathlib import Path


def diagnose_unsupported_blocks(onnx_path: str) -> list[dict] | None:
    """
    Analyse un fichier ONNX et identifie les blocs non supportés.

    Args:
        onnx_path: Chemin vers le fichier ONNX à analyser

    Returns:
        Liste des blocs non supportés avec leurs métadonnées, ou None en cas d'erreur
    """
    try:
        from cvcraft.onnx_importer import import_onnx_scene
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        print("Assurez-vous que CVCRAFT est installé: pip install -e .")
        return None

    try:
        scene = import_onnx_scene(onnx_path)
        unsupported = [b for b in scene["blocks"] if b["type"] == "UnsupportedOpBlock"]

        print(f"\n📊 Analyse du modèle: {scene['model']['name']}")
        print(f"   Famille: {scene['model']['family']}")
        print(f"   Total de blocs: {len(scene['blocks'])}")
        print(f"   Total d'arêtes: {len(scene['edges'])}")
        print()

        if unsupported:
            print(f"⚠️  {len(unsupported)} bloc(s) non supporté(s) trouvé(s):")
            print(f"   ({len(unsupported) / len(scene['blocks']) * 100:.1f}% du total)")
            print()

            # Compter les types d'opérations non supportées
            op_counts: dict[str, int] = {}
            for block in unsupported:
                op_type = block["params"].get("onnx_op", "Unknown")
                op_counts[op_type] = op_counts.get(op_type, 0) + 1

            print("📋 Types d'opérations ONNX non supportées:")
            for op_type, count in sorted(op_counts.items(), key=lambda x: -x[1]):
                print(f"   • {op_type}: {count} occurrence(s)")
            print()

            print("🔍 Détails des blocs non supportés:")
            for block in unsupported:
                print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━")
                print(f"   Block ID: {block['id']}")
                print(f"   ONNX Op: {block['params'].get('onnx_op', 'Unknown')}")
                print(f"   ONNX Name: {block['params'].get('onnx_name', 'Unknown')}")
                print(f"   Position: ({block['position']['x']}, {block['position']['y']}, {block['position']['z']})")
                print(f"   Inputs: {', '.join(block['io']['in']) if block['io']['in'] else 'None'}")
                print(f"   Outputs: {', '.join(block['io']['out']) if block['io']['out'] else 'None'}")
                if block["params"].get("has_weights"):
                    print(f"   ⚠️  Ce bloc contient des poids appris")
                print()

            print("\n💡 Suggestions:")
            print("   1. Vérifiez le README.md pour la liste des blocs supportés")
            print("   2. Ces opérations pourraient être ajoutées au mapping ONNX dans onnx_importer.py")
            print("   3. Consultez la documentation ONNX pour comprendre ces opérations:")
            print("      https://github.com/onnx/onnx/blob/main/docs/Operators.md")

        else:
            print("✅ Tous les blocs sont supportés!")
            print("   Ce modèle peut être importé et édité sans problème.")

        return unsupported

    except Exception as e:
        print(f"❌ Erreur lors de l'analyse: {e}")
        import traceback

        traceback.print_exc()
        return None


def main() -> int:
    """Point d'entrée principal du script."""
    if len(sys.argv) < 2:
        print("Usage: python diagnostic_unsupported.py <path_to_onnx>")
        print()
        print("Exemple:")
        print("  python diagnostic_unsupported.py models/yolox_s.onnx")
        return 1

    onnx_path = sys.argv[1]

    if not Path(onnx_path).exists():
        print(f"❌ Fichier introuvable: {onnx_path}")
        return 1

    result = diagnose_unsupported_blocks(onnx_path)

    if result is None:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
