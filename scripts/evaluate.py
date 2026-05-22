"""
scripts/evaluate.py
--------------------
Évalue le pipeline sur un split donné (val ou test scellé).

⚠️  N'évaluez sur le test set qu'une seule fois, pour le rendu final.

Usage:
    python scripts/evaluate.py --split val
    python scripts/evaluate.py --split test --model models/kraken_cremma_v1/best.mlmodel
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import full_evaluation_report, compute_mean_iou
from src.segmentation.kraken_segment import segment_page, validate_polygons
from src.htr.kraken_htr import transcribe
from src.preprocessing.preprocess import preprocess_image


def run_evaluation(split: str,
                   model_path: str,
                   splits_dir: str = "data/splits") -> dict:
    """Lance l'évaluation complète sur un split.

    Args:
        split: "val" ou "test".
        model_path: Chemin vers le modèle Kraken (.mlmodel).
        splits_dir: Dossier contenant les splits JSON.

    Returns:
        Dict du rapport d'évaluation.
    """
    manifest_path = Path(splits_dir) / f"{split}.json"
    if not manifest_path.exists():
        print(f"❌  Split introuvable : {manifest_path}")
        sys.exit(1)

    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"📊  Évaluation sur {split} — {len(records)} exemples\n")

    predictions = []
    references  = []

    for i, record in enumerate(records):
        img_path = record["image_path"]
        ref_text = record.get("transcription", "")

        if not Path(img_path).exists():
            continue

        # Prétraitement
        preprocessed = preprocess_image(img_path, binarize_enabled=False)
        pre_path = f"/tmp/eval_pre_{i}.jpg"
        import cv2
        cv2.imwrite(pre_path, preprocessed)

        # Segmentation + HTR
        try:
            lines = segment_page(pre_path)
            lines = validate_polygons(lines, pre_path)
            results = transcribe(pre_path, lines, model_path)
            pred_text = " ".join(r["text"] for r in results)
        except Exception as e:
            print(f"   ⚠️  Erreur sur {img_path} : {e}")
            pred_text = ""

        predictions.append(pred_text)
        references.append(ref_text)

        if (i + 1) % 10 == 0:
            print(f"   {i+1}/{len(records)} traités…")

    report = full_evaluation_report(predictions, references)

    # Sauvegarde du rapport
    report_path = Path("experiments") / f"eval_{split}.json"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n📄  Rapport sauvegardé → {report_path}")
    return report


def parse_args():
    p = argparse.ArgumentParser(description="Évaluation HTR")
    p.add_argument("--split",  default="val", choices=["val", "test"])
    p.add_argument("--model",  required=True)
    p.add_argument("--splits-dir", default="data/splits")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args.split, args.model, args.splits_dir)
