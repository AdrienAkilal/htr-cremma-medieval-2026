"""
scripts/run_pipeline.py
------------------------
Lance le pipeline HTR complet sur une image.

Usage:
    python scripts/run_pipeline.py --image data/raw/folio.jpeg
    python scripts/run_pipeline.py --image data/raw/folio.jpeg --model models/kraken_cremma_v1/best.mlmodel
"""

import argparse
import json
import sys
from pathlib import Path

# Ajout du dossier racine au PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.preprocess import preprocess_image, fixer_seeds
from src.segmentation.yologen_obb import main as run_yologen
from src.segmentation.kraken_segment import segment_page, validate_polygons
from src.htr.kraken_htr import transcribe
from src.aggregation.aggregate import (
    build_data_contract, save_data_contract, export_page_xml
)
from src.nlp.normalize import process_contract


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline HTR — manuscrits médiévaux")
    parser.add_argument("--image",   required=True, help="Chemin vers l'image source")
    parser.add_argument("--model",   default=None,  help="Chemin vers le modèle Kraken (.mlmodel)")
    parser.add_argument("--out-dir", default="output", help="Dossier de sortie (défaut: output/)")
    parser.add_argument("--skip-layout", action="store_true",
                        help="Passer la segmentation YOLO-gen")
    parser.add_argument("--seed",    type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    fixer_seeds(args.seed)

    image_path = args.image
    stem       = Path(image_path).stem
    out_dir    = Path(args.out_dir) / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Pipeline HTR — {Path(image_path).name}")
    print(f"{'='*55}\n")

    # ── Étape 1 : Prétraitement ───────────────────────────────────────────────
    print("[ 1/5 ] Prétraitement…")
    preprocessed_path = str(out_dir / f"{stem}_preprocessed.jpg")
    preprocess_image(image_path,
                     save_dir=str(out_dir / "preprocessing_steps"))
    # Sauvegarde de la version finale pour Kraken
    import cv2
    img_bin = preprocess_image(image_path, binarize_enabled=False)
    cv2.imwrite(preprocessed_path, img_bin)
    print(f"   → {preprocessed_path}")

    # ── Étape 2 : Segmentation layout (YOLO-gen) ─────────────────────────────
    layout_regions = []
    if not args.skip_layout:
        print("\n[ 2/5 ] Segmentation layout (YOLO-gen)…")
        try:
            # yologen_obb.py est lancé en mode import
            from src.segmentation.yologen_obb import (
                load_model, diagnose_thresholds,
                visualize, report_and_export
            )
            from PIL import Image as PILImage
            model         = load_model()
            results, conf = diagnose_thresholds(model, image_path)
            img_pil       = PILImage.open(image_path).convert("RGB")
            visualize(results, img_pil, conf,
                      out_dir / f"{stem}_layout.jpg")
            report_and_export(results, conf,
                              out_dir / f"{stem}_layout.json")
            # Récupération des régions pour le data contract
            layout_json = out_dir / f"{stem}_layout.json"
            if layout_json.exists():
                layout_regions = json.loads(layout_json.read_text())["regions"]
        except Exception as e:
            print(f"   ⚠️  YOLO-gen non disponible : {e}")
            print("   → Segmentation layout ignorée")
    else:
        print("\n[ 2/5 ] Segmentation layout ignorée (--skip-layout)")

    # ── Étape 3 : Segmentation lignes (Kraken BLLA) ───────────────────────────
    print("\n[ 3/5 ] Segmentation des lignes (Kraken BLLA)…")
    xml_out = str(out_dir / f"{stem}_segmentation.page.xml")
    try:
        lines = segment_page(preprocessed_path, xml_out=xml_out)
        lines = validate_polygons(lines, preprocessed_path)
    except Exception as e:
        print(f"   ❌  Erreur Kraken segmentation : {e}")
        sys.exit(1)

    # ── Étape 4 : Transcription HTR ───────────────────────────────────────────
    print("\n[ 4/5 ] Transcription HTR (Kraken)…")
    model_path = args.model
    if model_path is None:
        # Chercher un modèle local dans models/
        candidates = list(Path("models").glob("**/*.mlmodel"))
        if candidates:
            model_path = str(candidates[0])
            print(f"   Modèle trouvé : {model_path}")
        else:
            print("   ⚠️  Aucun modèle fine-tuné trouvé.")
            print("   → Lancez d'abord : python scripts/train_kraken.py")
            sys.exit(1)

    try:
        transcriptions = transcribe(preprocessed_path, lines, model_path)
    except Exception as e:
        print(f"   ❌  Erreur Kraken HTR : {e}")
        sys.exit(1)

    # ── Étape 5 : Agrégation & export ────────────────────────────────────────
    print("\n[ 5/5 ] Agrégation et export…")
    contract = build_data_contract(
        image_path=image_path,
        transcriptions=transcriptions,
        layout_regions=layout_regions,
    )
    save_data_contract(contract, str(out_dir / f"{stem}_data_contract.json"))
    export_page_xml(contract, str(out_dir / f"{stem}_transcription_cv.page.xml"))

    # ── Sortie texte lisible ──────────────────────────────────────────────────
    txt_out = out_dir / f"{stem}_transcription.txt"
    _save_transcription_txt(txt_out, contract, model_path)

    # ── Étape 6 : Normalisation NLP ──────────────────────────────────────────
    print("\n[ 6/6 ] Normalisation NLP des transcriptions…")
    nlp_output = process_contract(contract)

    nlp_json_out = out_dir / f"{stem}_nlp_normalized.json"
    import json as _json
    nlp_json_out.write_text(
        _json.dumps(nlp_output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"📊  NLP JSON normalisé   → {nlp_json_out}")

    final_txt_out = out_dir / f"{stem}_final_transcription.txt"
    _save_final_transcription_txt(final_txt_out, nlp_output)

    print(f"\n{'='*55}")
    print(f"  ✅  Pipeline terminé")
    print(f"  Lignes : {contract['stats']['n_lines']}")
    print(f"  needs_review : {contract['stats']['needs_review_rate']*100:.1f}%")
    print(f"  Confiance moy : {contract['stats']['mean_confidence']:.3f}")
    print(f"  Sortie texte  : {txt_out}")
    print(f"  NLP normalisé : {nlp_json_out}")
    print(f"  Transcription finale : {final_txt_out}")
    print(f"{'='*55}\n")


def _save_final_transcription_txt(path: Path, nlp_output: dict) -> None:
    """Sauvegarde la transcription finale normalisée (sortie du module NLP)."""
    from datetime import datetime
    stats = nlp_output["stats"]

    header = (
        f"{'='*60}\n"
        f"  Transcription finale normalisée — {nlp_output['image']}\n"
        f"  Date    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"  Lignes  : {stats['n_lines']} "
        f"(traitées : {stats['n_processed']}, "
        f"review : {stats['n_skipped_review']})\n"
        f"  Corrections NLP : {stats['n_corrections']}\n"
        f"{'='*60}\n\n"
    )

    body = ""
    for line in nlp_output["lines"]:
        if line["needs_review"]:
            flag = " [REVIEW — non normalisé]"
        else:
            n = len(line["corrections_appliquees"])
            flag = f" [{n} correction(s)]" if n else ""
        lang = "/".join(line["langue_detectee"])
        body += (
            f"[{line['confidence']:.2f}][{lang}]{flag}\n"
            f"  {line['transcription_normalisee']}\n\n"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + body, encoding="utf-8")
    print(f"📄  Transcription finale → {path}")


def _save_transcription_txt(path: Path, contract: dict, model_path: str) -> None:
    """Sauvegarde un fichier texte lisible du résultat de transcription."""
    from datetime import datetime
    stats = contract["stats"]
    lines = contract["lines"]

    header = (
        f"{'='*60}\n"
        f"  Transcription HTR — {contract['image']}\n"
        f"  Date    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"  Modèle  : {model_path}\n"
        f"  Lignes  : {stats['n_lines']}\n"
        f"  Conf.   : {stats['mean_confidence']:.3f}\n"
        f"  Review  : {stats['n_needs_review']} lignes "
        f"({stats['needs_review_rate']*100:.1f}%)\n"
        f"{'='*60}\n\n"
    )

    body = ""
    for line in lines:
        flag = " ⚠" if line["needs_review"] else "  "
        body += f"[{line['confidence']:.2f}]{flag}  {line['text']}\n"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + body, encoding="utf-8")
    print(f"📄  Transcription texte → {path}")


if __name__ == "__main__":
    main()
