"""
scripts/transcribe.py
----------------------
Transcrit un manuscrit et sauvegarde tous les outputs dans output/{folio}/.

Usage:
    # Sans vérité terrain
    python scripts/transcribe.py --image data/raw/CREMMA-MSS-17/data/recueil-lettres-pieces/f279.jpeg

    # Avec vérité terrain (XML ALTO) pour les métriques CER / ACC
    python scripts/transcribe.py \\
        --image data/raw/CREMMA-MSS-17/data/recueil-lettres-pieces/f279.jpeg \\
        --xml   data/raw/CREMMA-MSS-17/data/recueil-lettres-pieces/f279.xml

    # Avec un modèle spécifique
    python scripts/transcribe.py \\
        --image data/raw/.../f279.jpeg \\
        --xml   data/raw/.../f279.xml \\
        --model models/kraken_cremma_v1/model_10.mlmodel

Sorties dans output/{folio}/ :
    {folio}.jpeg               copie de l'image source
    {folio}_output.json        résultats complets (PRED, GT, CER, conf, stats)
    {folio}_transcription.txt  sortie lisible ligne à ligne
"""

import argparse
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.htr.kraken_htr import CONFIDENCE_THRESHOLD


# ── Parsing des arguments ─────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Transcription HTR — manuscrit unique")
    p.add_argument("--image",  required=True, help="Chemin vers l'image source (.jpeg/.jpg/.png)")
    p.add_argument("--xml",    default=None,  help="Chemin vers la vérité terrain ALTO XML (optionnel)")
    p.add_argument("--model",  default=None,  help="Chemin vers le modèle .mlmodel (détection auto si absent)")
    p.add_argument("--out-dir", default="output", help="Dossier de sortie racine (défaut: output/)")
    return p.parse_args()


# ── Lecture de la vérité terrain ALTO XML ────────────────────────────────────

def load_gt_lines(xml_path: str) -> list[str]:
    """Extrait les lignes de transcription depuis un fichier ALTO XML."""
    ns_list = [
        {"alto": "http://www.loc.gov/standards/alto/ns-v4#"},
        {"alto": "http://www.loc.gov/standards/alto/ns-v2#"},
        {},
    ]
    tree = ET.parse(xml_path)
    for ns in ns_list:
        tag   = "alto:TextLine" if ns else "TextLine"
        lines = (
            tree.findall(f".//{tag}", ns) if ns else tree.findall(".//TextLine")
        )
        if not lines:
            continue
        result = []
        for line in lines:
            s_tag = "alto:String" if ns else "String"
            parts = [
                s.get("CONTENT", "")
                for s in (line.findall(s_tag, ns) if ns else line.findall("String"))
                if s.get("CONTENT", "")
            ]
            text = " ".join(parts).strip()
            if text:
                result.append(text)
        if result:
            return result
    return []


# ── Segmentation + HTR ────────────────────────────────────────────────────────

def run_htr(image_path: str, model_path: str) -> tuple[list, list]:
    """Segmente et transcrit l'image. Retourne (seg.lines, preds)."""
    from kraken import blla, rpred
    from kraken.lib import models as kraken_models
    from PIL import Image

    img   = Image.open(image_path).convert("RGB")
    model = kraken_models.load_any(model_path)
    seg   = blla.segment(img)
    preds = list(rpred.rpred(model, img, seg))
    return seg.lines, preds


# ── Matching Levenshtein GT ↔ PRED ───────────────────────────────────────────

def match_lines(preds, gt_lines: list[str]) -> list[dict]:
    """Aligne chaque prédiction avec la GT la plus proche (Levenshtein)."""
    from rapidfuzz.distance import Levenshtein

    used_gt: set[int] = set()
    rows = []

    for i, pred in enumerate(preds):
        confs = pred.confidences
        conf  = float(sum(confs) / len(confs)) if confs else 0.0
        hyp   = pred.prediction

        best_j, best_dist = -1, float("inf")
        if gt_lines:
            for j, gt_candidate in enumerate(gt_lines):
                d = Levenshtein.distance(hyp, gt_candidate)
                if d < best_dist:
                    best_dist, best_j = d, j

        gt          = gt_lines[best_j] if best_j >= 0 else ""
        cer         = best_dist / max(len(gt), 1) if gt else None
        acc         = round(1.0 - cer, 4) if cer is not None else None
        gt_reused   = best_j in used_gt if best_j >= 0 else False
        if best_j >= 0:
            used_gt.add(best_j)

        rows.append({
            "line_idx":     i + 1,
            "pred":         hyp,
            "gt":           gt,
            "gt_idx":       best_j,
            "gt_reused":    gt_reused,
            "conf":         round(conf, 4),
            "needs_review": conf < CONFIDENCE_THRESHOLD or len(hyp) < 2,
            "acc":          acc,
            "cer":          round(cer, 4) if cer is not None else None,
        })

    return rows, used_gt


# ── Construction du JSON de sortie ───────────────────────────────────────────

def build_output_json(image_path: str,
                      model_path: str,
                      rows: list[dict],
                      gt_lines: list[str],
                      used_gt: set[int]) -> dict:
    has_gt = bool(gt_lines)

    if has_gt and rows:
        valid = [r for r in rows if r["cer"] is not None]
        total_dist  = sum(
            int(round(r["cer"] * max(len(r["gt"]), 1))) for r in valid
        )
        total_chars = sum(max(len(r["gt"]), 1) for r in valid)
        global_cer  = round(total_dist / total_chars, 4) if total_chars else None
        avg_acc     = round(sum(r["acc"] for r in valid) / len(valid), 4) if valid else None
        gt_unmatched = sorted(set(range(len(gt_lines))) - used_gt)
    else:
        global_cer = avg_acc = None
        gt_unmatched = []

    return {
        "image":              Path(image_path).name,
        "image_path":         str(image_path),
        "model":              model_path,
        "timestamp":          datetime.now().isoformat(),
        "n_lines_segmented":  len(rows),
        "n_gt_lines":         len(gt_lines),
        "global_cer":         global_cer,
        "global_acc":         round(1 - global_cer, 4) if global_cer is not None else None,
        "avg_acc_per_line":   avg_acc,
        "gt_unmatched_idx":   gt_unmatched,
        "lines":              rows,
    }


# ── Fichier texte lisible ─────────────────────────────────────────────────────

def save_txt(path: Path, data: dict) -> None:
    has_gt = data["n_gt_lines"] > 0
    lines  = [
        f"{'='*62}",
        f"  Transcription HTR — {data['image']}",
        f"  Date    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Modèle  : {data['model']}",
        f"  Lignes segmentées : {data['n_lines_segmented']}",
    ]
    if has_gt:
        lines += [
            f"  CER global        : {data['global_cer']*100:.1f}%",
            f"  Acc globale (CER) : {data['global_acc']*100:.1f}%",
            f"  Acc moy / ligne   : {data['avg_acc_per_line']*100:.1f}%",
            f"  GT non matchees   : {data['gt_unmatched_idx']}",
        ]
    lines += [f"{'='*62}", ""]

    for r in data["lines"]:
        review_flag = " [REVIEW]" if r.get("needs_review") else ""
        gt_flag     = " [GT reutilisee]" if r["gt_reused"] else ""
        acc_str     = f" | Acc: {r['acc']:.2f} | CER: {r['cer']:.2f}" if r["cer"] is not None else ""
        lines.append(f"Ligne {r['line_idx']:>2} | Conf: {r['conf']:.2f}{acc_str}{review_flag}{gt_flag}")
        lines.append(f"  PRED : {r['pred']}")
        if has_gt:
            lines.append(f"  GT   : {r['gt']}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[TXT]  Transcription texte  -> {path}")


# ── Journal ───────────────────────────────────────────────────────────────────

def log_run(data: dict) -> None:
    journal = Path("experiments/journal.jsonl")
    journal.parent.mkdir(exist_ok=True)
    entry = {
        "timestamp":           data["timestamp"],
        "type":                "transcription",
        "image":               data["image_path"],
        "model":               data["model"],
        "n_lines_segmented":   data["n_lines_segmented"],
        "n_gt_lines":          data["n_gt_lines"],
        "global_cer":          data["global_cer"],
        "global_acc":          data["global_acc"],
        "avg_acc_per_line":    data["avg_acc_per_line"],
    }
    with open(journal, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[LOG]  Journal mis a jour   -> {journal}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌  Image introuvable : {image_path}")
        sys.exit(1)

    # Résolution du modèle
    model_path = args.model
    if model_path is None:
        candidates = sorted(Path("models").glob("**/*.mlmodel"))
        if not candidates:
            print("[ERR]  Aucun modele .mlmodel trouve dans models/")
            print("   Lancez d'abord l'entrainement ou precisez --model")
            sys.exit(1)
        # Préférer le dernier checkpoint (numéro le plus élevé)
        model_path = str(
            max(candidates, key=lambda p: int(p.stem.split("_")[-1])
                if p.stem.split("_")[-1].isdigit() else -1)
        )
        print(f"   Modèle sélectionné : {model_path}")

    # Dossier de sortie dédié au folio
    stem    = image_path.stem
    out_dir = Path(args.out_dir) / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"  Transcription — {image_path.name}")
    print(f"{'='*62}\n")

    # Vérité terrain
    gt_lines: list[str] = []
    if args.xml:
        xml_path = Path(args.xml)
        if not xml_path.exists():
            print(f"[WARN]  Fichier XML introuvable : {xml_path} -- GT ignoree")
        else:
            gt_lines = load_gt_lines(str(xml_path))
            print(f"   Verites terrain : {len(gt_lines)} lignes")

    # HTR
    print("\n[ 1/3 ] Segmentation + transcription (Kraken)…")
    seg_lines, preds = run_htr(str(image_path), model_path)
    print(f"   Lignes segmentées : {len(seg_lines)}")
    print(f"   Prédictions       : {len(preds)}")

    # Alignement GT
    print("\n[ 2/3 ] Alignement GT <-> PRED...")
    rows, used_gt = match_lines(preds, gt_lines)

    # Affichage console
    for r in rows:
        flag    = " [GT deja utilisee]" if r["gt_reused"] else ""
        acc_str = f" | Acc: {r['acc']:.2f} | CER: {r['cer']:.2f}" if r["cer"] is not None else ""
        print(f"Ligne {r['line_idx']:>2} | Conf: {r['conf']:.2f}{acc_str} | GT matchee: #{r['gt_idx']}{flag}")
        print(f"  PRED : {r['pred']}")
        if gt_lines:
            print(f"  GT   : {r['gt']}")
        print()

    data = build_output_json(str(image_path), model_path, rows, gt_lines, used_gt)

    if gt_lines:
        print(f"{'='*50}")
        print(f"CER global       : {data['global_cer']:.4f} ({data['global_cer']*100:.1f}%)")
        print(f"Acc globale (CER): {data['global_acc']:.4f} ({data['global_acc']*100:.1f}%)")
        print(f"Acc moyenne/ligne: {data['avg_acc_per_line']:.4f} ({data['avg_acc_per_line']*100:.1f}%)")
        print(f"GT non matchees  : {data['gt_unmatched_idx']}")
        print(f"{'='*50}\n")

    # Sauvegarde des outputs
    print("[ 3/3 ] Sauvegarde des outputs...")

    # 1. Copie de l'image
    img_dest = out_dir / image_path.name
    shutil.copy2(image_path, img_dest)
    print(f"[IMG]  Image copiee        -> {img_dest}")

    # 2. JSON complet
    json_dest = out_dir / f"{stem}_output.json"
    json_dest.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[JSON] Resultats           -> {json_dest}")

    # 3. Texte lisible
    txt_dest = out_dir / f"{stem}_transcription.txt"
    save_txt(txt_dest, data)

    # 4. Journal
    log_run(data)

    print(f"\n{'='*62}")
    print(f"  OK  Outputs sauvegardes dans : {out_dir}/")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()