import os
import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon, Patch

from ultralytics import YOLO
from huggingface_hub import hf_hub_download

IMAGE_PATH  = r"C:\Users\amazi\Downloads\yologen_project\Protocole_ou_registre_des_minutes_[...]_btv1b10088937p_121.jpeg"
OUTPUT_DIR  = Path(r"C:\Users\amazi\Downloads\yologen_project\yologen_output")
HF_MODEL_ID = "magistermilitum/YOLO_manuscripts"
HF_TOKEN    = os.environ.get("HF_TOKEN")
CONF_FINAL  = 0.10


# ─────────────────────────────────────────────────────────────────────────────
# 1. load_model
# ─────────────────────────────────────────────────────────────────────────────
def load_model() -> YOLO:
    """Télécharge best.pt depuis HuggingFace et retourne un objet YOLO."""
    print("📥  Téléchargement du modèle depuis Hugging Face…")
    try:
        model_path = hf_hub_download(
            repo_id=HF_MODEL_ID,
            filename="best.pt",
            token=HF_TOKEN,
        )
    except Exception as e:
        if "401" in str(e) or "403" in str(e):
            sys.exit(
                "❌  Accès refusé (401/403). Générez un token sur "
                "https://huggingface.co/settings/tokens et définissez "
                "la variable d'environnement HF_TOKEN."
            )
        raise
    print(f"   Modèle en cache : {model_path}")
    model = YOLO(model_path)
    print(f"   Classes détectées : {list(model.names.values())}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 2. diagnose_thresholds
# ─────────────────────────────────────────────────────────────────────────────
def diagnose_thresholds(model: YOLO, image_path: str):
    """
    Teste plusieurs seuils de confiance décroissants et retourne
    (results, conf_retenu) au premier seuil ayant ≥1 détection et ≤ CONF_FINAL.
    """
    thresholds = [0.50, 0.25, 0.10, 0.05, 0.01]

    print("\n┌─────────────────────────────────────────────┐")
    print("│          Diagnostic des seuils de conf.     │")
    print("├──────────┬──────────────────────────────────┤")
    print("│  Seuil   │  Détections                      │")
    print("├──────────┼──────────────────────────────────┤")

    selected_results = None
    selected_conf    = None

    for conf in thresholds:
        results = model.predict(
            source=image_path,
            conf=conf,
            iou=0.45,
            imgsz=1280,
            verbose=False,
        )
        n = 0
        for r in results:
            if r.obb is not None:
                n += len(r.obb)
        print(f"│  {conf:.2f}    │  {n:>4} région(s)                   │")

        if selected_results is None and n >= 1 and conf <= CONF_FINAL:
            selected_results = results
            selected_conf    = conf

    print("└──────────┴──────────────────────────────────┘")

    # Fallback : si rien en dessous de CONF_FINAL, on prend conf=0.01
    if selected_results is None:
        print("⚠️  Aucune détection ≤ CONF_FINAL, relance à 0.01…")
        selected_results = model.predict(
            source=image_path, conf=0.01, iou=0.45, imgsz=1280, verbose=False
        )
        selected_conf = 0.01

    total = sum(len(r.obb) if r.obb else 0 for r in selected_results)
    print(f"\n✅  Seuil retenu : {selected_conf}  →  {total} détection(s)\n")
    return selected_results, selected_conf


# ─────────────────────────────────────────────────────────────────────────────
# 3. visualize
# ─────────────────────────────────────────────────────────────────────────────
# Palette par classe
CLASS_COLORS = [
    "#e63946", "#2a9d8f", "#e9c46a", "#f4a261",
    "#457b9d", "#8ecae6", "#a8dadc", "#c77dff",
]

def visualize(results, img_pil: Image.Image, conf: float, save_path: Path) -> None:
    """Dessine les OBB sur l'image et sauvegarde la figure."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 18))
    ax.imshow(np.array(img_pil))
    ax.axis("off")
    ax.set_title(
        f"Roman de la Rose — BnF fr. 25526\nYOLO-gen 11x-OBB  |  conf ≥ {conf}",
        fontsize=13, pad=10, fontweight="bold"
    )

    # Récupérer les classes disponibles
    names = results[0].names if results else {}

    legend_patches = {}
    n_total = 0

    for result in results:
        if result.obb is None or len(result.obb) == 0:
            continue
        for box in result.obb:
            cls_id   = int(box.cls[0])
            conf_val = float(box.conf[0])
            pts      = box.xyxyxyxy[0].cpu().numpy().reshape(4, 2)
            cls_name = names.get(cls_id, str(cls_id))
            color    = CLASS_COLORS[cls_id % len(CLASS_COLORS)]

            # Polygone rempli (semi-transparent)
            poly_fill = Polygon(pts, closed=True,
                                linewidth=0, facecolor=color, alpha=0.15)
            # Polygone contour
            poly_edge = Polygon(pts, closed=True,
                                linewidth=1.8, edgecolor=color,
                                facecolor="none", alpha=0.9)
            ax.add_patch(poly_fill)
            ax.add_patch(poly_edge)

            # Étiquette au coin supérieur du polygone
            top_idx = pts[:, 1].argmin()
            lx, ly  = pts[top_idx]
            ax.text(lx, ly - 4, f"{cls_name} {conf_val:.2f}",
                    fontsize=6.5, color="white",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=color,
                              alpha=0.85, linewidth=0))

            legend_patches[cls_name] = Patch(color=color, label=cls_name)
            n_total += 1

    if n_total == 0:
        ax.text(0.5, 0.5, "Aucune détection", transform=ax.transAxes,
                fontsize=18, ha="center", va="center", color="gray")
    else:
        ax.legend(handles=list(legend_patches.values()),
                  loc="lower right", fontsize=9, framealpha=0.85)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"🖼️   Visualisation sauvegardée → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. report_and_export
# ─────────────────────────────────────────────────────────────────────────────
def report_and_export(results, conf: float, save_path: Path) -> None:
    """Affiche un rapport groupé par classe et exporte en JSON."""
    names = results[0].names if results else {}

    by_class = defaultdict(list)
    regions  = []

    for result in results:
        if result.obb is None:
            continue
        for box in result.obb:
            cls_id   = int(box.cls[0])
            conf_val = float(box.conf[0])
            pts      = box.xyxyxyxy[0].cpu().numpy().reshape(4, 2)
            cls_name = names.get(cls_id, str(cls_id))
            by_class[cls_name].append(conf_val)
            regions.append({
                "class":      cls_name,
                "confidence": round(conf_val, 4),
                "polygon":    pts.tolist(),
            })

    # ── Rapport console ────────────────────────────────────────────────────
    print("┌─────────────────────────────────────────────────────────┐")
    print("│                   Rapport de détection                  │")
    print("├──────────────────────┬───────────┬──────────┬───────────┤")
    print("│  Classe              │  Effectif │  Conf.moy│  Conf.max │")
    print("├──────────────────────┼───────────┼──────────┼───────────┤")
    for cls_name, confs in sorted(by_class.items()):
        print(f"│  {cls_name:<20}│  {len(confs):>7}  │  {np.mean(confs):.4f}  │  {max(confs):.4f}   │")
    print("├──────────────────────┴───────────┴──────────┴───────────┤")
    print(f"│  Total : {sum(len(v) for v in by_class.values())} région(s) détectée(s)"
          f"                              │")
    print("└──────────────────────────────────────────────────────────┘")

    # ── Export JSON ────────────────────────────────────────────────────────
    payload = {
        "model":           HF_MODEL_ID,
        "paper":           "arXiv:2506.20326",
        "conf_threshold":  conf,
        "regions":         regions,
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"📄  JSON exporté → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    model         = load_model()
    results, conf = diagnose_thresholds(model, IMAGE_PATH)
    img_pil       = Image.open(IMAGE_PATH).convert("RGB")
    visualize(results, img_pil, conf, OUTPUT_DIR / "yologen_detections.jpg")
    report_and_export(results, conf, OUTPUT_DIR / "yologen_detections.json")


if __name__ == "__main__":
    main()
