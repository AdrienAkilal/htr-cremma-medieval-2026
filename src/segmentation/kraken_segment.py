"""
src/segmentation/kraken_segment.py
------------------------------------
Segmentation des lignes de texte avec Kraken BLLA.

Produit pour chaque image :
  - Les lignes de base (baselines)
  - Les polygones de chaque ligne
  - Un fichier PAGE XML exporté dans segmentations/

Usage:
    from src.segmentation.kraken_segment import segment_page
    lines = segment_page("data/raw/folio.jpeg",
                          xml_out="segmentations/folio.page.xml")
"""

import json
from pathlib import Path
from typing import Any

from PIL import Image


def segment_page(image_path: str,
                 xml_out: str | None = None,
                 model_name: str = "blla.mlmodel") -> list[dict[str, Any]]:
    """Segmente les lignes d'un folio avec Kraken BLLA.

    Args:
        image_path: Chemin vers l'image prétraitée.
        xml_out: Chemin de sortie du fichier PAGE XML (optionnel).
        model_name: Nom ou chemin du modèle de segmentation Kraken.
            Utiliser un modèle médiéval si disponible (HTR-United).

    Returns:
        Liste de dicts, un par ligne détectée :
        [
          {
            "line_id": "l_001",
            "baseline": [[x1,y1], [x2,y2], ...],
            "polygon":  [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
            "bbox":     [x_min, y_min, x_max, y_max]
          },
          ...
        ]

    Raises:
        FileNotFoundError: Si image_path n'existe pas.
        ImportError: Si kraken n'est pas installé.

    Example:
        >>> lines = segment_page("data/raw/folio.jpeg",
        ...                       xml_out="segmentations/folio.page.xml")
        >>> print(f"{len(lines)} lignes détectées")
    """
    try:
        from kraken import blla
        from kraken.lib import models as kraken_models
        from kraken.serialization import serialize
    except ImportError as e:
        raise ImportError(
            "Kraken n'est pas installé. Lancez : pip install kraken"
        ) from e

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image introuvable : {image_path}")

    img = Image.open(str(path)).convert("RGB")

    # Chargement du modèle BLLA
    # Si model_name est un chemin local, on l'utilise directement
    # Sinon Kraken télécharge le modèle par défaut
    if Path(model_name).exists():
        model = kraken_models.load_any(model_name)
    else:
        model = blla.load_default_model()

    # Segmentation BLLA
    baseline_seg = blla.segment(img, model=model)

    # Conversion en format interne
    lines = []
    for idx, line in enumerate(baseline_seg.lines):
        line_id = f"l_{idx + 1:04d}"

        baseline = [[int(p[0]), int(p[1])] for p in line.baseline]
        polygon  = [[int(p[0]), int(p[1])] for p in line.boundary]

        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        bbox = [min(xs), min(ys), max(xs), max(ys)]

        lines.append({
            "line_id":  line_id,
            "baseline": baseline,
            "polygon":  polygon,
            "bbox":     bbox,
        })

    # Export PAGE XML
    if xml_out:
        out_path = Path(xml_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        page_xml = serialize(baseline_seg,
                             image_name=path.name,
                             image_size=img.size,
                             template="pagexml")
        out_path.write_text(page_xml, encoding="utf-8")
        print(f"📄  PAGE XML exporté → {out_path}")

    print(f"✅  {len(lines)} ligne(s) segmentée(s) dans {path.name}")
    return lines


def validate_polygons(lines: list[dict],
                      image_path: str) -> list[dict]:
    """Valide que les polygones sont dans les limites de l'image.

    Supprime les polygones vides ou hors-image.
    Signale les lignes trop courtes (< 5 pixels).

    Args:
        lines: Sortie de segment_page().
        image_path: Chemin vers l'image source (pour récupérer les dimensions).

    Returns:
        Liste filtrée de lignes valides.
    """
    img = Image.open(image_path)
    W, H = img.size

    valid = []
    for line in lines:
        poly = line["polygon"]
        if len(poly) < 3:
            continue
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        if max(xs) > W or max(ys) > H:
            continue
        width = max(xs) - min(xs)
        if width < 5:
            continue
        valid.append(line)

    removed = len(lines) - len(valid)
    if removed:
        print(f"⚠️  {removed} polygone(s) invalide(s) supprimé(s)")
    return valid
