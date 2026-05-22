"""
src/aggregation/aggregate.py
------------------------------
Agrégation des transcriptions en data contract JSON et PAGE XML.

Le data contract est le livrable pour le module NLP (Volet 2).
Chaque ligne transcrite comporte : texte, confiance, polygone, flag needs_review.

Usage:
    from src.aggregation.aggregate import build_data_contract, export_page_xml
    contract = build_data_contract(
        image_path="data/raw/folio.jpeg",
        transcriptions=results,   # sortie de transcribe()
        conf_threshold=0.10,
    )
    export_page_xml(contract, "segmentations/folio.page.xml")
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

# Schéma minimal attendu pour chaque région du data contract
REQUIRED_KEYS = {"line_id", "text", "confidence", "polygon", "needs_review"}


def build_data_contract(image_path: str,
                         transcriptions: list[dict[str, Any]],
                         conf_threshold: float = 0.10,
                         layout_regions: list[dict] | None = None) -> dict:
    """Construit le data contract JSON pour un folio.

    Args:
        image_path: Chemin vers l'image source.
        transcriptions: Sortie de transcribe() — liste de dicts par ligne.
        conf_threshold: Seuil de confiance utilisé pour l'HTR.
        layout_regions: Régions de layout YOLO-gen (optionnel, pour métadonnées).

    Returns:
        Dict conforme au data contract :
        {
          "image": "folio.jpeg",
          "sha256": "abc123...",
          "date": "2026-05-21T...",
          "conf_threshold": 0.10,
          "coordinate_system": {"origin": "top-left", "unit": "pixels"},
          "lines": [
            {
              "line_id": "l_0001",
              "text": "ce est li romans de la rose",
              "confidence": 0.923,
              "needs_review": false,
              "polygon": [[x1,y1], ...]
            },
            ...
          ],
          "stats": {
            "n_lines": 42,
            "n_needs_review": 3,
            "needs_review_rate": 0.071,
            "mean_confidence": 0.887
          }
        }

    Raises:
        ValueError: Si une transcription manque un champ requis.

    Example:
        >>> contract = build_data_contract("folio.jpeg", transcriptions)
        >>> print(contract["stats"])
    """
    # Validation du schéma
    for i, t in enumerate(transcriptions):
        missing = REQUIRED_KEYS - set(t.keys())
        if missing:
            raise ValueError(
                f"Transcription [{i}] manque les champs : {missing}"
            )

    # SHA-256 de l'image source
    sha256 = _sha256_file(image_path)

    # Statistiques
    n = len(transcriptions)
    n_review = sum(1 for t in transcriptions if t["needs_review"])
    mean_conf = (
        sum(t["confidence"] for t in transcriptions) / n if n else 0.0
    )

    contract = {
        "image":        Path(image_path).name,
        "image_path":   str(image_path),
        "sha256":       sha256,
        "date":         datetime.now().isoformat(),
        "model":        "kraken-cremma-medieval",
        "conf_threshold": conf_threshold,
        "coordinate_system": {
            "origin": "top-left",
            "unit":   "pixels",
        },
        "lines": [
            {
                "line_id":      t["line_id"],
                "text":         t["text"],
                "confidence":   t["confidence"],
                "needs_review": t["needs_review"],
                "polygon":      t["polygon"],
                "baseline":     t.get("baseline", []),
            }
            for t in transcriptions
        ],
        "layout_regions": layout_regions or [],
        "stats": {
            "n_lines":          n,
            "n_needs_review":   n_review,
            "needs_review_rate": round(n_review / n, 4) if n else 0.0,
            "mean_confidence":  round(mean_conf, 4),
        },
    }
    return contract


def save_data_contract(contract: dict, output_path: str) -> None:
    """Sauvegarde le data contract en JSON.

    Args:
        contract: Dict retourné par build_data_contract().
        output_path: Chemin de sortie (.json).

    Example:
        >>> save_data_contract(contract, "dataset_nlp/output.json")
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(contract, f, ensure_ascii=False, indent=2)
    print(f"📄  Data contract sauvegardé → {out}")


def export_page_xml(contract: dict, output_path: str) -> None:
    """Exporte le data contract au format PAGE XML.

    Conforme au schéma PAGE XML 2019 (http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15).
    Compatible avec eScriptorium et Kraken.

    Args:
        contract: Dict retourné par build_data_contract().
        output_path: Chemin de sortie (.page.xml).

    Example:
        >>> export_page_xml(contract, "segmentations/folio.page.xml")
    """
    lines_xml = ""
    for i, line in enumerate(contract["lines"]):
        poly_str = " ".join(
            f"{int(p[0])},{int(p[1])}" for p in line["polygon"]
        )
        baseline_str = " ".join(
            f"{int(p[0])},{int(p[1])}" for p in line.get("baseline", [])
        ) if line.get("baseline") else ""

        needs_review_attr = ' custom="needs_review"' if line["needs_review"] else ""
        baseline_el = (
            f'<Baseline points="{baseline_str}"/>' if baseline_str else ""
        )

        lines_xml += f"""
    <TextLine id="{line['line_id']}" conf="{line['confidence']}"{needs_review_attr}>
      <Coords points="{poly_str}"/>
      {baseline_el}
      <TextEquiv conf="{line['confidence']}">
        <Unicode>{_escape_xml(line['text'])}</Unicode>
      </TextEquiv>
    </TextLine>"""

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <Metadata>
    <Creator>htr-cremma-medieval-2026</Creator>
    <Created>{contract['date']}</Created>
    <LastChange>{contract['date']}</LastChange>
  </Metadata>
  <Page imageFilename="{contract['image']}" imageWidth="0" imageHeight="0">
    <TextRegion id="r_main">
      <Coords points="0,0 0,0 0,0 0,0"/>{lines_xml}
    </TextRegion>
  </Page>
</PcGts>
"""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(xml, encoding="utf-8")
    print(f"📄  PAGE XML exporté → {out}")


def _sha256_file(path: str) -> str:
    """Calcule le hash SHA-256 d'un fichier.

    Args:
        path: Chemin vers le fichier.

    Returns:
        Chaîne hexadécimale SHA-256 (64 caractères).
    """
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    except FileNotFoundError:
        return "file_not_found"
    return h.hexdigest()


def _escape_xml(text: str) -> str:
    """Échappe les caractères spéciaux XML.

    Args:
        text: Texte brut.

    Returns:
        Texte avec &, <, > échappés.
    """
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))
