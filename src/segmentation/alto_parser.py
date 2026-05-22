"""
src/segmentation/alto_parser.py
---------------------------------
Parser pour les fichiers ALTO XML v4 produits par eScriptorium / Kraken.

Utilisé pour lire le ground truth de CREMMA-MSS-17 et pour exporter
les prédictions du pipeline en format compatible.

Format ALTO v4 attendu :
  <alto>
    <Layout>
      <Page>
        <PrintSpace>
          <TextBlock>
            <TextLine ID="l_001" HPOS="10" VPOS="20" WIDTH="500" HEIGHT="30">
              <String CONTENT="ce est li romans de la rose" HPOS="10" .../>
            </TextLine>
          </TextBlock>
        </PrintSpace>
      </Page>
    </Layout>
  </alto>

Usage:
    from src.segmentation.alto_parser import parse_alto, alto_to_kraken_manifest
    lines = parse_alto("folio.xml")
    manifest = alto_to_kraken_manifest("folio.xml", "folio.jpg")
"""

from pathlib import Path
from xml.etree import ElementTree as ET
from typing import Any


# Namespaces ALTO (v4 officiel + variantes produites par eScriptorium)
ALTO_NAMESPACES = [
    "http://www.loc.gov/standards/alto/ns-v4#",      # v4 officiel
    "http://www.loc.gov/standards/alto/ns-v4",       # v4 sans #
    "http://schema.ccs-gmbh.com/ALTO",               # v2
    "http://www.loc.gov/standards/alto/alto-v4.0.xsd",
    "",                                               # sans namespace
]


def _find(element, tag: str):
    """Cherche un tag en testant tous les namespaces ALTO connus.

    Args:
        element: Élément XML parent.
        tag: Nom du tag sans namespace.

    Returns:
        Premier élément trouvé, ou None.
    """
    for ns in ALTO_NAMESPACES:
        full = f"{{{ns}}}{tag}" if ns else tag
        result = element.find(full)
        if result is not None:
            return result
    return None


def _findall(element, tag: str) -> list:
    """findall multi-namespace (dédoublonné).

    Args:
        element: Élément XML parent.
        tag: Nom du tag sans namespace (recherche récursive .//).

    Returns:
        Liste d'éléments trouvés (sans doublons).
    """
    seen, results = set(), []
    for ns in ALTO_NAMESPACES:
        full = f".//{{{ns}}}{tag}" if ns else f".//{tag}"
        for el in element.findall(full):
            if id(el) not in seen:
                seen.add(id(el))
                results.append(el)
    return results


def parse_alto(xml_path: str) -> list[dict[str, Any]]:
    """Parse un fichier ALTO XML et retourne les lignes de texte.

    Args:
        xml_path: Chemin vers le fichier .xml ALTO.

    Returns:
        Liste de dicts par ligne de texte :
        [
          {
            "line_id":  "l_0001",
            "text":     "ce est li romans de la rose",
            "polygon":  [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],
            "baseline": [],
            "hpos": 10, "vpos": 20, "width": 500, "height": 30,
            "confidence": 1.0,
          },
          ...
        ]

    Raises:
        FileNotFoundError: Si xml_path n'existe pas.

    Example:
        >>> lines = parse_alto("data/raw/CREMMA-MSS-17/data/lettres-de-bossuet/page001.xml")
        >>> print(lines[0]["text"])
    """
    path = Path(xml_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier ALTO introuvable : {xml_path}")

    try:
        tree = ET.parse(str(path))
    except ET.ParseError as e:
        print(f"⚠️  XML invalide ({path.name}) : {e}")
        return []

    root = tree.getroot()
    lines = []

    for tl in _findall(root, "TextLine"):
        # ── Texte ────────────────────────────────────────────────────────────
        text_parts = []

        # Méthode 1 : <String CONTENT="..."/> (ALTO standard)
        for s in _findall(tl, "String"):
            content = s.get("CONTENT", "").strip()
            if content:
                text_parts.append(content)
            # Espaces entre mots : <SP/>
            # (ignorés ici — on joint avec espace)

        # Méthode 2 : <TextEquiv><Unicode>...</Unicode></TextEquiv>
        if not text_parts:
            te = _find(tl, "TextEquiv")
            if te is not None:
                uni = _find(te, "Unicode")
                if uni is not None and uni.text:
                    text_parts.append(uni.text.strip())

        text = " ".join(text_parts).strip()
        if not text:
            continue

        # ── Coordonnées ──────────────────────────────────────────────────────
        hpos   = int(float(tl.get("HPOS",   0)))
        vpos   = int(float(tl.get("VPOS",   0)))
        width  = int(float(tl.get("WIDTH",  0)))
        height = int(float(tl.get("HEIGHT", 0)))

        polygon = [
            [hpos,          vpos],
            [hpos + width,  vpos],
            [hpos + width,  vpos + height],
            [hpos,          vpos + height],
        ]

        # ── Confiance ────────────────────────────────────────────────────────
        # Certains fichiers ALTO stockent une confiance
        conf = float(tl.get("CC", 1.0))
        if conf > 1.0:
            conf = conf / 100.0  # normalisation si en %

        line_id = tl.get("ID", f"l_{len(lines)+1:04d}")

        lines.append({
            "line_id":    line_id,
            "text":       text,
            "polygon":    polygon,
            "baseline":   [],
            "hpos":       hpos,
            "vpos":       vpos,
            "width":      width,
            "height":     height,
            "confidence": round(conf, 4),
        })

    return lines


def get_image_filename(xml_path: str) -> str | None:
    """Lit le nom de l'image source dans un fichier ALTO.

    Args:
        xml_path: Chemin vers le fichier .xml ALTO.

    Returns:
        Nom de fichier image (str) ou None si absent.

    Example:
        >>> get_image_filename("page001.xml")
        'page001.jpg'
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        el = None
        for ns in ALTO_NAMESPACES:
            tag = f"{{{ns}}}fileName" if ns else "fileName"
            el = root.find(f".//{tag}")
            if el is not None and el.text:
                return Path(el.text).name
    except Exception:
        pass
    return None


def alto_to_kraken_manifest(xml_path: str,
                              image_path: str) -> list[dict[str, Any]]:
    """Convertit un fichier ALTO en manifest Kraken (paires image/texte).

    Kraken attend pour l'entraînement une liste de paires
    {"image": chemin_image_cropée, "text": transcription}.
    Cette fonction retourne les lignes avec leurs coordonnées pour
    permettre le crop en amont.

    Args:
        xml_path:   Chemin vers le fichier .xml ALTO.
        image_path: Chemin vers l'image correspondante.

    Returns:
        Liste de dicts prêts pour l'entraînement Kraken :
        [{"image_path": ..., "image_crop": [x,y,w,h], "text": ..., "line_id": ...}, ...]

    Example:
        >>> manifest = alto_to_kraken_manifest("page001.xml", "page001.jpg")
    """
    lines = parse_alto(xml_path)
    result = []
    for line in lines:
        result.append({
            "image_path":  image_path,
            "image_crop":  [line["hpos"], line["vpos"],
                            line["width"], line["height"]],
            "line_id":     line["line_id"],
            "text":        line["text"],
            "polygon":     line["polygon"],
        })
    return result


def count_lines_in_corpus(data_dir: str) -> dict[str, int]:
    """Compte les lignes par sous-corpus dans un dossier de fichiers ALTO.

    Args:
        data_dir: Chemin vers le dossier data/ du corpus.

    Returns:
        Dict {nom_sous_corpus: nombre_de_lignes}.

    Example:
        >>> counts = count_lines_in_corpus("data/raw/CREMMA-MSS-17/data")
        >>> print(counts)
    """
    from collections import defaultdict
    counts = defaultdict(int)
    for xml_file in Path(data_dir).rglob("*.xml"):
        if xml_file.name.startswith("_"):
            continue
        subcorpus = xml_file.parent.name
        lines = parse_alto(str(xml_file))
        counts[subcorpus] += len(lines)
    return dict(counts)
