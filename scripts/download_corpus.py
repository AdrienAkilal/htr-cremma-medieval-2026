"""
scripts/download_corpus.py
---------------------------
Télécharge le corpus CREMMA-MSS-17 depuis GitHub et parse les fichiers
ALTO XML pour constituer le manifest d'entraînement.

Structure du repo source :
  data/
    recueil-lettres-pieces/          *.xml (ALTO v4) + images
    dépêches-originales-*/           *.xml + images
    lettres-de-bossuet/              *.xml + images
    correspondance-dom-bernard-*/    *.xml + images
    pensées-sur-la-religion-*/       *.xml + images

Usage:
    python scripts/download_corpus.py
    python scripts/download_corpus.py --output data/raw --no-clone
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

REPO_URL   = "https://github.com/HTR-United/CREMMA-MSS-17.git"
REPO_NAME  = "CREMMA-MSS-17"
LICENSE    = "CC-BY 4.0"
SOURCE     = "HTR-United/CREMMA-MSS-17"

# Namespaces ALTO v4
ALTO_NS = {
    "alto4": "http://www.loc.gov/standards/alto/ns-v4#",
    "alto":  "http://schema.ccs-gmbh.com/ALTO",        # v2 fallback
}


# ─────────────────────────────────────────────────────────────────────────────
# Clonage
# ─────────────────────────────────────────────────────────────────────────────

def clone_repo(output_dir: str = "data/raw") -> Path:
    """Clone le repo CREMMA-MSS-17 dans output_dir.

    Args:
        output_dir: Dossier parent de destination.

    Returns:
        Chemin vers le repo cloné.
    """
    dest = Path(output_dir) / REPO_NAME
    if dest.exists():
        print(f"📁  Repo déjà présent : {dest}")
        print("   Pour mettre à jour : git -C {dest} pull")
        return dest

    print(f"📥  Clonage de {REPO_URL} …")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(dest)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"❌  Erreur git clone :\n{result.stderr}")
        sys.exit(1)

    print(f"✅  Repo cloné dans {dest}")
    return dest


# ─────────────────────────────────────────────────────────────────────────────
# Parser ALTO v4
# ─────────────────────────────────────────────────────────────────────────────

def _find_with_ns(element, tag: str, ns_dict: dict):
    """Cherche un tag en essayant tous les namespaces connus."""
    for prefix, uri in ns_dict.items():
        result = element.find(f"{{{uri}}}{tag}")
        if result is not None:
            return result
    return None


def _findall_with_ns(element, tag: str, ns_dict: dict):
    """findall multi-namespace."""
    results = []
    for prefix, uri in ns_dict.items():
        results.extend(element.findall(f".//{{{uri}}}{tag}"))
    # Dédoublonnage par id
    seen = set()
    unique = []
    for r in results:
        rid = id(r)
        if rid not in seen:
            seen.add(rid)
            unique.append(r)
    return unique


def parse_alto_xml(xml_path: Path) -> list[dict]:
    """Parse un fichier ALTO XML v4 et extrait les lignes de texte.

    Chaque ligne produit un dict avec :
      - line_id   : identifiant ALTO (attribut ID)
      - text      : transcription complète de la ligne
      - polygon   : coordonnées du bloc (HPOS, VPOS, WIDTH, HEIGHT → 4 coins)
      - baseline  : liste vide (ALTO ne stocke pas les baselines)
      - xml_file  : chemin source

    Args:
        xml_path: Chemin vers le fichier .xml ALTO.

    Returns:
        Liste de dicts, un par ligne de texte non vide.
    """
    try:
        tree = ET.parse(str(xml_path))
    except ET.ParseError as e:
        print(f"   ⚠️  XML invalide ({xml_path.name}) : {e}")
        return []

    root = tree.getroot()
    lines = []

    text_lines = _findall_with_ns(root, "TextLine", ALTO_NS)

    for tl in text_lines:
        # Récupère le texte de la ligne (STRING CONTENT ou TextEquiv)
        text_parts = []

        # ALTO v4 : <String CONTENT="..."/>
        strings = _findall_with_ns(tl, "String", ALTO_NS)
        for s in strings:
            content = s.get("CONTENT", "")
            if content:
                text_parts.append(content)

        # Fallback : TextEquiv/Unicode (format mixte)
        if not text_parts:
            te = _find_with_ns(tl, "TextEquiv", ALTO_NS)
            if te is not None:
                uni = _find_with_ns(te, "Unicode", ALTO_NS)
                if uni is not None and uni.text:
                    text_parts.append(uni.text.strip())

        text = " ".join(text_parts).strip()
        if not text:
            continue

        # Coordonnées bounding box
        hpos   = int(tl.get("HPOS",   0))
        vpos   = int(tl.get("VPOS",   0))
        width  = int(tl.get("WIDTH",  0))
        height = int(tl.get("HEIGHT", 0))

        polygon = [
            [hpos,          vpos],
            [hpos + width,  vpos],
            [hpos + width,  vpos + height],
            [hpos,          vpos + height],
        ]

        line_id = tl.get("ID", f"l_{len(lines)+1:04d}")

        lines.append({
            "line_id":  line_id,
            "text":     text,
            "polygon":  polygon,
            "baseline": [],
            "xml_file": str(xml_path),
        })

    return lines


def find_image_for_xml(xml_path: Path) -> Path | None:
    """Cherche l'image correspondant à un fichier ALTO XML.

    ALTO stocke le nom de l'image dans <sourceImageInformation><fileName>.
    Fallback : même nom de fichier avec extension image.

    Args:
        xml_path: Chemin vers le fichier .xml.

    Returns:
        Chemin vers l'image si trouvée, None sinon.
    """
    # 1) Lire le nom dans le XML
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
        for ns_uri in ALTO_NS.values():
            el = root.find(
                f".//{{{ns_uri}}}sourceImageInformation/{{{ns_uri}}}fileName"
            )
            if el is not None and el.text:
                img_name = Path(el.text).name
                candidate = xml_path.parent / img_name
                if candidate.exists():
                    return candidate
    except Exception:
        pass

    # 2) Fallback : même stem avec extension courante
    for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
        candidate = xml_path.with_suffix(ext)
        if candidate.exists():
            return candidate

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Constitution du manifest
# ─────────────────────────────────────────────────────────────────────────────

def build_manifest(repo_path: Path,
                   output_dir: str = "data/raw") -> list[dict]:
    """Parcourt tous les sous-corpus et constitue le manifest.

    Args:
        repo_path: Chemin vers le repo CREMMA-MSS-17 cloné.
        output_dir: Dossier où sauvegarder le manifest JSON.

    Returns:
        Liste de dicts — un par ligne de texte valide.
    """
    data_dir = repo_path / "data"
    if not data_dir.exists():
        print(f"❌  Dossier data/ introuvable dans {repo_path}")
        sys.exit(1)

    xml_files = sorted(data_dir.rglob("*.xml"))
    print(f"\n🔍  {len(xml_files)} fichiers XML trouvés dans {data_dir}")

    records  = []
    n_images_missing = 0

    for xml_path in xml_files:
        # Ignorer les fichiers de config/schéma
        if xml_path.name.startswith("_") or "schema" in xml_path.name.lower():
            continue

        subcorpus = xml_path.parent.name

        # Parser le XML
        lines = parse_alto_xml(xml_path)
        if not lines:
            continue

        # Chercher l'image associée
        img_path = find_image_for_xml(xml_path)
        if img_path is None:
            n_images_missing += 1
            img_path_str = ""   # pas d'image → on garde quand même la transcription
        else:
            img_path_str = str(img_path)

        # SHA-256 du fichier XML
        sha = hashlib.sha256(xml_path.read_bytes()).hexdigest()

        for line in lines:
            records.append({
                "line_id":          line["line_id"],
                "text":             line["text"],
                "polygon":          line["polygon"],
                "baseline":         line["baseline"],
                "image_path":       img_path_str,
                "xml_path":         str(xml_path),
                "subcorpus":        subcorpus,
                "source":           SOURCE,
                "license":          LICENSE,
                "sha256_xml":       sha,
            })

    print(f"✅  {len(records)} lignes extraites")
    print(f"   Sous-corpus : {sorted({r['subcorpus'] for r in records})}")
    if n_images_missing:
        print(f"   ⚠️  {n_images_missing} fichiers XML sans image associée")

    # Sauvegarde du manifest
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "cremma_mss17_manifest.json"
    manifest_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"📄  Manifest → {manifest_path}")

    # Statistiques par sous-corpus
    _print_stats(records)

    return records


def _print_stats(records: list[dict]) -> None:
    """Affiche les statistiques du corpus par sous-corpus."""
    from collections import Counter
    counts = Counter(r["subcorpus"] for r in records)
    print("\n┌──────────────────────────────────────────────────┬──────────┐")
    print("│  Sous-corpus                                     │  Lignes  │")
    print("├──────────────────────────────────────────────────┼──────────┤")
    for sub, n in sorted(counts.items()):
        print(f"│  {sub:<48}│  {n:>6}  │")
    print("├──────────────────────────────────────────────────┼──────────┤")
    print(f"│  {'TOTAL':<48}│  {sum(counts.values()):>6}  │")
    print("└──────────────────────────────────────────────────┴──────────┘")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Téléchargement et parsing du corpus CREMMA-MSS-17"
    )
    p.add_argument("--output",   default="data/raw",
                   help="Dossier de sortie (défaut: data/raw)")
    p.add_argument("--no-clone", action="store_true",
                   help="Ne pas cloner (repo déjà présent dans --output)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.no_clone:
        repo_path = Path(args.output) / REPO_NAME
        if not repo_path.exists():
            print(f"❌  Repo introuvable : {repo_path}")
            print("   Relancez sans --no-clone pour le télécharger.")
            sys.exit(1)
    else:
        repo_path = clone_repo(args.output)

    build_manifest(repo_path, args.output)
