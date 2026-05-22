"""
src/htr/kraken_htr.py
----------------------
Fine-tuning et inférence HTR avec Kraken.

Deux usages principaux :
  - fine_tune()   : fine-tune un modèle existant sur le corpus CREMMA
  - transcribe()  : transcrit les lignes segmentées d'un folio

Usage:
    # Entraînement
    from src.htr.kraken_htr import fine_tune
    fine_tune(train_manifest="data/splits/train.json",
              val_manifest="data/splits/val.json",
              base_model="htrunited/medieval-french",
              output_dir="models/kraken_cremma_v1")

    # Inférence
    from src.htr.kraken_htr import transcribe
    results = transcribe(image_path="data/raw/folio.jpeg",
                          lines=lines,
                          model_path="models/kraken_cremma_v1/best.mlmodel")
"""

import json
from pathlib import Path
from typing import Any

from PIL import Image


# Seuil de confiance en dessous duquel une ligne est marquée needs_review
CONFIDENCE_THRESHOLD = 0.80


def fine_tune(train_manifest: str,
              val_manifest: str,
              base_model: str = "catmus-medieval",
              output_dir: str = "models/kraken_cremma_v1",
              epochs: int = 10,
              batch_size: int = 16,
              learning_rate: float = 1e-4,
              seed: int = 42) -> None:
    """Fine-tune un modèle Kraken sur le corpus CREMMA Medieval.

    Utilise `ketos train` en sous-process ou l'API Python Kraken.
    Le meilleur checkpoint (CER val minimal) est sauvegardé dans output_dir.

    Args:
        train_manifest: Chemin vers le manifest d'entraînement (JSON).
            Format : liste de {"image": "...", "transcription": "..."}.
        val_manifest: Chemin vers le manifest de validation (JSON).
        base_model: Identifiant HuggingFace ou chemin local du modèle de base.
            Recommandé : "catmus-medieval" ou un modèle HTR-United médiéval.
        output_dir: Dossier de sortie pour les checkpoints.
        epochs: Nombre d'époques (défaut 10).
        batch_size: Taille de batch (défaut 16).
        learning_rate: Taux d'apprentissage initial (défaut 1e-4).
        seed: Seed pour la reproductibilité (défaut 42).

    Raises:
        FileNotFoundError: Si train_manifest ou val_manifest est introuvable.
        ImportError: Si kraken n'est pas installé.

    Example:
        >>> fine_tune("data/splits/train.json",
        ...           "data/splits/val.json",
        ...           output_dir="models/v1")
    """
    import subprocess
    import shutil
    import sys
    import os

    for p in [train_manifest, val_manifest]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Manifest introuvable : {p}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Extraire les chemins XML uniques depuis les manifests JSON
    with open(train_manifest, encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_manifest, encoding="utf-8") as f:
        val_data = json.load(f)

    train_xmls = sorted(set(item["xml_path"] for item in train_data))
    val_xmls   = sorted(set(item["xml_path"] for item in val_data))

    # ketos lit ces fichiers avec l'encodage du processus — on force UTF-8 via env
    train_list_file = Path(output_dir) / "train_files.txt"
    val_list_file   = Path(output_dir) / "val_files.txt"
    train_list_file.write_text("\n".join(train_xmls), encoding="utf-8")
    val_list_file.write_text("\n".join(val_xmls),     encoding="utf-8")

    # Résoudre le modèle de base : doit être un chemin local vers un .mlmodel
    resolved_model = None
    if base_model:
        if Path(base_model).exists():
            resolved_model = base_model
        else:
            print(f"⚠️  Modèle '{base_model}' introuvable localement.")
            print("   Fournissez un chemin local vers un .mlmodel, ou téléchargez")
            print("   le modèle CATMuS via : https://huggingface.co/CATMuS")
            print("   Entraînement depuis zéro (sans modèle de base).\n")

    ketos = shutil.which("ketos") or str(Path(sys.executable).parent / "ketos")

    cmd = [
        ketos, "train",
        "-f", "alto",
        "-N", str(epochs),
        "-B", str(batch_size),
        "-r", str(learning_rate),
        "-o", str(Path(output_dir) / "model"),
        "--resize", "add",
        "-t", str(train_list_file),
        "-e", str(val_list_file),
    ]

    if resolved_model:
        cmd += ["-i", resolved_model]

    # PYTHONUTF8=1 : force ketos à lire les chemins accentués en UTF-8
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"

    print(f"🚀  Lancement de l'entraînement Kraken…")
    print(f"   Commande : {' '.join(cmd)}")

    result = subprocess.run(cmd, check=False, env=env)
    if result.returncode != 0:
        print("⚠️  Entraînement terminé avec erreur — vérifiez les logs.")
    else:
        print(f"✅  Modèle sauvegardé dans {output_dir}")

    # Log dans le journal d'expériences
    _log_experiment({
        "type": "training",
        "base_model": base_model,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "output_dir": output_dir,
        "train_manifest": train_manifest,
        "val_manifest": val_manifest,
    })


def transcribe(image_path: str,
               lines: list[dict[str, Any]],
               model_path: str) -> list[dict[str, Any]]:
    """Transcrit les lignes d'un folio avec Kraken.

    Args:
        image_path: Chemin vers l'image source (RGB).
        lines: Sortie de segment_page() — liste de dicts avec "polygon".
        model_path: Chemin vers le modèle .mlmodel fine-tuné.

    Returns:
        Liste de dicts enrichis avec les champs de transcription :
        [
          {
            "line_id": "l_0001",
            "polygon": [...],
            "text": "ce est li romans de la rose",
            "confidence": 0.923,
            "needs_review": false
          },
          ...
        ]

    Raises:
        FileNotFoundError: Si image_path ou model_path est introuvable.

    Example:
        >>> results = transcribe("folio.jpeg", lines, "models/best.mlmodel")
        >>> for r in results:
        ...     print(r["text"], r["confidence"])
    """
    try:
        from kraken import rpred
        from kraken.lib import models as kraken_models
        from kraken.containers import Segmentation, BaselineLine
    except ImportError as e:
        raise ImportError("Kraken n'est pas installé.") from e

    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image introuvable : {image_path}")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")

    img = Image.open(image_path).convert("RGB")
    model = kraken_models.load_any(model_path)

    results = []
    for line in lines:
        # Reconstruction d'un objet Segmentation minimal pour rpred
        seg = Segmentation(
            type="baselines",
            imagename=image_path,
            text_direction="horizontal-lr",
            script_detection=False,
            lines=[BaselineLine(
                id=line["line_id"],
                baseline=line["baseline"],
                boundary=line["polygon"],
                tags={},
            )],
            regions={},
        )

        preds = list(rpred.rpred(network=model,
                                  im=img,
                                  segmentation=seg,
                                  pad=16))

        if preds:
            pred = preds[0]
            text       = pred.prediction
            confidence = float(
                sum(c for _, c in pred.cuts) / len(pred.cuts)
            ) if pred.cuts else 0.0
        else:
            text, confidence = "", 0.0

        results.append({
            **line,
            "text":         text,
            "confidence":   round(confidence, 4),
            "needs_review": confidence < CONFIDENCE_THRESHOLD or len(text) < 2,
        })

    n_review = sum(1 for r in results if r["needs_review"])
    print(f"✅  {len(results)} ligne(s) transcrite(s) — "
          f"{n_review} needs_review ({100*n_review/max(len(results),1):.1f}%)")
    return results


def _log_experiment(params: dict) -> None:
    """Ajoute une entrée au journal d'expériences (JSONL).

    Args:
        params: Dictionnaire des paramètres de l'expérience.
    """
    import datetime
    journal = Path("experiments/journal.jsonl")
    journal.parent.mkdir(exist_ok=True)
    entry = {"timestamp": datetime.datetime.now().isoformat(), **params}
    with open(journal, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
