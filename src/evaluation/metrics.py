"""
src/evaluation/metrics.py
--------------------------
Métriques d'évaluation du pipeline HTR.

Métriques implémentées :
  - CER (Character Error Rate)
  - WER (Word Error Rate)
  - IoU polygones (Intersection over Union)
  - Intervalle de confiance bootstrap sur le CER (N=1000)
  - Test de McNemar pour la comparaison de deux systèmes

Usage:
    from src.evaluation.metrics import compute_cer, compute_wer, bootstrap_cer_ci
    cer = compute_cer(predictions, references)
    ci_low, ci_high = bootstrap_cer_ci(predictions, references)
"""

import random
from typing import Sequence

import numpy as np
import editdistance


def compute_cer(predictions: Sequence[str],
                references: Sequence[str]) -> float:
    """Calcule le Character Error Rate (CER) global.

    CER = somme(distance_Levenshtein(pred, ref)) / somme(len(ref))

    Args:
        predictions: Séquence de transcriptions prédites.
        references: Séquence de transcriptions de référence (ground truth).

    Returns:
        CER entre 0.0 et 1.0+ (peut dépasser 1 si beaucoup d'insertions).

    Raises:
        ValueError: Si predictions et references n'ont pas la même longueur.

    Example:
        >>> compute_cer(["bonjour"], ["bonjour"])
        0.0
        >>> compute_cer(["bnjour"], ["bonjour"])
        0.14285714285714285
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Longueurs différentes : {len(predictions)} vs {len(references)}"
        )
    total_dist = sum(editdistance.eval(p, r) for p, r in zip(predictions, references))
    total_len  = sum(len(r) for r in references)
    return total_dist / total_len if total_len > 0 else 0.0


def compute_wer(predictions: Sequence[str],
                references: Sequence[str]) -> float:
    """Calcule le Word Error Rate (WER) global.

    WER = somme(distance_Levenshtein_mots(pred, ref)) / somme(len_mots(ref))

    Args:
        predictions: Séquence de transcriptions prédites.
        references: Séquence de transcriptions de référence.

    Returns:
        WER entre 0.0 et 1.0+.

    Example:
        >>> compute_wer(["le roman de la rose"], ["le romans de la rose"])
        0.2
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Longueurs différentes : {len(predictions)} vs {len(references)}"
        )
    total_dist = sum(
        editdistance.eval(p.split(), r.split())
        for p, r in zip(predictions, references)
    )
    total_len = sum(len(r.split()) for r in references)
    return total_dist / total_len if total_len > 0 else 0.0


def bootstrap_cer_ci(predictions: Sequence[str],
                      references: Sequence[str],
                      n_bootstrap: int = 1000,
                      alpha: float = 0.05,
                      seed: int = 42) -> tuple[float, float]:
    """Calcule l'intervalle de confiance bootstrap du CER.

    Ré-échantillonne n_bootstrap fois avec remise et calcule le CER
    sur chaque échantillon. Retourne le percentile alpha/2 et 1-alpha/2.

    Args:
        predictions: Transcriptions prédites.
        references: Transcriptions de référence.
        n_bootstrap: Nombre de ré-échantillonnages (défaut 1000).
        alpha: Niveau d'erreur (défaut 0.05 → IC à 95%).
        seed: Seed pour la reproductibilité.

    Returns:
        Tuple (cer_low, cer_high) — bornes de l'intervalle à 95%.

    Example:
        >>> low, high = bootstrap_cer_ci(preds, refs)
        >>> print(f"CER IC 95% : [{low:.3f}, {high:.3f}]")
    """
    random.seed(seed)
    np.random.seed(seed)

    n = len(predictions)
    cer_samples = []

    for _ in range(n_bootstrap):
        indices = np.random.randint(0, n, size=n)
        sample_preds = [predictions[i] for i in indices]
        sample_refs  = [references[i]  for i in indices]
        cer_samples.append(compute_cer(sample_preds, sample_refs))

    cer_samples = np.array(cer_samples)
    low  = float(np.percentile(cer_samples, 100 * alpha / 2))
    high = float(np.percentile(cer_samples, 100 * (1 - alpha / 2)))
    return low, high


def compute_iou_polygon(poly_pred: list[list[float]],
                         poly_ref:  list[list[float]]) -> float:
    """Calcule l'IoU entre deux polygones (approximation par bbox).

    Pour une IoU exacte polygone-à-polygone, utiliser shapely.
    Cette version utilise les bounding boxes pour éviter une dépendance.

    Args:
        poly_pred: Polygone prédit, liste de [x, y].
        poly_ref:  Polygone de référence, liste de [x, y].

    Returns:
        IoU entre 0.0 et 1.0.

    Example:
        >>> compute_iou_polygon([[0,0],[10,0],[10,10],[0,10]],
        ...                      [[0,0],[10,0],[10,10],[0,10]])
        1.0
    """
    def bbox(poly):
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        return min(xs), min(ys), max(xs), max(ys)

    x1, y1, x2, y2 = bbox(poly_pred)
    x3, y3, x4, y4 = bbox(poly_ref)

    inter_x1 = max(x1, x3)
    inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4)
    inter_y2 = min(y2, y4)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_pred  = (x2 - x1) * (y2 - y1)
    area_ref   = (x4 - x3) * (y4 - y3)
    union_area = area_pred + area_ref - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def compute_mean_iou(pred_lines: list[dict],
                      ref_lines:  list[dict]) -> float:
    """Calcule l'IoU moyen entre lignes prédites et de référence.

    Appariement greedy par ordre d'index (ligne i prédit ↔ ligne i référence).

    Args:
        pred_lines: Lignes prédites (dicts avec clé "polygon").
        ref_lines:  Lignes de référence (dicts avec clé "polygon").

    Returns:
        IoU moyen sur les paires appariées.
    """
    n = min(len(pred_lines), len(ref_lines))
    if n == 0:
        return 0.0
    ious = [
        compute_iou_polygon(pred_lines[i]["polygon"], ref_lines[i]["polygon"])
        for i in range(n)
    ]
    return float(np.mean(ious))


def mcnemar_test(errors_a: list[bool],
                 errors_b: list[bool]) -> tuple[float, float]:
    """Test de McNemar pour comparer deux systèmes HTR.

    Compare les profils d'erreurs de deux systèmes sur les mêmes exemples.
    Utile pour comparer Kraken vs TrOCR (bonus +1 point).

    Args:
        errors_a: Liste de booléens — True si le système A fait une erreur.
        errors_b: Liste de booléens — True si le système B fait une erreur.

    Returns:
        Tuple (statistique_chi2, p_value).

    Example:
        >>> chi2, p = mcnemar_test(errors_kraken, errors_trocr)
        >>> print(f"McNemar p={p:.4f}")
    """
    from scipy.stats import chi2 as chi2_dist  # type: ignore

    b = sum(1 for a, b_ in zip(errors_a, errors_b) if a and not b_)
    c = sum(1 for a, b_ in zip(errors_a, errors_b) if not a and b_)

    if b + c == 0:
        return 0.0, 1.0

    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)  # correction de continuité
    p_value   = 1 - chi2_dist.cdf(chi2_stat, df=1)
    return float(chi2_stat), float(p_value)


def full_evaluation_report(predictions: Sequence[str],
                             references:  Sequence[str],
                             pred_lines:  list[dict] | None = None,
                             ref_lines:   list[dict] | None = None) -> dict:
    """Produit le rapport d'évaluation complet.

    Args:
        predictions: Transcriptions prédites.
        references:  Transcriptions de référence.
        pred_lines:  Lignes segmentées prédites (pour IoU, optionnel).
        ref_lines:   Lignes de référence (pour IoU, optionnel).

    Returns:
        Dict avec CER, WER, IC bootstrap, IoU moyen.
    """
    cer = compute_cer(predictions, references)
    wer = compute_wer(predictions, references)
    ci_low, ci_high = bootstrap_cer_ci(predictions, references)

    report = {
        "CER":      round(cer, 4),
        "WER":      round(wer, 4),
        "CER_IC95": [round(ci_low, 4), round(ci_high, 4)],
        "n_lines":  len(predictions),
    }

    if pred_lines and ref_lines:
        report["mean_IoU"] = round(compute_mean_iou(pred_lines, ref_lines), 4)

    # Affichage console
    print("\n┌─────────────────────────────────────────┐")
    print("│         Rapport d'évaluation HTR        │")
    print("├─────────────────┬───────────────────────┤")
    print(f"│  CER            │  {cer*100:>6.2f} %              │")
    print(f"│  WER            │  {wer*100:>6.2f} %              │")
    print(f"│  IC 95% CER     │  [{ci_low*100:.2f}%, {ci_high*100:.2f}%]       │")
    if "mean_IoU" in report:
        print(f"│  IoU moyen      │  {report['mean_IoU']:>6.4f}               │")
    print("└─────────────────┴───────────────────────┘")

    seuil_val  = cer < 0.15
    seuil_exc  = cer < 0.08
    print(f"   Seuil validation (CER < 15%) : {'✅' if seuil_val else '❌'}")
    print(f"   Seuil excellence  (CER < 8%) : {'✅' if seuil_exc else '❌'}")

    return report
