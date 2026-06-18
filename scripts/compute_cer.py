"""
scripts/compute_cer.py
-----------------------
Mesure l'impact chiffré de chaque règle de normalisation NLP sur le corpus.

En l'absence de vérité terrain, on utilise une *évaluation relative* :
pour chaque règle, on calcule la distance de Levenshtein entre la version
précédente et la version après application de la règle, normalisée par la
longueur du texte d'entrée.

    distance_relative = Levenshtein(v_avant, v_apres) / max(len(v_avant), 1)

Cela donne une courbe d'évolution du texte sans référence humaine, conforme
aux consignes NLP (section 2 — évaluation relative).

Usage :
    python scripts/compute_cer.py
    python scripts/compute_cer.py --contract output/f279/f279_data_contract.json
    python scripts/compute_cer.py --all          # f12 + f279 agrégés
"""

import argparse
import json
import sys
import unicodedata
from pathlib import Path

import editdistance

# Ajoute la racine du projet au path pour les imports src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nlp.normalize import (
    _normalize_unicode_nfc,
    _normalize_uv,
    _normalize_ij,
    _strip_rubric_tags,
    _normalize_tironian,
    _normalize_p_barre,
    _validate_abbreviations,
    _confirm_uncertain_readings,
    _normalize_column_separator,
)

# ── Définition des étapes dans l'ordre ────────────────────────────────────────

STEPS = [
    ("NFC",                    _normalize_unicode_nfc),
    ("u/v  (u initial -> v)",  _normalize_uv),
    ("i/j  (i initial -> j)",  _normalize_ij),
    ("Tironien -> et",         _normalize_tironian),
    ("P barre -> per/par",     _normalize_p_barre),
    ("Abreviations (mot)",     _validate_abbreviations),
    ("Lectures [mot?]",        _confirm_uncertain_readings),
    ("Rubriques <R>",          _strip_rubric_tags),
    ("Separateur ||",          _normalize_column_separator),
]


# ── Fonctions utilitaires ─────────────────────────────────────────────────────

def _load_processable_lines(contract_path: Path) -> list[str]:
    """Charge les lignes avec needs_review=false d'un data contract."""
    with contract_path.open(encoding="utf-8") as f:
        contract = json.load(f)
    return [
        line["text"]
        for line in contract["lines"]
        if not line.get("needs_review", False)
    ]


def _relative_distance(before: list[str], after: list[str]) -> float:
    """Distance relative entre deux versions (0.0 = identiques)."""
    total_dist = sum(editdistance.eval(b, a) for b, a in zip(before, after))
    total_len  = sum(max(len(b), 1) for b in before)
    return total_dist / total_len if total_len > 0 else 0.0


def compute_per_step_impact(lines: list[str]) -> list[dict]:
    """
    Applique chaque règle séquentiellement et mesure son impact.

    Retourne une liste de dicts avec :
        - name        : nom de la règle
        - corrections : nombre de corrections appliquées
        - dist_rel    : distance relative (v_avant → v_apres)
        - cum_dist    : distance relative cumulée (v_raw → v_courante)
    """
    current = list(lines)
    results = []

    for name, fn in STEPS:
        after     = []
        n_corrs   = 0
        for text in current:
            normalized, corr = fn(text)
            after.append(normalized)
            n_corrs += len(corr)

        dist_rel = _relative_distance(current, after)
        results.append({
            "name":        name,
            "corrections": n_corrs,
            "dist_rel":    dist_rel,
        })
        current = after

    # Calcul de la distance cumulée (raw → après chaque étape)
    running = list(lines)
    for r, (_, fn) in zip(results, STEPS):
        after = [fn(t)[0] for t in running]
        running = after

    # Distance cumulée : raw → final
    cum = _relative_distance(lines, current)
    for r in results:
        pass  # Le cumul global est reporté à la fin

    return results, current, cum


def print_report(results: list[dict], n_lines: int, cum_dist: float,
                 source: str) -> None:
    """Affiche le tableau de résultats et le résumé global."""
    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  Rapport d'impact normalisation NLP -- {source}")
    print(f"  Lignes processables : {n_lines}")
    print(f"{sep}")
    print(f"  {'Etape':<35} {'Corrections':>12} {'Dist. rel.':>12}")
    print(f"  {'-'*35} {'-'*12} {'-'*12}")
    total_corr = 0
    for r in results:
        total_corr += r["corrections"]
        print(f"  {r['name']:<35} {r['corrections']:>12} {r['dist_rel']*100:>11.2f}%")
    print(f"  {'-'*35} {'-'*12} {'-'*12}")
    print(f"  {'TOTAL':35} {total_corr:>12} {cum_dist*100:>11.2f}%")
    print(f"{sep}\n")
    print("  Interpretation :")
    print("  - 'Corrections'  = nb de substitutions appliquees sur le corpus")
    print("  - 'Dist. rel.'   = Levenshtein(v_avant, v_apres) / len(v_avant)")
    print("  - 'TOTAL dist.'  = Levenshtein(brut, normalise) / len(brut)")
    print()


def save_markdown_table(results: list[dict], cum_dist: float, out_path: Path) -> None:
    """Sauvegarde le tableau au format Markdown (pour copier dans README / CONVENTIONS)."""
    lines_md = [
        "| Étape                          | Corrections | Dist. relative |",
        "|-------------------------------|-------------|----------------|",
    ]
    total_corr = 0
    for r in results:
        total_corr += r["corrections"]
        lines_md.append(
            f"| {r['name']:<30} | {r['corrections']:>11} | {r['dist_rel']*100:>12.2f}% |"
        )
    lines_md.append(
        f"| **TOTAL**                      | **{total_corr}**  | **{cum_dist*100:.2f}%**  |"
    )
    out_path.write_text("\n".join(lines_md), encoding="utf-8")
    print(f"  → Tableau Markdown sauvegardé : {out_path}")


# ── Point d'entrée ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mesure l'impact de chaque règle de normalisation NLP."
    )
    parser.add_argument(
        "--contract", type=Path,
        help="Chemin vers un data contract JSON spécifique.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Agrège tous les data contracts dans output/.",
    )
    parser.add_argument(
        "--save-md", type=Path, default=None,
        help="Chemin optionnel pour sauvegarder le tableau Markdown.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent

    if args.contract:
        contract_paths = [args.contract]
    elif args.all:
        contract_paths = sorted(root.glob("output/*/*.json"))
        contract_paths = [p for p in contract_paths if "data_contract" in p.name]
    else:
        # Par défaut : tous les data contracts disponibles
        contract_paths = sorted(root.glob("output/*/*.json"))
        contract_paths = [p for p in contract_paths if "data_contract" in p.name]

    if not contract_paths:
        print("Aucun data contract trouvé. Lancez d'abord run_pipeline.py.")
        sys.exit(1)

    all_lines: list[str] = []
    for p in contract_paths:
        lines = _load_processable_lines(p)
        all_lines.extend(lines)
        print(f"  Chargé : {p.name} ({len(lines)} lignes processables)")

    if not all_lines:
        print("Aucune ligne processable (toutes needs_review=true).")
        sys.exit(0)

    results, _, cum_dist = compute_per_step_impact(all_lines)
    source = "f12 + f279" if len(contract_paths) > 1 else contract_paths[0].stem
    print_report(results, len(all_lines), cum_dist, source)

    if args.save_md:
        save_markdown_table(results, cum_dist, args.save_md)


if __name__ == "__main__":
    main()
