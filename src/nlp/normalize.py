"""
src/nlp/normalize.py
---------------------
Module de normalisation NLP pour les transcriptions médiévales.

Applique les règles définies dans CONVENTIONS_NLP.md (section 2 et 9) :
  - Normalisation Unicode NFC (encodage canonique)
  - Normalisation des allographes u/v et i/j
  - Suppression des balises structurelles <R>...</R>
  - Normalisation de la note tironienne ⁊ → et
  - Normalisation du p barré ꝑ → per/par
  - Validation des développements d'abréviations (mot) → mot
  - Confirmation des lectures incertaines [mot?] → mot
  - Normalisation du séparateur de colonne || → espace

Les marqueurs non résolvables sans contexte ([?], [...], [†], [abr])
sont conservés tels quels conformément à la section 2.2.

Usage:
    from src.nlp.normalize import process_contract
    nlp_output = process_contract(contract)
"""

import re
import unicodedata
from pathlib import Path
from typing import Any


# ── Constantes pour les allographes u/v et i/j ───────────────────────────────

# Mots courants dont le 'u' initial est bien une voyelle (pas la consonne 'v').
_UV_WHITELIST: frozenset[str] = frozenset({"un", "une", "uns", "unes", "ut"})

# Pattern : 'u' précédé d'un non-lettre (début de mot), suivi de lettres.
# On utilise une classe explicite pour éviter les problèmes Unicode avec \b.
_UV_RE = re.compile(
    r"(?<![A-Za-zÀ-ɏ])u([A-Za-zÀ-ɏ]+)"
)

# Pattern : 'i' précédé d'un non-lettre, suivi d'une voyelle (position consonantique).
_IJ_RE = re.compile(
    r"(?<![A-Za-zÀ-ɏ])i([aeiouyÀ-ɏ][A-Za-zÀ-ɏ]*)"
)


# ── Vocabulaires pour la détection de langue ──────────────────────────────────

_LATIN_WORDS = frozenset([
    "et", "in", "de", "est", "ut", "cum", "ad", "per", "non", "sed",
    "qui", "que", "quod", "quia", "ergo", "item", "vel", "nec", "aut",
    "dominus", "deus", "rex", "papa", "ecclesia", "sanctus", "sancti",
    "anno", "die", "mense", "anno", "domini", "nostri", "super", "sub",
    "pro", "contra", "ante", "post", "inter", "sine", "nisi", "sicut",
])

_FRENCH_WORDS = frozenset([
    "le", "la", "les", "de", "du", "des", "et", "en", "un", "une",
    "ce", "est", "par", "pour", "que", "qui", "ou", "au", "aux",
    "il", "elle", "nous", "vous", "ils", "elles", "me", "te", "se",
    "dit", "fait", "bien", "tout", "plus", "car", "si", "ne", "pas",
    "son", "sa", "ses", "mon", "ma", "mes", "ton", "ta", "tes",
    "cest", "ceste", "ainsi", "mais", "sur", "sous", "vers",
])


# ── API publique ──────────────────────────────────────────────────────────────

def process_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Applique la normalisation NLP à un data contract CV complet.

    Seules les lignes avec ``needs_review=False`` sont normalisées
    automatiquement. Les lignes ``needs_review=True`` sont passées sans
    modification avec le flag conservé, conformément à la section 5.2.

    Args:
        contract: Dict retourné par ``build_data_contract()``.

    Returns:
        Dict de sortie NLP conforme à la section 9 de CONVENTIONS_NLP.md :
        {
          "page_id": "folio_stem",
          "image": "folio.jpeg",
          "date_cv": "...",
          "pipeline_cv": "kraken-cremma-medieval",
          "lines": [
            {
              "line_id": "l_0001",
              "transcription_cv": "...",
              "transcription_normalisee": "...",
              "corrections_appliquees": [...],
              "langue_detectee": ["fr"],
              "needs_review": false,
              "confidence": 0.923
            },
            ...
          ],
          "stats": {
            "n_lines": 42,
            "n_processed": 39,
            "n_skipped_review": 3,
            "n_corrections": 87
          }
        }
    """
    page_id = Path(contract["image"]).stem
    nlp_lines = []

    for line in contract["lines"]:
        cv_text = line["text"]
        needs_review = line["needs_review"]

        if needs_review:
            nlp_lines.append({
                "line_id":                  line["line_id"],
                "transcription_cv":         cv_text,
                "transcription_normalisee": cv_text,
                "corrections_appliquees":   [],
                "langue_detectee":          ["inconnu"],
                "needs_review":             True,
                "confidence":               line["confidence"],
            })
        else:
            normalized, corrections = normalize_line(cv_text)
            nlp_lines.append({
                "line_id":                  line["line_id"],
                "transcription_cv":         cv_text,
                "transcription_normalisee": normalized,
                "corrections_appliquees":   corrections,
                "langue_detectee":          detect_language(normalized),
                "needs_review":             False,
                "confidence":               line["confidence"],
            })

    n_processed  = sum(1 for l in nlp_lines if not l["needs_review"])
    n_skipped    = sum(1 for l in nlp_lines if l["needs_review"])
    n_corrections = sum(len(l["corrections_appliquees"]) for l in nlp_lines)

    return {
        "page_id":     page_id,
        "image":       contract["image"],
        "date_cv":     contract.get("date", ""),
        "pipeline_cv": contract.get("model", "kraken"),
        "lines":       nlp_lines,
        "stats": {
            "n_lines":            len(nlp_lines),
            "n_processed":        n_processed,
            "n_skipped_review":   n_skipped,
            "n_corrections":      n_corrections,
        },
    }


def normalize_line(text: str) -> tuple[str, list[dict]]:
    """Normalise une ligne de transcription médiévale.

    Applique dans l'ordre les règles de la section 2 de CONVENTIONS_NLP.md.
    Les marqueurs irréductibles ([?], [...], [†], [abr]) sont conservés.

    Args:
        text: Transcription brute issue du pipeline CV.

    Returns:
        Tuple (texte_normalisé, liste_corrections) où chaque correction est :
        {"position": int, "avant": str, "apres": str, "type": str}

    Example:
        >>> normalize_line("du saint esperit ⁊ par le (com)mandement")
        ('du saint esperit et par le commandement', [...])
    """
    corrections: list[dict] = []
    result = text

    # Les positions dans les corrections référencent le texte d'entrée original.
    # On applique les règles dans l'ordre de la doc (section 2).

    steps = [
        _normalize_unicode_nfc,
        _normalize_uv,
        _normalize_ij,
        _strip_rubric_tags,
        _normalize_tironian,
        _normalize_p_barre,
        _validate_abbreviations,
        _confirm_uncertain_readings,
        _normalize_column_separator,
    ]

    for step in steps:
        result, corr = step(result)
        corrections.extend(corr)

    return result, corrections


def detect_language(text: str) -> list[str]:
    """Détecte la/les langue(s) dominante(s) d'une ligne (section 4.1).

    Retourne ``["fr"]``, ``["la"]``, ``["fr", "la"]`` (mélange), ou
    ``["inconnu"]`` si aucun indice lexical n'est trouvé.

    Args:
        text: Texte normalisé.

    Returns:
        Liste d'étiquettes de langue ISO 639-1.
    """
    tokens = re.findall(r"\b[a-zæœȝ]+\b", text.lower())
    if not tokens:
        return ["inconnu"]

    latin_score  = sum(1 for t in tokens if t in _LATIN_WORDS)
    french_score = sum(1 for t in tokens if t in _FRENCH_WORDS)

    if latin_score == 0 and french_score == 0:
        return ["inconnu"]
    if latin_score > french_score:
        return ["la"]
    if french_score > latin_score:
        return ["fr"]
    return ["fr", "la"]


# ── Fonctions de normalisation (privées) ──────────────────────────────────────

def _normalize_unicode_nfc(text: str) -> tuple[str, list[dict]]:
    """Normalise le texte en forme canonique Unicode NFC (section 2.6).

    Fusionne les caractères décomposés (e + accent combining → é).
    Sans effet si le texte est déjà en NFC, ce qui est le cas pour la majorité
    des sorties Kraken.
    """
    normalized = unicodedata.normalize("NFC", text)
    if normalized == text:
        return text, []
    return normalized, [{"position": 0, "avant": text, "apres": normalized,
                         "type": "normalisation_nfc"}]


def _normalize_uv(text: str) -> tuple[str, list[dict]]:
    """Normalise l'allographe u/v : 'u' initial de mot → 'v' (section 2.5).

    Dans les manuscrits médiévaux et du XVIIe siècle, la lettre 'u' en tête
    de mot remplit souvent le rôle consonantique de 'v'.
    Ex : 'uous' → 'vous', 'uouy' → 'vouy', 'uolume' → 'volume'.

    Exceptions (_UV_WHITELIST) : mots courants dont le 'u' initial est une
    vraie voyelle ('un', 'une', 'ut'). Les faux positifs résiduels doivent
    être revus manuellement.
    """
    corrections: list[dict] = []
    result = list(text)
    for m in _UV_RE.finditer(text):
        word = "u" + m.group(1)
        if word.lower() in _UV_WHITELIST:
            continue
        if text[m.start()] == "u":  # minuscule uniquement
            corrections.append({
                "position": m.start(),
                "avant":    "u",
                "apres":    "v",
                "type":     "normalisation_uv",
            })
            result[m.start()] = "v"
    return "".join(result), corrections


def _normalize_ij(text: str) -> tuple[str, list[dict]]:
    """Normalise l'allographe i/j : 'i' initial avant voyelle → 'j' (section 2.5).

    Dans les manuscrits anciens, 'j' n'était pas différencié de 'i'. Le 'i'
    en tête de mot devant une voyelle est presque toujours la consonne 'j'.
    Ex : 'ie' → 'je', 'iugrimo' → 'jugrimo'.

    Les mots comme 'il', 'in' ne sont pas touchés car 'l' et 'n' sont des
    consonnes — la regex ne s'applique qu'avant une voyelle.
    """
    corrections: list[dict] = []
    result = list(text)
    for m in _IJ_RE.finditer(text):
        if text[m.start()] == "i":  # minuscule uniquement
            corrections.append({
                "position": m.start(),
                "avant":    "i",
                "apres":    "j",
                "type":     "normalisation_ij",
            })
            result[m.start()] = "j"
    return "".join(result), corrections


def _strip_rubric_tags(text: str) -> tuple[str, list[dict]]:
    """Supprime les balises <R>...</R> en conservant le contenu (section 2.3)."""
    corrections = []
    for m in re.finditer(r"<R>(.*?)</R>", text):
        corrections.append({
            "position": m.start(),
            "avant":    m.group(0),
            "apres":    m.group(1),
            "type":     "suppression_balise_rubrique",
        })
    return re.sub(r"<R>(.*?)</R>", r"\1", text), corrections


def _normalize_tironian(text: str) -> tuple[str, list[dict]]:
    """Normalise ⁊ (U+204A, note tironienne) → 'et' (section 2.1 et 2.4)."""
    corrections = [
        {"position": m.start(), "avant": "⁊", "apres": "et",
         "type": "normalisation_tironien"}
        for m in re.finditer("⁊", text)
    ]
    return text.replace("⁊", "et"), corrections


def _normalize_p_barre(text: str) -> tuple[str, list[dict]]:
    """Normalise ꝑ (U+A751, p barré) → 'per' ou 'par' selon le contexte (2.1, 2.4).

    Heuristique : si le caractère suivant est 'a', expansion en 'par', sinon 'per'.
    Cette règle couvre les cas les plus courants en latin médiéval et vieux français.
    """
    corrections: list[dict] = []
    result_chars: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == "ꝑ":
            following = text[i + 1:i + 2].lower()
            expansion = "par" if following == "a" else "per"
            corrections.append({
                "position": i,
                "avant":    "ꝑ",
                "apres":    expansion,
                "type":     "normalisation_p_barre",
            })
            result_chars.append(expansion)
        else:
            result_chars.append(text[i])
        i += 1
    return "".join(result_chars), corrections


def _validate_abbreviations(text: str) -> tuple[str, list[dict]]:
    """Valide les développements d'abréviations incertains (mot) → mot (section 2.1).

    Les parenthèses dans le champ ``text`` signalent toujours une incertitude
    éditoriale du pipeline CV, jamais la ponctuation originale (piège fréquent
    mentionné en section 2.1). On les retire donc pour valider l'expansion.
    """
    corrections = []
    for m in re.finditer(r"\(([^)]+)\)", text):
        corrections.append({
            "position": m.start(),
            "avant":    m.group(0),
            "apres":    m.group(1),
            "type":     "validation_abreviation",
        })
    return re.sub(r"\(([^)]+)\)", r"\1", text), corrections


def _confirm_uncertain_readings(text: str) -> tuple[str, list[dict]]:
    """Confirme les lectures incertaines [mot?] → mot (section 2.2).

    Les marqueurs [?], [...] et [†] (un seul caractère ou longueur inconnue)
    sont préservés car ils ne peuvent pas être résolus sans contexte plus large.
    """
    corrections = []
    for m in re.finditer(r"\[([^\]\[?][^\]]*)\?\]", text):
        corrections.append({
            "position": m.start(),
            "avant":    m.group(0),
            "apres":    m.group(1),
            "type":     "confirmation_lecture_incertaine",
        })
    return re.sub(r"\[([^\]\[?][^\]]*)\?\]", r"\1", text), corrections


def _normalize_column_separator(text: str) -> tuple[str, list[dict]]:
    """Remplace || (fin de colonne) par un espace (section 2.3)."""
    corrections = [
        {"position": m.start(), "avant": "||", "apres": " ",
         "type": "normalisation_separateur_colonne"}
        for m in re.finditer(r"\|\|", text)
    ]
    return text.replace("||", " "), corrections
