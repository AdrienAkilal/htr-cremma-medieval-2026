"""
tests/test_nlp.py
------------------
Tests unitaires du module de normalisation NLP.
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jsonschema
import pytest
from src.nlp.normalize import (
    normalize_line,
    detect_language,
    process_contract,
)

# ── Schéma JSON du data contract ─────────────────────────────────────────────

DATA_CONTRACT_SCHEMA = {
    "type": "object",
    "required": ["image", "sha256", "date", "model", "lines", "stats"],
    "properties": {
        "image":      {"type": "string"},
        "sha256":     {"type": "string", "minLength": 64, "maxLength": 64},
        "date":       {"type": "string"},
        "model":      {"type": "string"},
        "conf_threshold": {"type": "number"},
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["line_id", "text", "confidence", "needs_review",
                             "polygon", "baseline"],
                "properties": {
                    "line_id":      {"type": "string"},
                    "text":         {"type": "string"},
                    "confidence":   {"type": "number", "minimum": 0, "maximum": 1},
                    "needs_review": {"type": "boolean"},
                    "polygon":      {"type": "array"},
                    "baseline":     {"type": "array"},
                },
            },
        },
        "stats": {
            "type": "object",
            "required": ["n_lines", "n_needs_review", "mean_confidence"],
            "properties": {
                "n_lines":          {"type": "integer"},
                "n_needs_review":   {"type": "integer"},
                "mean_confidence":  {"type": "number"},
            },
        },
    },
}


class TestNormalizeLine:
    def test_tironian_note(self):
        text, corr = normalize_line("roi ⁊ reine")
        assert text == "roi et reine"
        assert len(corr) == 1
        assert corr[0]["type"] == "normalisation_tironien"
        assert corr[0]["avant"] == "⁊"
        assert corr[0]["apres"] == "et"

    def test_multiple_tironian(self):
        text, corr = normalize_line("père ⁊ mère ⁊ fils")
        assert text == "père et mère et fils"
        assert len(corr) == 2

    def test_abbreviation_parentheses(self):
        text, corr = normalize_line("par le (com)mandement")
        assert text == "par le commandement"
        assert any(c["type"] == "validation_abreviation" for c in corr)

    def test_abbreviation_multiple(self):
        text, corr = normalize_line("(com)me (tou)t")
        assert text == "comme tout"
        assert len(corr) == 2

    def test_uncertain_reading_confirmed(self):
        text, corr = normalize_line("le [grant?] seigneur")
        assert text == "le grant seigneur"
        assert any(c["type"] == "confirmation_lecture_incertaine" for c in corr)

    def test_unresolvable_markers_preserved(self):
        text, corr = normalize_line("le [?] seigneur [...]")
        assert "[?]" in text
        assert "[...]" in text
        assert not corr

    def test_physical_lacuna_preserved(self):
        text, corr = normalize_line("le [†] mot")
        assert "[†]" in text
        assert not corr

    def test_unresolved_abbreviation_preserved(self):
        text, corr = normalize_line("le [abr] mot")
        assert "[abr]" in text
        assert not corr

    def test_rubric_tags_stripped(self):
        text, corr = normalize_line("<R>Incipit</R> liber primus")
        assert text == "Incipit liber primus"
        assert any(c["type"] == "suppression_balise_rubrique" for c in corr)

    def test_column_separator(self):
        text, corr = normalize_line("premier || deuxième")
        assert "||" not in text
        assert any(c["type"] == "normalisation_separateur_colonne" for c in corr)

    def test_p_barre_per(self):
        text, corr = normalize_line("ꝑ omnia")
        assert text.startswith("per")
        assert any(c["type"] == "normalisation_p_barre" for c in corr)

    def test_p_barre_par(self):
        text, corr = normalize_line("ꝑar excellence")
        assert text.startswith("par")

    def test_no_change_plain_text(self):
        text, corr = normalize_line("ce est li romans de la rose")
        assert text == "ce est li romans de la rose"
        assert corr == []

    def test_combined_corrections(self):
        text, corr = normalize_line("du saint esperit ⁊ par le (com)mandement")
        assert "et" in text
        assert "(com)" not in text
        assert len(corr) == 2


class TestDetectLanguage:
    def test_french(self):
        lang = detect_language("le roi et la reine")
        assert "fr" in lang

    def test_latin(self):
        lang = detect_language("dominus deus rex ecclesia")
        assert "la" in lang

    def test_unknown(self):
        lang = detect_language("xzqpfk")
        assert lang == ["inconnu"]

    def test_empty(self):
        lang = detect_language("")
        assert lang == ["inconnu"]

    def test_mixed(self):
        lang = detect_language("dominus le roi")
        assert isinstance(lang, list)
        assert len(lang) >= 1


class TestProcessContract:
    SAMPLE_CONTRACT = {
        "image": "folio_001.jpeg",
        "date": "2026-06-01T10:00:00",
        "model": "kraken-cremma-medieval",
        "lines": [
            {
                "line_id": "l_0001",
                "text": "du saint esperit ⁊ par le (com)mandement",
                "confidence": 0.92,
                "needs_review": False,
                "polygon": [[0, 0], [100, 0], [100, 20], [0, 20]],
                "baseline": [],
            },
            {
                "line_id": "l_0002",
                "text": "ligne peu fiable [?] ici",
                "confidence": 0.45,
                "needs_review": True,
                "polygon": [[0, 25], [100, 25], [100, 45], [0, 45]],
                "baseline": [],
            },
        ],
        "stats": {
            "n_lines": 2, "n_needs_review": 1,
            "needs_review_rate": 0.5, "mean_confidence": 0.685,
        },
    }

    def test_output_keys(self):
        out = process_contract(self.SAMPLE_CONTRACT)
        for key in ["page_id", "image", "date_cv", "pipeline_cv", "lines", "stats"]:
            assert key in out

    def test_page_id_derived_from_image(self):
        out = process_contract(self.SAMPLE_CONTRACT)
        assert out["page_id"] == "folio_001"

    def test_reliable_line_normalized(self):
        out = process_contract(self.SAMPLE_CONTRACT)
        line = next(l for l in out["lines"] if l["line_id"] == "l_0001")
        assert line["needs_review"] is False
        assert "et" in line["transcription_normalisee"]
        assert "(com)" not in line["transcription_normalisee"]
        assert len(line["corrections_appliquees"]) > 0

    def test_review_line_not_normalized(self):
        out = process_contract(self.SAMPLE_CONTRACT)
        line = next(l for l in out["lines"] if l["line_id"] == "l_0002")
        assert line["needs_review"] is True
        assert line["transcription_normalisee"] == line["transcription_cv"]
        assert line["corrections_appliquees"] == []
        assert line["langue_detectee"] == ["inconnu"]

    def test_stats(self):
        out = process_contract(self.SAMPLE_CONTRACT)
        assert out["stats"]["n_lines"] == 2
        assert out["stats"]["n_processed"] == 1
        assert out["stats"]["n_skipped_review"] == 1

    def test_cv_transcription_preserved(self):
        out = process_contract(self.SAMPLE_CONTRACT)
        line = next(l for l in out["lines"] if l["line_id"] == "l_0001")
        assert line["transcription_cv"] == "du saint esperit ⁊ par le (com)mandement"


class TestNewNormalizationRules:
    """Tests pour les règles NFC, u/v et i/j ajoutées (section 2.5–2.6)."""

    def test_nfc_no_change_on_precomposed(self):
        # Kraken produit généralement du NFC — aucune correction attendue.
        text = "été roi"
        result, corr = normalize_line(text)
        nfc_corrs = [c for c in corr if c["type"] == "normalisation_nfc"]
        assert nfc_corrs == []
        assert result == text

    def test_uv_initial_u_before_consonant_sequence(self):
        # 'uous' en tête de mot → 'vous'
        text = "Ie uous mandars"
        result, corr = normalize_line(text)
        assert "vous" in result
        assert any(c["type"] == "normalisation_uv" for c in corr)

    def test_uv_whitelist_un_preserved(self):
        # 'un' doit rester 'un' (vraie voyelle initiale)
        text = "un seul homme"
        result, corr = normalize_line(text)
        assert result.startswith("un")
        assert not any(c["type"] == "normalisation_uv" for c in corr)

    def test_uv_multiple_occurrences(self):
        # Plusieurs 'u' initiaux dans la même ligne
        text = "uous uolume uers"
        result, corr = normalize_line(text)
        uv_corrs = [c for c in corr if c["type"] == "normalisation_uv"]
        assert len(uv_corrs) == 3
        assert result == "vous volume vers"

    def test_uv_medial_u_not_changed(self):
        # 'u' en position médiane ne doit PAS être modifié
        text = "auteur sauver"
        result, corr = normalize_line(text)
        uv_corrs = [c for c in corr if c["type"] == "normalisation_uv"]
        assert uv_corrs == []

    def test_ij_initial_i_before_vowel(self):
        # 'ie' en tête de mot → 'je'
        text = "que ie me suis"
        result, corr = normalize_line(text)
        assert "je" in result
        assert any(c["type"] == "normalisation_ij" for c in corr)

    def test_ij_il_preserved(self):
        # 'il' ne doit pas être modifié ('l' est une consonne)
        text = "il est venu"
        result, corr = normalize_line(text)
        assert result.startswith("il")
        assert not any(c["type"] == "normalisation_ij" for c in corr)

    def test_ij_uppercase_not_changed(self):
        # Majuscule 'I' non affectée par la règle minuscule
        text = "Ie suis"
        result, corr = normalize_line(text)
        ij_corrs = [c for c in corr if c["type"] == "normalisation_ij"]
        assert ij_corrs == []

    def test_normalization_does_not_degrade_plain_text(self):
        # Texte sans marqueurs médiévaux → aucun changement NFC/u-v/i-j
        text = "le roi de france est puissant"
        result, corr = normalize_line(text)
        new_rule_corrs = [c for c in corr if c["type"] in
                          ("normalisation_nfc", "normalisation_uv", "normalisation_ij")]
        assert new_rule_corrs == []
        assert result == text


class TestDataContractSchema:
    """Valide le schéma JSON des data contracts produits (consignes NLP §1)."""

    _CONTRACT_PATHS = [
        Path(__file__).parent.parent / "output" / "f12" / "f12_data_contract.json",
        Path(__file__).parent.parent / "output" / "f279" / "f279_data_contract.json",
    ]

    @pytest.mark.parametrize("contract_path", _CONTRACT_PATHS)
    def test_contract_schema_valid(self, contract_path: Path):
        if not contract_path.exists():
            pytest.skip(f"Data contract introuvable : {contract_path}")
        with contract_path.open(encoding="utf-8") as f:
            contract = json.load(f)
        # Lève jsonschema.ValidationError si invalide
        jsonschema.validate(instance=contract, schema=DATA_CONTRACT_SCHEMA)

    @pytest.mark.parametrize("contract_path", _CONTRACT_PATHS)
    def test_contract_sha256_length(self, contract_path: Path):
        if not contract_path.exists():
            pytest.skip(f"Data contract introuvable : {contract_path}")
        with contract_path.open(encoding="utf-8") as f:
            contract = json.load(f)
        assert len(contract["sha256"]) == 64, "sha256 doit être un hash hex de 64 caractères"

    @pytest.mark.parametrize("contract_path", _CONTRACT_PATHS)
    def test_contract_lines_have_required_fields(self, contract_path: Path):
        if not contract_path.exists():
            pytest.skip(f"Data contract introuvable : {contract_path}")
        with contract_path.open(encoding="utf-8") as f:
            contract = json.load(f)
        for line in contract["lines"]:
            assert "line_id"      in line
            assert "text"         in line
            assert "confidence"   in line
            assert "needs_review" in line
            assert "polygon"      in line
