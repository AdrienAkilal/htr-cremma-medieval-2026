"""
tests/test_aggregation.py
--------------------------
Tests unitaires du module d'agrégation (data contract).
"""
import json
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.aggregation.aggregate import (
    build_data_contract, save_data_contract, export_page_xml, _escape_xml
)

SAMPLE_LINES = [
    {
        "line_id":      "l_0001",
        "text":         "ce est li romans de la rose",
        "confidence":   0.923,
        "needs_review": False,
        "polygon":      [[10, 20], [200, 20], [200, 40], [10, 40]],
        "baseline":     [[10, 35], [200, 35]],
    },
    {
        "line_id":      "l_0002",
        "text":         "n art damors est tote eclos",
        "confidence":   0.712,
        "needs_review": True,
        "polygon":      [[10, 45], [200, 45], [200, 65], [10, 65]],
        "baseline":     [[10, 60], [200, 60]],
    },
]


class TestBuildDataContract:
    def test_required_keys(self):
        contract = build_data_contract("image.jpg", SAMPLE_LINES)
        for key in ["image", "sha256", "date", "lines", "stats",
                    "conf_threshold", "coordinate_system"]:
            assert key in contract

    def test_coordinate_system(self):
        contract = build_data_contract("image.jpg", SAMPLE_LINES)
        cs = contract["coordinate_system"]
        assert cs["origin"] == "top-left"
        assert cs["unit"] == "pixels"

    def test_stats(self):
        contract = build_data_contract("image.jpg", SAMPLE_LINES)
        assert contract["stats"]["n_lines"] == 2
        assert contract["stats"]["n_needs_review"] == 1
        assert contract["stats"]["needs_review_rate"] == pytest.approx(0.5)

    def test_lines_content(self):
        contract = build_data_contract("image.jpg", SAMPLE_LINES)
        assert len(contract["lines"]) == 2
        assert contract["lines"][0]["text"] == "ce est li romans de la rose"

    def test_missing_key_raises(self):
        bad_line = {"line_id": "l_001", "text": "test"}  # manque confidence etc.
        with pytest.raises(ValueError):
            build_data_contract("image.jpg", [bad_line])


class TestSaveDataContract:
    def test_creates_file(self, tmp_path):
        contract = build_data_contract("image.jpg", SAMPLE_LINES)
        out = str(tmp_path / "contract.json")
        save_data_contract(contract, out)
        assert Path(out).exists()

    def test_valid_json(self, tmp_path):
        contract = build_data_contract("image.jpg", SAMPLE_LINES)
        out = str(tmp_path / "contract.json")
        save_data_contract(contract, out)
        data = json.loads(Path(out).read_text(encoding="utf-8"))
        assert "lines" in data


class TestExportPageXml:
    def test_creates_file(self, tmp_path):
        contract = build_data_contract("image.jpg", SAMPLE_LINES)
        out = str(tmp_path / "folio.page.xml")
        export_page_xml(contract, out)
        assert Path(out).exists()

    def test_valid_xml(self, tmp_path):
        from lxml import etree
        contract = build_data_contract("image.jpg", SAMPLE_LINES)
        out = str(tmp_path / "folio.page.xml")
        export_page_xml(contract, out)
        tree = etree.parse(out)
        assert tree is not None

    def test_line_ids_present(self, tmp_path):
        contract = build_data_contract("image.jpg", SAMPLE_LINES)
        out = str(tmp_path / "folio.page.xml")
        export_page_xml(contract, out)
        content = Path(out).read_text(encoding="utf-8")
        assert "l_0001" in content
        assert "l_0002" in content


class TestEscapeXml:
    def test_ampersand(self):
        assert _escape_xml("a & b") == "a &amp; b"

    def test_less_than(self):
        assert _escape_xml("a < b") == "a &lt; b"

    def test_greater_than(self):
        assert _escape_xml("a > b") == "a &gt; b"
