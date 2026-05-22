"""
tests/test_metrics.py
----------------------
Tests unitaires des métriques d'évaluation.
"""
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import (
    compute_cer, compute_wer, bootstrap_cer_ci,
    compute_iou_polygon, compute_mean_iou
)


class TestCER:
    def test_perfect(self):
        assert compute_cer(["bonjour"], ["bonjour"]) == 0.0

    def test_all_wrong(self):
        cer = compute_cer(["xxx"], ["abc"])
        assert cer == pytest.approx(1.0)

    def test_partial(self):
        # "bnjour" vs "bonjour" → 1 erreur / 7 chars
        cer = compute_cer(["bnjour"], ["bonjour"])
        assert cer == pytest.approx(1/7, rel=1e-3)

    def test_empty_reference(self):
        assert compute_cer([""], [""]) == 0.0

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            compute_cer(["a", "b"], ["a"])

    def test_multiple_lines(self):
        preds = ["bonjour", "monde"]
        refs  = ["bonjour", "monde"]
        assert compute_cer(preds, refs) == 0.0


class TestWER:
    def test_perfect(self):
        assert compute_wer(["le roman de la rose"], ["le roman de la rose"]) == 0.0

    def test_one_error(self):
        # 1 mot différent sur 5
        wer = compute_wer(["le romans de la rose"], ["le roman de la rose"])
        assert wer == pytest.approx(1/5, rel=1e-3)

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            compute_wer(["a"], ["a", "b"])


class TestBootstrapCI:
    def test_returns_tuple(self):
        preds = ["bonjour"] * 20
        refs  = ["bonjour"] * 20
        result = bootstrap_cer_ci(preds, refs, n_bootstrap=100)
        assert len(result) == 2

    def test_perfect_cer_ci_is_zero(self):
        preds = ["abc"] * 50
        refs  = ["abc"] * 50
        low, high = bootstrap_cer_ci(preds, refs, n_bootstrap=200)
        assert low == pytest.approx(0.0)
        assert high == pytest.approx(0.0)

    def test_low_leq_high(self):
        preds = ["bonjur", "mnde", "ici"]
        refs  = ["bonjour", "monde", "ici"]
        low, high = bootstrap_cer_ci(preds, refs, n_bootstrap=200)
        assert low <= high


class TestIoU:
    def test_perfect_overlap(self):
        poly = [[0,0],[10,0],[10,10],[0,10]]
        assert compute_iou_polygon(poly, poly) == pytest.approx(1.0)

    def test_no_overlap(self):
        p1 = [[0,0],[5,0],[5,5],[0,5]]
        p2 = [[10,10],[20,10],[20,20],[10,20]]
        assert compute_iou_polygon(p1, p2) == 0.0

    def test_partial_overlap(self):
        p1 = [[0,0],[10,0],[10,10],[0,10]]
        p2 = [[5,0],[15,0],[15,10],[5,10]]
        iou = compute_iou_polygon(p1, p2)
        assert 0.0 < iou < 1.0

    def test_mean_iou(self):
        lines_pred = [{"polygon": [[0,0],[10,0],[10,5],[0,5]]}]
        lines_ref  = [{"polygon": [[0,0],[10,0],[10,5],[0,5]]}]
        assert compute_mean_iou(lines_pred, lines_ref) == pytest.approx(1.0)
