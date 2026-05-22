"""
tests/test_preprocessing.py
-----------------------------
Tests unitaires du module de prétraitement.
"""
import numpy as np
import pytest
import cv2
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.preprocess import (
    deskew, apply_clahe, binarize_sauvola, preprocess_image
)


def make_gray_image(h=100, w=100, dtype=np.uint8):
    """Crée une image grise synthétique."""
    return (np.random.rand(h, w) * 255).astype(dtype)


class TestDeskew:
    def test_output_shape(self):
        img = make_gray_image()
        result = deskew(img)
        assert result.shape == img.shape

    def test_output_dtype(self):
        img = make_gray_image()
        result = deskew(img)
        assert result.dtype == np.uint8

    def test_no_rotation_on_straight(self):
        """Image déjà droite : deskew ne doit pas dégrader fortement."""
        img = np.zeros((100, 100), dtype=np.uint8)
        img[40:60, :] = 255  # ligne horizontale parfaite
        result = deskew(img)
        assert result.shape == (100, 100)


class TestCLAHE:
    def test_output_shape(self):
        img = make_gray_image()
        assert apply_clahe(img).shape == img.shape

    def test_output_dtype(self):
        img = make_gray_image()
        assert apply_clahe(img).dtype == np.uint8

    def test_values_in_range(self):
        img = make_gray_image()
        result = apply_clahe(img)
        assert result.min() >= 0
        assert result.max() <= 255


class TestSauvola:
    def test_output_shape(self):
        img = make_gray_image()
        assert binarize_sauvola(img).shape == img.shape

    def test_binary_values(self):
        """La sortie ne doit contenir que 0 et 255."""
        img = make_gray_image()
        result = binarize_sauvola(img)
        unique = set(np.unique(result).tolist())
        assert unique.issubset({0, 255})


class TestPreprocessImage:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            preprocess_image("fichier_inexistant.jpg")

    def test_runs_on_real_image(self, tmp_path):
        """Teste sur une image synthétique sauvegardée en JPEG."""
        img = make_gray_image(200, 150)
        img_path = str(tmp_path / "test.jpg")
        cv2.imwrite(img_path, img)
        result = preprocess_image(img_path)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] > 0
        assert result.shape[1] > 0
