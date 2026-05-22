"""
src/preprocessing/preprocess.py
--------------------------------
Pipeline de prétraitement des images de manuscrits.

Chaîne appliquée dans l'ordre :
  1. Redressement (deskew) par projection horizontale
  2. Amélioration du contraste (CLAHE)
  3. Binarisation adaptative (Sauvola)

Usage:
    from src.preprocessing.preprocess import preprocess_image
    img_bin = preprocess_image("folio.jpeg", save_dir="data/preprocessed")
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from skimage.filters import threshold_sauvola


def fixer_seeds(seed: int = 42) -> None:
    """Fixe les seeds pour la reproductibilité.

    Args:
        seed: Valeur du seed (défaut 42).
    """
    np.random.seed(seed)


def deskew(image_gray: np.ndarray) -> np.ndarray:
    """Corrige l'inclinaison d'une image en niveaux de gris.

    Utilise la méthode des projections horizontales : on fait pivoter
    l'image sur une plage d'angles et on retient celui qui maximise
    la variance des projections (lignes les plus nettes).

    Args:
        image_gray: Image en niveaux de gris (uint8, shape HxW).

    Returns:
        Image redressée (uint8, shape HxW).

    Example:
        >>> gray = cv2.imread("folio.jpg", cv2.IMREAD_GRAYSCALE)
        >>> straight = deskew(gray)
    """
    # Binarisation rapide pour la détection de l'angle
    _, thresh = cv2.threshold(image_gray, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    angles = np.arange(-10, 10, 0.5)
    scores = []
    h, w = thresh.shape

    for angle in angles:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(thresh, M, (w, h),
                                  flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=0)
        # Variance des sommes de lignes = score de netteté
        projection = np.sum(rotated, axis=1).astype(float)
        scores.append(np.var(projection))

    best_angle = angles[np.argmax(scores)]

    if abs(best_angle) < 0.3:
        return image_gray  # pas besoin de corriger

    M = cv2.getRotationMatrix2D((w / 2, h / 2), best_angle, 1.0)
    deskewed = cv2.warpAffine(image_gray, M, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
    return deskewed


def apply_clahe(image_gray: np.ndarray,
                clip_limit: float = 2.0,
                tile_grid: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Améliore le contraste local par CLAHE.

    Args:
        image_gray: Image en niveaux de gris (uint8).
        clip_limit: Limite de clipping (défaut 2.0).
        tile_grid: Taille de la grille de tuiles (défaut (8, 8)).

    Returns:
        Image avec contraste amélioré (uint8).
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(image_gray)


def binarize_sauvola(image_gray: np.ndarray,
                     window_size: int = 25,
                     k: float = 0.2) -> np.ndarray:
    """Binarise une image par la méthode de Sauvola.

    Adaptée aux manuscrits : tient compte des variations locales de fond
    (parchemin jauni, taches, dégradés).

    Args:
        image_gray: Image en niveaux de gris (uint8).
        window_size: Taille de la fenêtre locale (défaut 25, impair).
        k: Paramètre de sensibilité (défaut 0.2).

    Returns:
        Image binaire (uint8, 0 ou 255).
    """
    thresh = threshold_sauvola(image_gray, window_size=window_size, k=k)
    binary = (image_gray > thresh).astype(np.uint8) * 255
    return binary


def preprocess_image(image_path: str,
                     save_dir: str | None = None,
                     deskew_enabled: bool = True,
                     clahe_enabled: bool = True,
                     binarize_enabled: bool = True) -> np.ndarray:
    """Applique la chaîne complète de prétraitement sur une image.

    Args:
        image_path: Chemin vers l'image source (JPEG ou TIFF).
        save_dir: Si fourni, sauvegarde les étapes intermédiaires.
        deskew_enabled: Active le redressement (défaut True).
        clahe_enabled: Active le CLAHE (défaut True).
        binarize_enabled: Active la binarisation Sauvola (défaut True).

    Returns:
        Image prétraitée (uint8, niveaux de gris ou binaire).

    Raises:
        FileNotFoundError: Si image_path n'existe pas.

    Example:
        >>> result = preprocess_image("data/raw/folio.jpeg",
        ...                           save_dir="data/preprocessed")
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image introuvable : {image_path}")

    # Chargement en niveaux de gris
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Impossible de lire : {image_path}")

    if save_dir:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)

    stem = path.stem

    # Étape 1 — Deskew
    if deskew_enabled:
        img = deskew(img)
        if save_dir:
            cv2.imwrite(str(out / f"{stem}_1_deskew.jpg"), img)

    # Étape 2 — CLAHE
    if clahe_enabled:
        img = apply_clahe(img)
        if save_dir:
            cv2.imwrite(str(out / f"{stem}_2_clahe.jpg"), img)

    # Étape 3 — Binarisation Sauvola
    if binarize_enabled:
        img = binarize_sauvola(img)
        if save_dir:
            cv2.imwrite(str(out / f"{stem}_3_binary.jpg"), img)

    return img
