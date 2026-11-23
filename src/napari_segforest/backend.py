"""Abstraction de backend de calcul : bascule GPU si disponible, sinon CPU.

Le but est d'ecrire les filtres une seule fois et de laisser le backend
choisir l'implementation. On expose un objet `xp` facon numpy/cupy et un
ensemble de filtres 3D (gaussien, gradient, hessian...) qui delegent a
pyclesperanto sur GPU ou scikit-image/scipy sur CPU.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_FORCE_CPU = os.environ.get("SEGFOREST_FORCE_CPU", "0") == "1"


def _try_gpu():
    if _FORCE_CPU:
        return None
    try:
        import pyclesperanto_prototype as cle  # noqa

        # Verifie qu'un device existe reellement
        dev = cle.get_device()
        logger.info("Backend GPU actif : %s", dev)
        return cle
    except Exception as exc:  # pragma: no cover - depend du materiel
        logger.info("GPU indisponible (%s), bascule sur CPU.", exc)
        return None


_CLE = _try_gpu()


def gpu_available() -> bool:
    return _CLE is not None


def backend_name() -> str:
    return "gpu (pyclesperanto)" if gpu_available() else "cpu (scikit-image/scipy)"


# --------------------------------------------------------------------------
# Filtres unifies. Entree/sortie : numpy.ndarray float32 (2D ou 3D).
# On garde une API minimale ; le moteur de features compose ces primitives.
# --------------------------------------------------------------------------
import numpy as np
from scipy import ndimage as ndi
from skimage import feature as skfeature


def gaussian(image: np.ndarray, sigma: float) -> np.ndarray:
    if gpu_available():
        out = _CLE.gaussian_blur(image, sigma_x=sigma, sigma_y=sigma,
                                 sigma_z=sigma if image.ndim == 3 else 0)
        return _CLE.pull(out).astype(np.float32)
    return ndi.gaussian_filter(image, sigma=sigma, output=np.float32)


def gradient_magnitude(image: np.ndarray, sigma: float) -> np.ndarray:
    sm = gaussian(image, sigma) if sigma > 0 else image.astype(np.float32)
    grads = np.gradient(sm)
    return np.sqrt(np.sum([g ** 2 for g in grads], axis=0)).astype(np.float32)


def laplacian_of_gaussian(image: np.ndarray, sigma: float) -> np.ndarray:
    sm = gaussian(image, sigma) if sigma > 0 else image.astype(np.float32)
    return ndi.laplace(sm).astype(np.float32)


def difference_of_gaussians(image: np.ndarray, sigma: float, ratio: float = 1.6) -> np.ndarray:
    return (gaussian(image, sigma) - gaussian(image, sigma * ratio)).astype(np.float32)


def hessian_eigen(image: np.ndarray, sigma: float) -> list[np.ndarray]:
    """Valeurs propres de la matrice hessienne (structure : blobs/ridges)."""
    from skimage.feature import hessian_matrix, hessian_matrix_eigvals

    H = hessian_matrix(image, sigma=sigma, use_gaussian_derivatives=True)
    eigs = hessian_matrix_eigvals(H)  # shape (ndim, *image.shape)
    return [e.astype(np.float32) for e in eigs]


def structure_tensor_eigen(image: np.ndarray, sigma: float) -> list[np.ndarray]:
    from skimage.feature import structure_tensor, structure_tensor_eigenvalues

    A = structure_tensor(image, sigma=sigma, order="rc")
    eigs = structure_tensor_eigenvalues(A)
    return [e.astype(np.float32) for e in eigs]


# --------------------------------------------------------------------------
# Morphologie mathematique (niveaux de gris), parametree par un rayon.
# CPU : scipy.ndimage + footprint disk/ball (n-D). GPU : pyclesperanto.
# OpenCV serait plus rapide mais ne gere que la 2D -> ecarte pour la 3D.
# --------------------------------------------------------------------------
def _footprint(ndim: int, radius: float) -> np.ndarray:
    from skimage.morphology import ball, disk

    r = max(1, int(round(radius)))
    return ball(r) if ndim == 3 else disk(r)


def erosion(image: np.ndarray, radius: float) -> np.ndarray:
    if radius < 0.5:
        return image.astype(np.float32)
    if gpu_available():
        out = _CLE.minimum_sphere(image, radius_x=radius, radius_y=radius,
                                  radius_z=radius if image.ndim == 3 else 0)
        return _CLE.pull(out).astype(np.float32)
    return ndi.grey_erosion(image, footprint=_footprint(image.ndim, radius)).astype(np.float32)


def dilation(image: np.ndarray, radius: float) -> np.ndarray:
    if radius < 0.5:
        return image.astype(np.float32)
    if gpu_available():
        out = _CLE.maximum_sphere(image, radius_x=radius, radius_y=radius,
                                  radius_z=radius if image.ndim == 3 else 0)
        return _CLE.pull(out).astype(np.float32)
    return ndi.grey_dilation(image, footprint=_footprint(image.ndim, radius)).astype(np.float32)


def opening(image: np.ndarray, radius: float) -> np.ndarray:
    """Erosion puis dilatation : supprime les petits objets clairs / le bruit."""
    if radius < 0.5:
        return image.astype(np.float32)
    return dilation(erosion(image, radius), radius)


def closing(image: np.ndarray, radius: float) -> np.ndarray:
    """Dilatation puis erosion : bouche les petits trous sombres."""
    if radius < 0.5:
        return image.astype(np.float32)
    return erosion(dilation(image, radius), radius)


def morphological_gradient(image: np.ndarray, radius: float) -> np.ndarray:
    """Dilatation - erosion : reponse forte sur les contours (epaisseur ~ rayon)."""
    if radius < 0.5:
        return np.zeros_like(image, dtype=np.float32)
    return (dilation(image, radius) - erosion(image, radius)).astype(np.float32)


def white_tophat(image: np.ndarray, radius: float) -> np.ndarray:
    """image - ouverture : fait ressortir les petits objets CLAIRS."""
    if radius < 0.5:
        return np.zeros_like(image, dtype=np.float32)
    return (image.astype(np.float32) - opening(image, radius)).astype(np.float32)


def black_tophat(image: np.ndarray, radius: float) -> np.ndarray:
    """fermeture - image : fait ressortir les petits objets SOMBRES."""
    if radius < 0.5:
        return np.zeros_like(image, dtype=np.float32)
    return (closing(image, radius) - image.astype(np.float32)).astype(np.float32)


def median(image: np.ndarray, radius: float) -> np.ndarray:
    """Filtre median (rang) : debruitage preservant les contours."""
    if radius < 0.5:
        return image.astype(np.float32)
    if gpu_available():
        out = _CLE.median_sphere(image, radius_x=int(round(radius)),
                                 radius_y=int(round(radius)),
                                 radius_z=int(round(radius)) if image.ndim == 3 else 0)
        return _CLE.pull(out).astype(np.float32)
    return ndi.median_filter(image, footprint=_footprint(image.ndim, radius)).astype(np.float32)
