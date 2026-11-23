"""Primitives de filtrage n-D (2D/3D) avec acceleration GPU optionnelle.

Toutes les fonctions prennent et rendent des ``numpy.ndarray`` float32.
Les parametres d'echelle sont donnes **par axe** (tuple), ce qui permet de
gerer l'anisotropie (voxels plus epais en z) et le calcul "par coupe 2D"
(sigma nul sur l'axe z).

Peripherique (variable d'environnement ``XWI_DEVICE``) :
  * ``auto`` (defaut) : GPU via pyclesperanto si disponible, sinon CPU ;
  * ``cpu`` : force le CPU (scipy / scikit-image) ;
  * ``gpu`` : demande le GPU (repli CPU avec avertissement si absent).

Le GPU n'est utilise que pour les operations ou il est clairement benefique
(lissage gaussien, min/max/median). En cas d'erreur GPU sur une operation,
on bascule definitivement sur le CPU pour celle-ci.
"""
from __future__ import annotations

import importlib
import logging
import os
from functools import lru_cache

import numpy as np
from scipy import ndimage as ndi

logger = logging.getLogger(__name__)

_device = os.environ.get("XWI_DEVICE", "auto").lower()
if os.environ.get("SEGFOREST_FORCE_CPU", "0") == "1":  # compat v0.1
    _device = "cpu"

_cle = None
_cle_checked = False
_gpu_failed: set[str] = set()


# --------------------------------------------------------------------------
# Selection du peripherique
# --------------------------------------------------------------------------
def set_device(name: str) -> None:
    """Choisit ``"auto"``, ``"cpu"`` ou ``"gpu"`` (a appeler avant les calculs)."""
    global _device, _cle, _cle_checked
    name = name.lower()
    if name not in ("auto", "cpu", "gpu"):
        raise ValueError("device doit etre 'auto', 'cpu' ou 'gpu'")
    _device = name
    _cle, _cle_checked = None, False
    _gpu_failed.clear()


def _load_cle():
    """Charge pyclesperanto (nouvelle API) ou pyclesperanto_prototype, paresseusement."""
    global _cle, _cle_checked
    if _cle_checked:
        return _cle
    _cle_checked = True
    if _device == "cpu":
        return None
    for modname in ("pyclesperanto", "pyclesperanto_prototype"):
        try:
            mod = importlib.import_module(modname)
            probe = np.zeros((4, 4), np.float32)
            mod.pull(mod.push(probe))  # verifie qu'un device OpenCL repond
            _cle = mod
            logger.info("Backend GPU actif (%s).", modname)
            break
        except Exception as exc:  # pragma: no cover - depend du materiel
            logger.debug("GPU via %s indisponible : %s", modname, exc)
    if _cle is None and _device == "gpu":
        logger.warning("GPU demande mais indisponible : utilisation du CPU.")
    return _cle


def gpu_available() -> bool:
    return _load_cle() is not None


def backend_name() -> str:
    return "GPU (pyclesperanto)" if gpu_available() else "CPU (scipy/scikit-image)"


def _gpu_call(op: str, image: np.ndarray, **kwargs):
    """Execute ``cle.<op>`` ; renvoie None si le GPU n'est pas utilisable."""
    cle = _load_cle()
    if cle is None or op in _gpu_failed:
        return None
    try:
        fn = getattr(cle, op)
        out = fn(cle.push(np.ascontiguousarray(image, dtype=np.float32)), **kwargs)
        return np.asarray(cle.pull(out), dtype=np.float32)
    except Exception as exc:  # pragma: no cover - depend du materiel
        _gpu_failed.add(op)
        logger.warning("Operation GPU %s en echec (%s) : repli CPU.", op, exc)
        return None


def _xyz_kwargs(prefix: str, values) -> dict:
    """Convertit un tuple par axe (z,y,x) ou (y,x) en kwargs clesperanto."""
    vals = list(values)
    if len(vals) == 2:
        vals = [0.0] + vals
    z, y, x = vals
    return {f"{prefix}_x": float(x), f"{prefix}_y": float(y), f"{prefix}_z": float(z)}


# --------------------------------------------------------------------------
# Filtres gaussiens
# --------------------------------------------------------------------------
def kernel_radius(sigma: float, truncate: float) -> int:
    """Rayon (en pixels) du noyau gaussien de scipy pour ``sigma``."""
    return int(truncate * float(sigma) + 0.5)


def gaussian(image: np.ndarray, sigma, truncate: float = 3.0, order=0) -> np.ndarray:
    """Lissage gaussien (ou derivee gaussienne si ``order`` != 0), par axe.

    ``sigma`` et ``order`` sont des scalaires ou des tuples par axe. Un sigma
    nul sur un axe laisse cet axe intact (utile pour le calcul en 2D par coupe).
    """
    sig = np.broadcast_to(np.asarray(sigma, dtype=float), (image.ndim,))
    ordr = np.broadcast_to(np.asarray(order, dtype=int), (image.ndim,))
    if not np.any(ordr) and not np.any(sig > 0):
        return np.asarray(image, dtype=np.float32)
    if not np.any(ordr) and image.ndim in (2, 3):
        out = _gpu_call("gaussian_blur", image, **_xyz_kwargs("sigma", sig))
        if out is not None:
            return out
    return ndi.gaussian_filter(np.asarray(image, dtype=np.float32), sigma=tuple(sig),
                               order=tuple(ordr), truncate=truncate,
                               output=np.float32, mode="reflect")


# --------------------------------------------------------------------------
# Morphologie mathematique en niveaux de gris
# --------------------------------------------------------------------------
@lru_cache(maxsize=64)
def ellipsoid_footprint(radii: tuple) -> np.ndarray:
    """Element structurant ellipsoidal (boule si rayons egaux) ; rayon 0 = axe ignore."""
    grids = np.ogrid[tuple(slice(-r, r + 1) for r in radii)]
    dist = np.zeros([2 * r + 1 for r in radii], dtype=float)
    for g, r in zip(grids, radii):
        if r > 0:
            dist = dist + (g / r) ** 2
    return dist <= 1.0 + 1e-9


def _decomposed_footprint(radii: tuple):
    """Decomposition rapide (sequence) d'une boule/disque isotrope, si possible."""
    active = [r for r in radii if r > 0]
    if len(set(active)) != 1 or active[0] < 3 or len(active) not in (2, 3):
        return None
    try:
        from skimage.morphology import ball, disk

        r = active[0]
        seq = ball(r, decomposition="sequence") if len(active) == 3 else \
            disk(r, decomposition="sequence")
        if len(active) == len(radii):
            return seq
        # rayon nul sur un axe (calcul 2D par coupe) : on insere un axe singleton
        axis = radii.index(0)
        return tuple((np.expand_dims(fp, axis), n) for fp, n in seq)
    except Exception:  # pragma: no cover - vieilles versions de scikit-image
        return None


def _min_max(image: np.ndarray, radii: tuple, shape: str, op: str) -> np.ndarray:
    radii = tuple(int(r) for r in radii)
    img = np.asarray(image, dtype=np.float32)
    if not any(radii):
        return img
    if shape == "box":
        size = tuple(2 * r + 1 for r in radii)
        f = ndi.minimum_filter if op == "min" else ndi.maximum_filter
        return f(img, size=size, mode="reflect")
    active = [r for r in radii if r > 0]
    if len(set(active)) == 1 and img.ndim in (2, 3):
        out = _gpu_call("minimum_sphere" if op == "min" else "maximum_sphere", img,
                        **_xyz_kwargs("radius", radii))
        if out is not None:
            return out
    seq = _decomposed_footprint(radii)
    if seq is not None:
        from skimage import morphology

        f = morphology.erosion if op == "min" else morphology.dilation
        return np.asarray(f(img, seq), dtype=np.float32)
    fp = ellipsoid_footprint(radii)
    f = ndi.grey_erosion if op == "min" else ndi.grey_dilation
    return f(img, footprint=fp, mode="reflect").astype(np.float32, copy=False)


def erosion(image, radii, shape: str = "ball"):
    return _min_max(image, radii, shape, "min")


def dilation(image, radii, shape: str = "ball"):
    return _min_max(image, radii, shape, "max")


def median(image, radii, shape: str = "ball"):
    radii = tuple(int(r) for r in radii)
    img = np.asarray(image, dtype=np.float32)
    if not any(radii):
        return img
    active = [r for r in radii if r > 0]
    if len(set(active)) == 1 and img.ndim in (2, 3):
        out = _gpu_call("median_sphere", img, **_xyz_kwargs("radius", radii))
        if out is not None:
            return out
    if shape == "box":
        return ndi.median_filter(img, size=tuple(2 * r + 1 for r in radii), mode="reflect")
    return ndi.median_filter(img, footprint=ellipsoid_footprint(radii), mode="reflect")


def morpho_radius_halo(radius: int, shape: str) -> int:
    """Portee effective d'une erosion/dilatation (la decomposition peut deborder de 1)."""
    return int(radius) + (1 if shape == "ball" and radius >= 3 else 0)
