"""Reproduction des features de "Smart Segmentation" d'IPSDK Explorer (Reactiv'IP).

Specification reconstituee a partir du code d'IPSDK Explorer 4.0
(``moduleRandomForest``) et de la documentation IPSDK :

* vecteur de features = image originale (float32, sans normalisation), puis
  pour chaque resolution, chaque filtre (ordre du fichier Settings.mho) et
  chaque taille cochee, la reponse du filtre ; un canal d'entree a la suite de
  l'autre a l'interieur de chaque filtre ;
* ``taille`` s = demi-taille de noyau (le nom affiche est "(2s+1)x(2s+1)") :
    - Gaussian              : lissage gaussien, ecart-type s ; demi-noyau = plus
                              petit n couvrant 99.7 % de la gaussienne (min 2) ;
    - Mean                  : moyenne sur une boite (2s+1)^d ;
    - Laplacian Difference  : G(s) - G(1.6 s) ;
    - High Pass             : noyau (2s+1)^d de -1, centre (2s+1)^d, bords miroir ;
    - Variance              : variance (de population) sur une boite (2s+1)^d ;
    - Erosion / Dilation / Opening / Closing : element structurant disque
      (2D) ou boule (3D) de rayon s, defini par x^2 + y^2 (+ z^2) <= s^2 ;
* ``Dimension`` = "2D" sur un volume : filtres appliques coupe par coupe ;
* multi-resolution r > 1 : reduction lineaire de l'image (taille int(n/r + 0.5)),
  filtre, puis agrandissement lineaire a la taille d'origine.

Deux details non documentes (normalisation du noyau gaussien tronque,
convention de grille du zoom) sont parametrables et peuvent etre calibres
automatiquement sur les annotations stockees avec le modele.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
from scipy import ndimage as ndi

from ..backend import ellipsoid_footprint

IPSDK_FILTERS = ("Gaussian", "Mean", "Laplacian Difference", "High Pass", "Variance")
IPSDK_MORPHO = ("Erosion", "Dilation", "Opening", "Closing")
IPSDK_ALL = IPSDK_FILTERS + IPSDK_MORPHO
GAUSSIAN_RATIO = 0.997
GAUSSIAN_MIN_HALF = 2
DOG_FACTOR = 1.6
COLORS = ("Red", "Green", "Blue")


@lru_cache(maxsize=256)
def ipsdk_gaussian_kernel(sigma: float, normalize: bool = True,
                          ratio: float = GAUSSIAN_RATIO, min_half: int = GAUSSIAN_MIN_HALF) -> np.ndarray:
    """Noyau gaussien 1D IPSDK : taille choisie par la couverture de la distribution."""
    sigma = float(sigma)
    if sigma <= 0:
        return np.ones(1)
    big = int(12 * sigma) + 12
    o = np.arange(-big, big + 1, dtype=np.float64)
    g = np.exp(-0.5 * (o / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
    total = g.sum()
    half = 0
    while g[big - half: big + half + 1].sum() < ratio * total and half < big:
        half += 1
    half = max(min_half, half)
    k = g[big - half: big + half + 1].copy()
    if normalize:
        k /= k.sum()
    return k


def _zoom_linear(a: np.ndarray, shape: tuple, axes: tuple, mode: str) -> np.ndarray:
    """Interpolation lineaire separable vers ``shape`` sur les axes ``axes``."""
    out = a
    for ax in axes:
        n, m = out.shape[ax], shape[ax]
        if n == m:
            continue
        if mode == "center":
            pos = (np.arange(m) + 0.5) * (n / m) - 0.5
        else:  # "corner" : x_in = x_out * n / m (convention du plus proche voisin IPSDK)
            pos = np.arange(m) * (n / m)
        pos = np.clip(pos, 0, n - 1)
        i0 = np.floor(pos).astype(np.int64)
        i1 = np.minimum(i0 + 1, n - 1)
        w = (pos - i0).astype(np.float32)
        shp = [1] * out.ndim
        shp[ax] = m
        w = w.reshape(shp)
        out = np.take(out, i0, axis=ax) * (1 - w) + np.take(out, i1, axis=ax) * w
    return out.astype(np.float32, copy=False)


class IPSDKFeatures:
    """Extracteur reproduisant la grille de features d'IPSDK Explorer."""

    kind = "ipsdk"

    def __init__(self, grid, dimension: str = "2D", gaussian_normalize: bool = True,
                 zoom_mode: str = "corner", n_threads: int | None = None):
        self.grid = [(int(r), str(n), int(s)) for r, n, s in grid]
        bad = [n for _, n, _ in self.grid if n not in IPSDK_ALL]
        if bad:
            raise ValueError(f"Filtres IPSDK inconnus : {bad}")
        self.dimension = "3D" if str(dimension).upper() == "3D" else "2D"
        self.gaussian_normalize = bool(gaussian_normalize)
        self.zoom_mode = zoom_mode
        self.n_threads = n_threads or min(os.cpu_count() or 1, 8)

    # ------------------------------------------------------------------
    @classmethod
    def from_settings(cls, sizes, flags: dict, resolutions=(1,), dimension="2D", order=None, **kw):
        """Construit la grille comme IPSDK Explorer : resolution > filtre > taille."""
        order = order or [n for n in IPSDK_ALL if n in flags]
        grid = []
        for r in resolutions:
            for name in order:
                for j, s in enumerate(sizes):
                    if j < len(flags[name]) and flags[name][j]:
                        grid.append((int(r), name, int(s)))
        return cls(grid, dimension=dimension, **kw)

    @property
    def requires_full_image(self) -> bool:
        """La multi-resolution depend de la taille globale : pas de calcul par blocs exact."""
        return any(r != 1 for r, _, _ in self.grid)

    def _active_axes(self, ndim: int) -> tuple:
        if ndim == 3 and self.dimension == "2D":
            return (1, 2)
        return tuple(range(ndim))

    def channel_names(self, ndim: int, n_channels: int = 1) -> list:
        def suffix(c):
            if n_channels == 1:
                return ""
            return f"( {COLORS[c]})" if n_channels == 3 else f" (channel {c})"

        names = []
        for c in range(n_channels):
            names.append("Original image" + ("" if n_channels == 1 else
                                              (f" ({COLORS[c]})" if n_channels == 3 else f" (channel {c})")))
        for r, name, s in self.grid:
            base = f"{name} {2 * s + 1}x{2 * s + 1}"
            if ndim == 3:
                base += " " + self.dimension
            if r != 1:
                base += f" (x 1/{r})"
            names += [base + suffix(c) for c in range(n_channels)]
        return names

    def n_features(self, ndim: int, n_channels: int = 1) -> int:
        return n_channels * (1 + len(self.grid))

    def halo(self, ndim: int) -> tuple:
        h = 0
        for _r, name, s in self.grid:
            if name == "Gaussian":
                v = len(ipsdk_gaussian_kernel(s, self.gaussian_normalize)) // 2
            elif name == "Laplacian Difference":
                v = len(ipsdk_gaussian_kernel(DOG_FACTOR * s, self.gaussian_normalize)) // 2
            elif name in ("Opening", "Closing"):
                v = 2 * s
            else:
                v = s
            h = max(h, v)
        axes = self._active_axes(ndim)
        return tuple(h + 1 if a in axes else 0 for a in range(ndim))

    def fit_normalization(self, image, channel_axis=None) -> list:
        n = 1 if channel_axis is None else np.asarray(image).shape[channel_axis]
        return [{"method": "none"}] * n

    # ------------------------------------------------------------------
    def _filter(self, a: np.ndarray, name: str, s: int, axes: tuple, cache: dict) -> np.ndarray:
        nd = a.ndim

        def gauss(sigma):
            key = ("g", sigma)
            if key not in cache:
                k = ipsdk_gaussian_kernel(sigma, self.gaussian_normalize)
                out = a
                for ax in axes:
                    out = ndi.correlate1d(out, k, axis=ax, output=np.float32, mode="mirror")
                cache[key] = out
            return cache[key]

        size = tuple(2 * s + 1 if ax in axes else 1 for ax in range(nd))
        if name == "Gaussian":
            return gauss(float(s))
        if name == "Laplacian Difference":
            return gauss(float(s)) - gauss(DOG_FACTOR * s)
        if name == "Mean":
            return ndi.uniform_filter(a, size=size, output=np.float32, mode="mirror")
        if name == "High Pass":
            n = float(np.prod(size))
            box = ndi.uniform_filter(a, size=size, output=np.float32, mode="mirror") * n
            return ((n + 1.0) * a - box).astype(np.float32)
        if name == "Variance":
            m = ndi.uniform_filter(a, size=size, output=np.float32, mode="mirror")
            m2 = ndi.uniform_filter(a * a, size=size, output=np.float32, mode="mirror")
            return np.maximum(m2 - m * m, 0.0).astype(np.float32)
        fp = ellipsoid_footprint(tuple(s if ax in axes else 0 for ax in range(nd)))
        if name == "Erosion":
            return ndi.grey_erosion(a, footprint=fp, mode="nearest")
        if name == "Dilation":
            return ndi.grey_dilation(a, footprint=fp, mode="nearest")
        if name == "Opening":
            return ndi.grey_dilation(ndi.grey_erosion(a, footprint=fp, mode="nearest"), footprint=fp, mode="nearest")
        if name == "Closing":
            return ndi.grey_erosion(ndi.grey_dilation(a, footprint=fp, mode="nearest"), footprint=fp, mode="nearest")
        raise KeyError(name)

    def compute(self, image: np.ndarray, norm=None, channel_axis: int | None = None) -> np.ndarray:
        img = np.asarray(image)
        img = img[..., None] if channel_axis is None else np.moveaxis(img, channel_axis, -1)
        n_ch, ndim = img.shape[-1], img.ndim - 1
        axes = self._active_axes(ndim)
        out = np.empty(img.shape[:-1] + (self.n_features(ndim, n_ch),), np.float32)
        chans = [np.ascontiguousarray(img[..., c], dtype=np.float32) for c in range(n_ch)]
        for c in range(n_ch):
            out[..., c] = chans[c]
        # taches groupees par (resolution, canal) : partage des images reduites et des gaussiennes
        groups: dict = {}
        for k, (r, name, s) in enumerate(self.grid):
            for c in range(n_ch):
                groups.setdefault((r, c), []).append((n_ch + k * n_ch + c, name, s))

        def run(item):
            (r, c), todo = item
            base = chans[c]
            small = base
            if r != 1:
                shape = tuple(int(n / r + 0.5) if ax in axes else n for ax, n in enumerate(base.shape))
                small = _zoom_linear(base, shape, axes, self.zoom_mode)
            cache: dict = {}
            for idx, name, s in todo:
                f = self._filter(small, name, s, axes, cache)
                if r != 1:
                    f = _zoom_linear(f, base.shape, axes, self.zoom_mode)
                out[..., idx] = f

        items = list(groups.items())
        if self.n_threads > 1 and len(items) > 1:
            with ThreadPoolExecutor(self.n_threads) as ex:
                list(ex.map(run, items))
        else:
            for it in items:
                run(it)
        return out

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"type": "ipsdk", "grid": [list(g) for g in self.grid], "dimension": self.dimension,
                "gaussian_normalize": self.gaussian_normalize, "zoom_mode": self.zoom_mode}

    @classmethod
    def from_dict(cls, d: dict) -> "IPSDKFeatures":
        return cls(d["grid"], d.get("dimension", "2D"), d.get("gaussian_normalize", True),
                   d.get("zoom_mode", "corner"))
