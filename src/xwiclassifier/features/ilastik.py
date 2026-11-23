"""Reproduction fidele des features de pixel classification d'ilastik.

Pipeline ilastik (lazyflow ``OpPixelFeaturesPresmoothed``), reproduit ici :
  * pre-lissage gaussien par echelle (fenetre 3.5 sigma) avec
    sigma_pre = sqrt(s^2 - 1) si s > 1, sinon sigma_pre = s ;
  * puis filtre de l'operateur a l'echelle s_op = 1 si s > 1, sinon s
    (fenetre 2 sigma) ;
  * DoG : G(s_op) - G(0.66 s_op) ; tenseur de structure : derivee a s_op,
    integration a 0.5 s_op ; valeurs propres par ordre decroissant ;
  * pas de normalisation : intensites brutes en float32 ;
  * ordre des canaux : feature, puis echelle, puis canal d'entree, puis composante.
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .. import backend as bk
from .eigen import eig_sym

ILASTIK_FEATURES = ("GaussianSmoothing", "LaplacianOfGaussian", "GaussianGradientMagnitude",
                    "DifferenceOfGaussians", "StructureTensorEigenvalues",
                    "HessianOfGaussianEigenvalues")
DISPLAY_NAMES = {
    "GaussianSmoothing": "Gaussian Smoothing",
    "LaplacianOfGaussian": "Laplacian of Gaussian",
    "GaussianGradientMagnitude": "Gaussian Gradient Magnitude",
    "DifferenceOfGaussians": "Difference of Gaussians",
    "StructureTensorEigenvalues": "Structure Tensor Eigenvalues",
    "HessianOfGaussianEigenvalues": "Hessian of Gaussian Eigenvalues",
}
EIGEN = ("StructureTensorEigenvalues", "HessianOfGaussianEigenvalues")
DEFAULT_ILASTIK_SCALES = (0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10.0)
WINDOW_FEATURE = 2.0
WINDOW_SMOOTHER = 3.5
MODE = "mirror"  # BORDER_TREATMENT_REFLECT de vigra


class IlastikFeatures:
    kind = "ilastik"

    def __init__(self, feature_ids, scales, selection_matrix, compute_in_2d=None,
                 n_threads: int | None = None):
        self.feature_ids = [str(f) for f in feature_ids]
        unknown = [f for f in self.feature_ids if f not in ILASTIK_FEATURES]
        if unknown:
            raise ValueError(f"Features ilastik non supportees : {unknown}")
        self.scales = [float(s) for s in scales]
        self.matrix = np.asarray(selection_matrix, dtype=bool)
        if self.matrix.shape != (len(self.feature_ids), len(self.scales)):
            raise ValueError("Matrice de selection incoherente avec features/echelles.")
        if compute_in_2d is None:
            compute_in_2d = [False] * len(self.scales)
        self.compute_in_2d = [bool(v) for v in compute_in_2d]
        self.n_threads = n_threads or min(os.cpu_count() or 1, 8)

    # ------------------------------------------------------------------
    @classmethod
    def all_features(cls, scales=DEFAULT_ILASTIK_SCALES, compute_in_2d=None):
        """Selection "tout coche" d'ilastik (lissage des 0.3, derivees des 0.7)."""
        m = np.ones((len(ILASTIK_FEATURES), len(scales)), bool)
        for j, s in enumerate(scales):
            if s < 0.7:
                m[1:, j] = False
        return cls(ILASTIK_FEATURES, scales, m, compute_in_2d)

    def _selected(self):
        return [(i, j) for i in range(len(self.feature_ids)) for j in range(len(self.scales))
                if self.matrix[i, j]]

    def _in2d(self, j: int, ndim: int) -> bool:
        return ndim == 3 and self.compute_in_2d[j]

    def _n_comp(self, fid: str, j: int, ndim: int) -> int:
        if fid in EIGEN:
            return 2 if (ndim == 2 or self._in2d(j, ndim)) else 3
        return 1

    def channel_names(self, ndim: int, n_channels: int = 1) -> list:
        names = []
        for i, j in self._selected():
            fid = self.feature_ids[i]
            base = f"{DISPLAY_NAMES[fid]} (σ={self.scales[j]})"
            if self.compute_in_2d[j]:
                base += " in 2D"
            n = self._n_comp(fid, j, ndim) * n_channels
            names += [base] if n == 1 else [f"{base} [{k}]" for k in range(n)]
        return names

    def n_features(self, ndim: int, n_channels: int = 1) -> int:
        return len(self.channel_names(ndim, n_channels))

    def halo(self, ndim: int) -> tuple:
        h = [0] * ndim
        for i, j in self._selected():
            s = self.scales[j]
            pre, s_op = self._sigmas(s)
            r = bk.kernel_radius(pre, WINDOW_SMOOTHER) + bk.kernel_radius(s_op, WINDOW_FEATURE)
            if self.feature_ids[i] == "StructureTensorEigenvalues":
                r += bk.kernel_radius(0.5 * s_op, WINDOW_FEATURE)
            for a in range(ndim):
                if a == 0 and self._in2d(j, ndim):
                    continue
                h[a] = max(h[a], r)
        return tuple(x + 1 for x in h)

    @staticmethod
    def _sigmas(s: float):
        return (math.sqrt(s * s - 1.0), 1.0) if s > 1.0 else (s, s)

    def fit_normalization(self, image, channel_axis=None) -> list:
        n = 1 if channel_axis is None else np.asarray(image).shape[channel_axis]
        return [{"method": "none"}] * n

    # ------------------------------------------------------------------
    def compute(self, image: np.ndarray, norm=None, channel_axis: int | None = None) -> np.ndarray:
        img = np.asarray(image)
        img = img[..., None] if channel_axis is None else np.moveaxis(img, channel_axis, -1)
        n_ch, ndim = img.shape[-1], img.ndim - 1
        sel = self._selected()
        offsets, off = {}, 0
        for i, j in sel:
            offsets[(i, j)] = off
            off += self._n_comp(self.feature_ids[i], j, ndim) * n_ch
        out = np.empty(img.shape[:-1] + (off,), np.float32)
        by_scale: dict = {}
        for i, j in sel:
            by_scale.setdefault(j, []).append(i)
        tasks = [(j, c, feats) for j, feats in by_scale.items() for c in range(n_ch)]

        def run(task):
            j, c, feats = task
            in2d = self._in2d(j, ndim)
            axes = tuple(range(1, ndim)) if in2d else tuple(range(ndim))

            def sig(v):
                return tuple(0.0 if (in2d and a == 0) else v for a in range(ndim))

            pre_s, s_op = self._sigmas(self.scales[j])
            pre = bk.gaussian(np.asarray(img[..., c], np.float32), sig(pre_s), WINDOW_SMOOTHER, mode=MODE)
            cache: dict = {}

            def d(order):
                if order not in cache:
                    cache[order] = bk.gaussian(pre, sig(s_op), WINDOW_FEATURE, order, mode=MODE)
                return cache[order]

            def unit(*ax):
                o = [0] * ndim
                for a in ax:
                    o[a] += 1
                return tuple(o)

            for i in feats:
                fid = self.feature_ids[i]
                ncomp = self._n_comp(fid, j, ndim)
                if fid == "GaussianSmoothing":
                    res = [d(unit())]
                elif fid == "LaplacianOfGaussian":
                    res = [sum(d(unit(a, a)) for a in axes)]
                elif fid == "GaussianGradientMagnitude":
                    res = [np.sqrt(sum(d(unit(a)) ** 2 for a in axes))]
                elif fid == "DifferenceOfGaussians":
                    res = [d(unit()) - bk.gaussian(pre, sig(0.66 * s_op), WINDOW_FEATURE, mode=MODE)]
                elif fid == "HessianOfGaussianEigenvalues":
                    comps = {(a, b): d(unit(a, b)) for k, a in enumerate(axes) for b in axes[k:]}
                    res = eig_sym(comps, axes)
                else:  # StructureTensorEigenvalues
                    comps = {(a, b): bk.gaussian(d(unit(a)) * d(unit(b)), sig(0.5 * s_op), WINDOW_FEATURE, mode=MODE)
                             for k, a in enumerate(axes) for b in axes[k:]}
                    res = eig_sym(comps, axes)
                base = offsets[(i, j)] + c * ncomp
                for k, arr in enumerate(res):
                    out[..., base + k] = arr

        if self.n_threads > 1 and len(tasks) > 1:
            with ThreadPoolExecutor(self.n_threads) as ex:
                list(ex.map(run, tasks))
        else:
            for t in tasks:
                run(t)
        return out

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"type": "ilastik", "feature_ids": self.feature_ids, "scales": self.scales,
                "selection_matrix": self.matrix.astype(int).tolist(),
                "compute_in_2d": self.compute_in_2d}

    @classmethod
    def from_dict(cls, d: dict) -> "IlastikFeatures":
        return cls(d["feature_ids"], d["scales"], d["selection_matrix"], d.get("compute_in_2d"))
