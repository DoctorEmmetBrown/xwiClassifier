"""Moteur de features : transforme une image en pile de canaux descriptifs.

C'est le point ou l'on depasse ilastik : le catalogue de features est
ouvert et entierement parametrable (familles, liste de sigmas, ajout de
features custom). Chaque feature produit un ou plusieurs canaux empiles
sur le dernier axe -> matrice (N_pixels, N_features) prete pour le RF.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .. import backend as bk

# Catalogue : nom de famille -> fonction(image, sigma_ou_rayon) -> ndarray ou list.
# Familles "lisses" (parametre = sigma) et familles morphologiques (parametre = rayon).
FEATURE_FAMILIES: dict[str, Callable] = {
    "intensity": lambda img, s: img.astype(np.float32) if s == 0 else bk.gaussian(img, s),
    "gaussian": bk.gaussian,
    "gradient": bk.gradient_magnitude,
    "log": bk.laplacian_of_gaussian,
    "dog": bk.difference_of_gaussians,
    "hessian_eigen": bk.hessian_eigen,
    "structure_eigen": bk.structure_tensor_eigen,
    # --- morphologie mathematique (parametre interprete comme un rayon) ---
    "erosion": bk.erosion,
    "dilation": bk.dilation,
    "opening": bk.opening,
    "closing": bk.closing,
    "morph_gradient": bk.morphological_gradient,
    "white_tophat": bk.white_tophat,
    "black_tophat": bk.black_tophat,
    "median": bk.median,
}

# Familles morphologiques : leur parametre est un rayon entier, pas un sigma.
MORPHO_FAMILIES = frozenset({
    "erosion", "dilation", "opening", "closing",
    "morph_gradient", "white_tophat", "black_tophat", "median",
})

# Echelles par defaut (en pixels). Plus de sigmas = plus de features = decision RF plus facile.
DEFAULT_SIGMAS = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0)
DEFAULT_FAMILIES = ("intensity", "gaussian", "gradient", "log", "dog", "hessian_eigen")


@dataclass
class FeatureSpec:
    """Specification reproductible d'un jeu de features."""

    families: tuple[str, ...] = DEFAULT_FAMILIES
    sigmas: tuple[float, ...] = DEFAULT_SIGMAS
    extra: list[tuple[str, Callable]] = field(default_factory=list)
    # Selection fine optionnelle : liste explicite de couples (famille, sigma).
    # Si fournie, elle prime sur le produit cartesien families x sigmas.
    pairs: tuple[tuple[str, float], ...] | None = None

    def validate(self) -> None:
        fams = [f for f, _ in self.pairs] if self.pairs else self.families
        unknown = [f for f in fams if f not in FEATURE_FAMILIES]
        if unknown:
            raise ValueError(f"Familles de features inconnues : {unknown}")

    def iter_pairs(self):
        """Genere les couples (famille, sigma) effectivement calcules."""
        if self.pairs is not None:
            yield from self.pairs
            return
        for family in self.families:
            sigmas = self.sigmas if family != "intensity" else (0.0,)
            for s in sigmas:
                yield family, s


class FeatureEngine:
    """Calcule et met en cache la pile de features pour une image."""

    def __init__(self, spec: FeatureSpec | None = None):
        self.spec = spec or FeatureSpec()
        self.spec.validate()
        self._cache: dict[tuple, np.ndarray] = {}
        self.channel_names: list[str] = []

    # --------------------------------------------------------------
    def compute(self, image: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Retourne un array (*image.shape, n_features) en float32."""
        img = image.astype(np.float32)
        if normalize:
            rng = float(img.max() - img.min())
            if rng > 0:
                img = (img - img.min()) / rng

        channels: list[np.ndarray] = []
        names: list[str] = []

        for family, s in self.spec.iter_pairs():
            fn = FEATURE_FAMILIES[family]
            key = (id(image), family, s)
            if key in self._cache:
                res = self._cache[key]
            else:
                res = fn(img, s)
                self._cache[key] = res
            if isinstance(res, list):
                for i, r in enumerate(res):
                    channels.append(r)
                    names.append(f"{family}_s{s:g}_e{i}")
            else:
                channels.append(res)
                names.append(f"{family}_s{s:g}")

        for name, fn in self.spec.extra:
            res = fn(img)
            channels.append(np.asarray(res, dtype=np.float32))
            names.append(f"custom_{name}")

        self.channel_names = names
        stack = np.stack(channels, axis=-1)
        return stack

    def clear_cache(self) -> None:
        self._cache.clear()

    @property
    def n_features(self) -> int:
        n = 0
        for family in self.spec.families:
            sigmas = self.spec.sigmas if family != "intensity" else (0.0,)
            mult = 3 if "eigen" in family else 1  # approx pour 3D
            n += len(sigmas) * mult
        return n + len(self.spec.extra)
