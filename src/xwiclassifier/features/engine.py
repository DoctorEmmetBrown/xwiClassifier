"""Moteur de features xwi : image (2D/3D, mono ou multicanal) -> pile de features.

Points cles (v0.2) :
  * catalogue ouvert : familles "lisses" (parametre = sigma, en pixels de l'axe
    le plus fin) et morphologiques (parametre = rayon) ;
  * derivees gaussiennes exactes (et non differences finies sur l'image lissee) ;
  * cache partage par echelle : lissage, derivees 1 et 2, valeurs propres sont
    calculees une seule fois et reutilisees par toutes les familles ;
  * valeurs propres analytiques (2x2 / 3x3) ;
  * calcul multi-thread (une tache par echelle / par rayon) ecrit directement
    dans le tableau de sortie (pas de copie np.stack) ;
  * anisotropie (``spacing``) et calcul "2D par coupe" (``compute_in_2d``) ;
  * normalisation d'intensite calculee une fois sur l'image entiere (et non par
    bloc), pour des features coherentes entre blocs et entre images ;
  * ``halo()`` donne la marge exacte necessaire au calcul par blocs.
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .. import backend as bk
from .eigen import eig_sym

DEFAULT_SCALES = (0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10.0)
DEFAULT_RADII = (1, 2, 3, 5, 8)
DERIVATIVE_MIN_SIGMA = 0.5  # sigma effectif minimal des filtres derivatifs
DOG_RATIO = 1.6
NORMALIZATIONS = ("none", "minmax", "percentile", "zscore")


# --------------------------------------------------------------------------
# Registre des familles
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Family:
    name: str
    kind: str                 # "raw" | "smooth" | "morpho" | "custom"
    channels: str             # "1", "2" ou "ndim" (une valeur par axe spatial)
    min_scale: float          # plus petite echelle proposee dans l'interface
    description: str
    fn: Callable | None = None  # familles personnalisees uniquement

    def n_channels(self, n_axes: int) -> int:
        if self.channels == "ndim":
            return n_axes
        return int(self.channels)


FAMILIES: dict[str, Family] = {}


def _reg(name, kind, channels, min_scale, description, fn=None):
    FAMILIES[name] = Family(name, kind, channels, min_scale, description, fn)


_reg("intensity", "raw", "1", 0.0, "Intensite brute (apres normalisation)")
_reg("gaussian", "smooth", "1", 0.3, "Lissage gaussien")
_reg("gradient", "smooth", "1", 0.7, "Norme du gradient gaussien (contours)")
_reg("log", "smooth", "1", 0.7, "Laplacien de gaussienne (blobs, contours)")
_reg("dog", "smooth", "1", 0.7, "Difference de gaussiennes (sigma, 1.6 sigma)")
_reg("hessian_eigen", "smooth", "ndim", 0.7, "Valeurs propres de la Hessienne (structures)")
_reg("structure_eigen", "smooth", "ndim", 0.7, "Valeurs propres du tenseur de structure (texture)")
_reg("variance", "smooth", "1", 0.7, "Ecart-type local pondere (texture)")
_reg("ridge", "smooth", "2", 0.7, "Tubularite claire / sombre (fils, fibres) depuis la Hessienne")
_reg("erosion", "morpho", "1", 1, "Erosion (minimum local)")
_reg("dilation", "morpho", "1", 1, "Dilatation (maximum local)")
_reg("opening", "morpho", "1", 1, "Ouverture : supprime les petits objets clairs")
_reg("closing", "morpho", "1", 1, "Fermeture : bouche les petits trous sombres")
_reg("morph_gradient", "morpho", "1", 1, "Gradient morphologique (dilatation - erosion)")
_reg("white_tophat", "morpho", "1", 1, "Top-hat blanc : petits objets clairs")
_reg("black_tophat", "morpho", "1", 1, "Top-hat noir : petits objets sombres")
_reg("median", "morpho", "1", 1, "Filtre median (lent pour les grands rayons)")

SMOOTH_FAMILIES = tuple(n for n, f in FAMILIES.items() if f.kind == "smooth")
MORPHO_FAMILIES = frozenset(n for n, f in FAMILIES.items() if f.kind == "morpho")
DEFAULT_FAMILIES = ("gaussian", "log", "gradient", "dog", "structure_eigen",
                    "hessian_eigen", "variance")


def register_family(name: str, fn: Callable, channels: int = 1, min_scale: float = 0.0,
                    description: str = "", halo: float = 4.0) -> None:
    """Ajoute une famille personnalisee ``fn(image, sigma_par_axe) -> array | list``.

    ``halo`` : portee du filtre en multiples de sigma (pour le calcul par blocs).
    La famille doit etre re-enregistree avant de recharger un modele qui l'utilise.
    """
    FAMILIES[name] = Family(name, "custom", str(int(channels)), min_scale,
                            description or name, fn)
    _CUSTOM_HALO[name] = float(halo)


_CUSTOM_HALO: dict[str, float] = {}


# --------------------------------------------------------------------------
# Specification
# --------------------------------------------------------------------------
@dataclass
class FeatureSpec:
    """Specification reproductible (et serialisable) d'un jeu de features.

    ``pairs`` : couples (famille, parametre). Le parametre est un sigma pour
    les familles lisses et un rayon entier pour les familles morphologiques.
    """

    pairs: tuple = field(default_factory=lambda: FeatureSpec.from_grid().pairs)
    spacing: tuple | None = None
    compute_in_2d: bool = False
    normalize: str = "none"
    truncate: float = 3.0
    morpho_shape: str = "ball"

    def __post_init__(self):
        self.pairs = tuple((str(f), float(p)) for f, p in self.pairs)
        unknown = sorted({f for f, _ in self.pairs if f not in FAMILIES})
        if unknown:
            raise ValueError(f"Familles de features inconnues : {unknown}")
        if self.normalize not in NORMALIZATIONS:
            raise ValueError(f"normalize doit etre parmi {NORMALIZATIONS}")
        if self.morpho_shape not in ("ball", "box"):
            raise ValueError("morpho_shape doit etre 'ball' ou 'box'")
        if self.spacing is not None:
            self.spacing = tuple(float(s) for s in self.spacing)
        # dedoublonnage en conservant l'ordre
        seen, uniq = set(), []
        for pr in self.pairs:
            if pr not in seen:
                seen.add(pr)
                uniq.append(pr)
        self.pairs = tuple(uniq)

    # ------------------------------------------------------------------
    @classmethod
    def from_grid(cls, families=DEFAULT_FAMILIES, scales=DEFAULT_SCALES,
                  radii=DEFAULT_RADII, **kwargs) -> "FeatureSpec":
        """Produit cartesien familles x echelles (en respectant les echelles minimales)."""
        pairs = []
        for fam in families:
            info = FAMILIES[fam]
            if info.kind == "raw":
                pairs.append((fam, 0.0))
            elif info.kind == "morpho":
                pairs.extend((fam, float(r)) for r in radii if r >= 1)
            else:
                pairs.extend((fam, float(s)) for s in scales if s >= info.min_scale)
        return cls(pairs=tuple(pairs), **kwargs)

    @property
    def families(self) -> tuple:
        return tuple(dict.fromkeys(f for f, _ in self.pairs))

    def to_dict(self) -> dict:
        return {"type": "xwi", "pairs": [list(p) for p in self.pairs],
                "spacing": list(self.spacing) if self.spacing else None,
                "compute_in_2d": self.compute_in_2d, "normalize": self.normalize,
                "truncate": self.truncate, "morpho_shape": self.morpho_shape}

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureSpec":
        d = {k: v for k, v in d.items() if k != "type"}
        d["pairs"] = tuple(tuple(p) for p in d.get("pairs", ()))
        return cls(**d)


# --------------------------------------------------------------------------
# Normalisation (parametres calcules UNE fois sur l'image complete)
# --------------------------------------------------------------------------
def fit_normalization(image: np.ndarray, method: str, max_samples: int = 2_000_000) -> dict:
    """Calcule les parametres de normalisation d'une image mono-canal."""
    if method == "none":
        return {"method": "none"}
    flat = np.asarray(image).ravel()
    if flat.size > max_samples:
        idx = np.random.default_rng(0).choice(flat.size, max_samples, replace=False)
        flat = flat[idx]
    flat = flat.astype(np.float64)
    if method == "minmax":
        a, b = float(flat.min()), float(flat.max())
    elif method == "percentile":
        a, b = (float(v) for v in np.percentile(flat, [0.5, 99.5]))
    elif method == "zscore":
        mu, sd = float(flat.mean()), float(flat.std())
        a, b = mu, mu + (sd if sd > 0 else 1.0)
    else:
        raise ValueError(method)
    if b <= a:
        b = a + 1.0
    return {"method": method, "offset": a, "scale": b - a}


def apply_normalization(image: np.ndarray, params: dict | None) -> np.ndarray:
    img = np.asarray(image, dtype=np.float32)
    if not params or params.get("method", "none") == "none":
        return img
    return (img - np.float32(params["offset"])) / np.float32(params["scale"])


# --------------------------------------------------------------------------
# Contextes de calcul (caches partages)
# --------------------------------------------------------------------------
class _SmoothCtx:
    def __init__(self, img, factors, truncate):
        self.img = img
        self.f = factors
        self.t = truncate
        self.axes = tuple(i for i, f in enumerate(factors) if f > 0)
        self._c: dict = {}

    def sig(self, s):
        return tuple(s * f for f in self.f)

    def _get(self, key, fn):
        if key not in self._c:
            self._c[key] = fn()
        return self._c[key]

    def smooth(self, s):
        return self._get(("g", s), lambda: bk.gaussian(self.img, self.sig(s), self.t))

    def d1(self, s, i):
        order = [0] * self.img.ndim
        order[i] = 1
        return self._get(("d1", s, i), lambda: bk.gaussian(self.img, self.sig(s), self.t, order))

    def d2(self, s, i, j):
        order = [0] * self.img.ndim
        order[i] += 1
        order[j] += 1
        key = ("d2", s) + tuple(sorted((i, j)))
        return self._get(key, lambda: bk.gaussian(self.img, self.sig(s), self.t, order))

    def hessian_eigs(self, s):
        def run():
            comps = {(i, j): self.d2(s, i, j) for a, i in enumerate(self.axes)
                     for j in self.axes[a:]}
            return eig_sym(comps, self.axes)
        return self._get(("heig", s), run)

    def structure_eigs(self, s):
        sd = max(0.5 * s, DERIVATIVE_MIN_SIGMA)

        def run():
            comps = {}
            for a, i in enumerate(self.axes):
                for j in self.axes[a:]:
                    prod = self.d1(sd, i) * self.d1(sd, j)
                    comps[(i, j)] = bk.gaussian(prod, self.sig(s), self.t)
            return eig_sym(comps, self.axes)
        return self._get(("seig", s), run)

    # -- familles -------------------------------------------------------
    def compute(self, fam: str, s: float) -> list:
        se = max(s, DERIVATIVE_MIN_SIGMA)
        if fam == "intensity":
            return [self.img]
        if fam == "gaussian":
            return [self.smooth(s)]
        if fam == "gradient":
            acc = np.zeros(self.img.shape, np.float32)
            for i in self.axes:
                acc += self.d1(se, i) ** 2
            return [np.sqrt(acc)]
        if fam == "log":
            acc = np.zeros(self.img.shape, np.float32)
            for i in self.axes:
                acc += self.d2(se, i, i)
            return [acc]
        if fam == "dog":
            return [self.smooth(se) - self.smooth(DOG_RATIO * se)]
        if fam == "hessian_eigen":
            return list(self.hessian_eigs(se))
        if fam == "structure_eigen":
            return list(self.structure_eigs(se))
        if fam == "variance":
            g = self.smooth(se)
            g2 = bk.gaussian(self.img * self.img, self.sig(se), self.t)
            return [np.sqrt(np.maximum(g2 - g * g, 0.0))]
        if fam == "ridge":
            ev = self.hessian_eigs(se)
            if len(ev) == 3:
                l1, l2, l3 = ev
                bright = np.where((l2 < 0) & (l3 < 0), np.sqrt(np.abs(l2 * l3)), 0.0)
                dark = np.where((l1 > 0) & (l2 > 0), np.sqrt(np.abs(l1 * l2)), 0.0)
            else:
                l1, l2 = ev[0], ev[-1]
                bright, dark = np.maximum(-l2, 0.0), np.maximum(l1, 0.0)
            return [bright.astype(np.float32), dark.astype(np.float32)]
        info = FAMILIES[fam]
        if info.kind == "custom":
            res = info.fn(self.img, self.sig(s))
            return [np.asarray(r, np.float32) for r in (res if isinstance(res, (list, tuple)) else [res])]
        raise KeyError(fam)


class _MorphoCtx:
    def __init__(self, img, radii, shape):
        self.img, self.r, self.shape = img, radii, shape
        self._c: dict = {}

    def _get(self, key, fn):
        if key not in self._c:
            self._c[key] = fn()
        return self._c[key]

    def ero(self):
        return self._get("ero", lambda: bk.erosion(self.img, self.r, self.shape))

    def dil(self):
        return self._get("dil", lambda: bk.dilation(self.img, self.r, self.shape))

    def opening(self):
        return self._get("open", lambda: bk.dilation(self.ero(), self.r, self.shape))

    def closing(self):
        return self._get("close", lambda: bk.erosion(self.dil(), self.r, self.shape))

    def compute(self, fam: str) -> list:
        if fam == "erosion":
            return [self.ero()]
        if fam == "dilation":
            return [self.dil()]
        if fam == "opening":
            return [self.opening()]
        if fam == "closing":
            return [self.closing()]
        if fam == "morph_gradient":
            return [self.dil() - self.ero()]
        if fam == "white_tophat":
            return [self.img - self.opening()]
        if fam == "black_tophat":
            return [self.closing() - self.img]
        if fam == "median":
            return [bk.median(self.img, self.r, self.shape)]
        raise KeyError(fam)


# --------------------------------------------------------------------------
# Moteur
# --------------------------------------------------------------------------
class FeatureEngine:
    """Calcule la pile de features (``(*spatial, n_features)`` float32)."""

    kind = "xwi"

    def __init__(self, spec: FeatureSpec | dict | None = None, n_threads: int | None = None):
        if isinstance(spec, dict):
            spec = FeatureSpec.from_dict(spec)
        self.spec = spec or FeatureSpec()
        self.n_threads = n_threads or min(os.cpu_count() or 1, 8)

    # -- geometrie ------------------------------------------------------
    def _factors(self, ndim: int) -> tuple:
        sp = self.spec.spacing
        if sp is None or len(sp) != ndim:
            f = [1.0] * ndim
        else:
            m = min(sp)
            f = [m / s for s in sp]
        if self.spec.compute_in_2d and ndim == 3:
            f[0] = 0.0
        return tuple(f)

    def _radii(self, r: float, ndim: int) -> tuple:
        return tuple(int(round(r * f)) for f in self._factors(ndim))

    def _n_axes(self, ndim: int) -> int:
        return sum(1 for f in self._factors(ndim) if f > 0)

    def channel_names(self, ndim: int, n_channels: int = 1) -> list:
        n_axes = self._n_axes(ndim)
        names = []
        for c in range(n_channels):
            pre = f"c{c}_" if n_channels > 1 else ""
            for fam, p in self.spec.pairs:
                base = f"{pre}{fam}" if fam == "intensity" else f"{pre}{fam}_s{p:g}"
                n = FAMILIES[fam].n_channels(n_axes)
                if fam == "ridge":
                    names += [base + "_bright", base + "_dark"]
                elif n == 1:
                    names.append(base)
                else:
                    names += [f"{base}_e{k}" for k in range(n)]
        return names

    def n_features(self, ndim: int, n_channels: int = 1) -> int:
        return len(self.channel_names(ndim, n_channels))

    def halo(self, ndim: int) -> tuple:
        """Marge (par axe) assurant un calcul par blocs identique au calcul global."""
        f = self._factors(ndim)
        t = max(self.spec.truncate, 4.0) if bk.gpu_available() else self.spec.truncate
        halo = [0] * ndim

        def kr(s):
            return [bk.kernel_radius(s * fi, t) for fi in f]

        for fam, p in self.spec.pairs:
            info = FAMILIES[fam]
            se = max(p, DERIVATIVE_MIN_SIGMA)
            if info.kind == "raw":
                h = [0] * ndim
            elif fam == "gaussian":
                h = kr(p)
            elif fam == "dog":
                h = kr(DOG_RATIO * se)
            elif fam == "structure_eigen":
                h = [a + b for a, b in zip(kr(max(0.5 * se, DERIVATIVE_MIN_SIGMA)), kr(se))]
            elif info.kind == "smooth":
                h = kr(se)
            elif info.kind == "morpho":
                rr = [bk.morpho_radius_halo(r, self.spec.morpho_shape) for r in self._radii(p, ndim)]
                mult = 2 if fam in ("opening", "closing", "white_tophat", "black_tophat") else 1
                h = [mult * r for r in rr]
            else:  # custom
                h = [int(math.ceil(_CUSTOM_HALO.get(fam, 4.0) * p * fi)) for fi in f]
            halo = [max(a, b) for a, b in zip(halo, h)]
        return tuple(h + 1 for h in halo)

    # -- normalisation --------------------------------------------------
    def fit_normalization(self, image: np.ndarray, channel_axis: int | None = None) -> list:
        img = _channels_last(image, channel_axis)
        return [fit_normalization(img[..., c], self.spec.normalize) for c in range(img.shape[-1])]

    # -- calcul ---------------------------------------------------------
    def compute(self, image: np.ndarray, norm: list | None = None,
                channel_axis: int | None = None) -> np.ndarray:
        img = _channels_last(image, channel_axis)
        n_ch = img.shape[-1]
        ndim = img.ndim - 1
        if norm is None:
            norm = self.fit_normalization(img, channel_axis=-1)
        factors = self._factors(ndim)
        n_axes = self._n_axes(ndim)
        sizes = [FAMILIES[fam].n_channels(n_axes) for fam, _ in self.spec.pairs]
        total = sum(sizes) * n_ch
        out = np.empty(img.shape[:-1] + (total,), dtype=np.float32)

        tasks = []
        offset = 0
        for c in range(n_ch):
            chan = np.ascontiguousarray(apply_normalization(img[..., c], norm[c]))
            groups: dict = {}
            for (fam, p), n in zip(self.spec.pairs, sizes):
                info = FAMILIES[fam]
                key = ("m", p) if info.kind == "morpho" else ("s", p)
                groups.setdefault(key, []).append((fam, offset, n))
                offset += n
            for (kind, p), items in groups.items():
                tasks.append((kind, p, chan, items))

        def run(task):
            kind, p, chan, items = task
            if kind == "m":
                ctx = _MorphoCtx(chan, self._radii(p, ndim), self.spec.morpho_shape)
                for fam, off, _n in items:
                    for k, arr in enumerate(ctx.compute(fam)):
                        out[..., off + k] = arr
            else:
                ctx = _SmoothCtx(chan, factors, self.spec.truncate)
                for fam, off, _n in items:
                    for k, arr in enumerate(ctx.compute(fam, p)):
                        out[..., off + k] = arr

        if self.n_threads > 1 and len(tasks) > 1:
            with ThreadPoolExecutor(self.n_threads) as ex:
                for _ in ex.map(run, tasks):
                    pass
        else:
            for task in tasks:
                run(task)
        return out

    def to_dict(self) -> dict:
        return self.spec.to_dict()


def _channels_last(image: np.ndarray, channel_axis: int | None) -> np.ndarray:
    img = np.asarray(image)
    if channel_axis is None:
        return img[..., None]
    return np.moveaxis(img, channel_axis, -1)
