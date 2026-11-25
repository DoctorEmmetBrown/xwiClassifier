"""PixelClassifier : entrainement et prediction (par blocs) de classes de pixels."""
from __future__ import annotations

import logging
import re
from typing import Callable, Sequence

import numpy as np

from ..blocks import default_block_shape, iter_blocks, normalize_region, with_halo
from ..features import FeatureEngine, FeatureSpec, make_extractor
from . import estimators as est_mod
from .sampler import TrainingSampler, image_fingerprint, subsample_per_class

logger = logging.getLogger(__name__)
Progress = Callable[[int, int], None] | None


class PixelClassifier:
    """Classifieur de pixels/voxels base sur des features + un ensemble d'arbres.

    Parameters
    ----------
    features : FeatureSpec | dict | extracteur, optionnel
        Jeu de features (defaut : ``FeatureSpec()``).
    kind : "rf" | "extratrees" | "lightgbm" | "xgboost"
    params : dict, optionnel
        Hyperparametres passes a l'estimateur (ex. ``{"n_estimators": 200}``).
    class_names, class_colors : dict valeur_label -> nom / couleur RGB(A)
    max_samples_per_class : int
        Plafond d'echantillons par classe (accelere fortement l'entrainement
        quand on peint de grandes zones, sans perte de qualite notable).
    """

    def __init__(self, features=None, kind: str = "rf", params: dict | None = None,
                 class_names: dict | None = None, class_colors: dict | None = None,
                 max_samples_per_class: int | None = 200_000, random_state: int = 0,
                 n_threads: int | None = None):
        self.extractor = make_extractor(features, n_threads=n_threads)
        self.kind = kind
        self.params = dict(params or {})
        self.class_names = {int(k): v for k, v in (class_names or {}).items()}
        self.class_colors = {int(k): tuple(v) for k, v in (class_colors or {}).items()}
        self.max_samples_per_class = max_samples_per_class
        self.random_state = random_state
        self.estimator = None
        self.classes_: np.ndarray | None = None
        self.ndim_: int | None = None
        self.n_channels_: int | None = None
        self.n_features_: int | None = None
        self.source: dict = {"type": "xwi"}
        self.training_info: dict = {}
        self._memo = None  # (empreinte image, features image entiere) issues du dernier fit

    # ------------------------------------------------------------------
    @property
    def feature_spec(self) -> FeatureSpec | None:
        return self.extractor.spec if isinstance(self.extractor, FeatureEngine) else None

    @property
    def is_fitted(self) -> bool:
        return self.estimator is not None and self.classes_ is not None

    def feature_names(self, ndim: int | None = None, n_channels: int | None = None) -> list:
        return self.extractor.channel_names(ndim or self.ndim_ or 2, n_channels or self.n_channels_ or 1)

    def class_label(self, value: int) -> str:
        return self.class_names.get(int(value), f"classe {value}")

    # ------------------------------------------------------------------
    def fit(self, images, labels, channel_axis: int | None = None,
            block_shape: tuple | None = None, sampler: TrainingSampler | None = None,
            image_keys: Sequence | None = None, progress: Progress = None) -> "PixelClassifier":
        """Entraine sur une ou plusieurs images.

        ``labels`` : meme forme spatiale que l'image ; 0 = non annote, >=1 = classe.
        """
        if isinstance(images, np.ndarray) or not isinstance(images, (list, tuple)):
            images, labels = [images], [labels]
            image_keys = [image_keys] if image_keys is not None and not isinstance(image_keys, (list, tuple)) else image_keys
        if len(images) != len(labels):
            raise ValueError("Autant d'images que de cartes de labels sont attendues.")
        sampler = sampler or TrainingSampler(self.extractor, block_shape=block_shape, cache_bytes=0)

        Xs, ys = [], []
        self._memo = None
        for i, (img, lab) in enumerate(zip(images, labels)):
            img = np.asarray(img)
            key = image_keys[i] if image_keys is not None else None
            X, y = sampler.samples(img, lab, image_key=key, channel_axis=channel_axis, progress=progress)
            if len(images) == 1 and sampler.last_full is not None:
                self._memo = sampler.last_full
            Xs.append(X)
            ys.append(y)
            spatial = img.shape if channel_axis is None else np.delete(img.shape, channel_axis)
            self.ndim_ = len(spatial)
            self.n_channels_ = 1 if channel_axis is None else img.shape[channel_axis]
        X = np.concatenate(Xs)
        y = np.concatenate(ys).astype(np.int64)
        n_total = len(y)
        X, y = subsample_per_class(X, y, self.max_samples_per_class, self.random_state)

        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("Il faut annoter au moins deux classes differentes.")
        y_enc = np.searchsorted(self.classes_, y)
        self.estimator = est_mod.make_estimator(self.kind, random_state=self.random_state, **self.params)
        est_mod.fit_estimator(self.estimator, self.kind, X, y_enc)
        self.n_features_ = X.shape[1]
        counts = {int(c): int((y == c).sum()) for c in self.classes_}
        self.training_info = {"n_annotated": int(n_total), "n_used": int(len(y)),
                              "per_class": counts, "n_images": len(images)}
        for c in self.classes_:
            self.class_names.setdefault(int(c), f"classe {int(c)}")
        return self

    # ------------------------------------------------------------------
    def forget_features(self) -> None:
        """Libere les features d'entrainement gardees pour la prediction."""
        self._memo = None

    def _check_input(self, ndim: int, n_ch: int) -> None:
        if not self.is_fitted:
            raise RuntimeError("Modele non entraine.")
        n = self.extractor.n_features(ndim, n_ch)
        expected = self.n_features_ or getattr(self.estimator, "n_features_in_", n)
        if n != expected:
            raise ValueError(
                f"Le modele attend {expected} features mais l'image ({ndim}D, {n_ch} canal/canaux) "
                f"en produit {n}. Verifiez la dimension (2D/3D) et le nombre de canaux.")

    def predict_proba(self, image: np.ndarray, channel_axis: int | None = None,
                      region=None, block_shape: tuple | None = None, smooth: float = 0.0,
                      norm=None, progress: Progress = None) -> np.ndarray:
        """Probabilites par classe, forme ``(*region, n_classes)`` (ordre de ``classes_``).

        Le calcul se fait par blocs avec halo : la memoire reste bornee et le
        resultat est identique a un calcul global. ``region`` restreint le calcul
        (ex. la coupe affichee dans napari). ``smooth`` lisse les probabilites
        (sigma en pixels) avant decision, ce qui reduit le bruit "poivre et sel".
        """
        img = np.asarray(image)
        if channel_axis is not None:
            img = np.moveaxis(img, channel_axis, -1)
            spatial, ca = img.shape[:-1], -1
            n_ch = img.shape[-1]
        else:
            spatial, ca, n_ch = img.shape, None, 1
        ndim = len(spatial)
        self._check_input(ndim, n_ch)
        region = normalize_region(region, spatial)
        rshape = tuple(r.stop - r.start for r in region)
        if norm is None:
            norm = self.extractor.fit_normalization(img, channel_axis=ca)
        halo = self.extractor.halo(ndim)
        sh = int(np.ceil(3 * smooth)) + 1 if smooth > 0 else 0
        halo = tuple(h + sh for h in halo)
        n_feat = self.extractor.n_features(ndim, n_ch)
        memo = None
        if self._memo is not None and self._memo[0] == image_fingerprint(image):
            memo = self._memo[1]  # features deja calculees lors de l'entrainement
        full_only = getattr(self.extractor, "requires_full_image", False)
        if full_only and memo is None:
            # multi-resolution (IPSDK) : features calculees sur l'image entiere puis decoupees
            memo = self.extractor.compute(img, norm=norm, channel_axis=ca)
            halo = (0,) * ndim if not sh else halo
        bs = block_shape or (rshape if memo is not None else default_block_shape(rshape, n_feat, halo))
        bs = tuple(min(b, s) for b, s in zip(bs, rshape))
        K = len(self.classes_)
        out = np.empty(rshape + (K,), np.float32)
        blocks = list(iter_blocks(rshape, bs))
        for k, rel in enumerate(blocks):
            inner = tuple(slice(r.start + b.start, r.start + b.stop) for r, b in zip(region, rel))
            outer, local = with_halo(inner, halo, spatial)
            if memo is not None:
                feats = memo[outer]
            else:
                feats = self.extractor.compute(img[outer], norm=norm, channel_axis=ca)
            if sh:
                oshape = feats.shape[:-1]
                mid, mlocal = with_halo(local, (sh,) * ndim, oshape)
                sub = feats[mid]
                proba = est_mod.predict_proba(self.estimator, sub.reshape(-1, sub.shape[-1]))
                proba = proba.reshape(sub.shape[:-1] + (K,))
                from scipy import ndimage as ndi
                for c in range(K):
                    proba[..., c] = ndi.gaussian_filter(proba[..., c], smooth, mode="nearest")
                proba /= np.maximum(proba.sum(-1, keepdims=True), 1e-12)
                proba = proba[mlocal]
            else:
                sub = feats[local]
                proba = est_mod.predict_proba(self.estimator, sub.reshape(-1, sub.shape[-1]))
                proba = proba.reshape(sub.shape[:-1] + (K,))
            out[rel] = proba
            if progress is not None:
                progress(k + 1, len(blocks))
        return out

    def predict(self, image: np.ndarray, channel_axis: int | None = None, region=None,
                block_shape: tuple | None = None, smooth: float = 0.0,
                return_proba: bool = False, return_uncertainty: bool = False,
                progress: Progress = None):
        """Segmentation (valeurs de labels d'origine) ; optionnellement probas/incertitude."""
        proba = self.predict_proba(image, channel_axis=channel_axis, region=region,
                                   block_shape=block_shape, smooth=smooth, progress=progress)
        seg = labels_from_proba(proba, self.classes_)
        if not (return_proba or return_uncertainty):
            return seg
        res = [seg]
        if return_proba:
            res.append(proba)
        if return_uncertainty:
            res.append(uncertainty(proba))
        return tuple(res)

    # ------------------------------------------------------------------
    def feature_importances(self, group: str | None = None) -> dict | None:
        """Contribution relative des features (somme = 1), triee par ordre decroissant.

        ``group`` : None (par canal), ``"family"`` ou ``"scale"``.
        """
        if not self.is_fitted:
            return None
        n = self.n_features_ or getattr(self.estimator, "n_features_in_", 0)
        imp = getattr(self.estimator, "feature_importances_", None)
        imp = None if imp is None else np.asarray(imp, float)
        if imp is None or not np.isfinite(imp).all() or imp.sum() <= 0:
            imp = est_mod.split_count_importances(self.estimator, n)
        if imp is None:
            return None
        imp = imp / imp.sum()
        names = self.feature_names()
        if len(names) != len(imp):
            names = [f"f{i}" for i in range(len(imp))]
        if group:
            agg: dict = {}
            for nm, v in zip(names, imp):
                agg[_group_key(nm, group)] = agg.get(_group_key(nm, group), 0.0) + float(v)
            items = agg.items()
        else:
            items = zip(names, imp.tolist())
        return dict(sorted(items, key=lambda kv: -kv[1]))

    def describe(self) -> str:
        lines = [f"Modele xwiClassifier ({self.source.get('type', 'xwi')})",
                 f"  classifieur : {self.kind} ({type(self.estimator).__name__ if self.estimator is not None else 'non entraine'})",
                 f"  features    : {self.extractor.kind}, {self.n_features_} canaux, "
                 f"donnees {self.ndim_}D, {self.n_channels_} canal/canaux"]
        if self.classes_ is not None:
            lines.append("  classes     : " + ", ".join(f"{int(c)}={self.class_label(c)}" for c in self.classes_))
        if self.feature_spec is not None:
            sp = self.feature_spec
            lines.append(f"  familles    : {', '.join(sp.families)}")
            lines.append(f"  normalisation={sp.normalize} spacing={sp.spacing} 2D_par_coupe={sp.compute_in_2d}")
        if self.training_info:
            lines.append(f"  entrainement: {self.training_info}")
        src = self.source or {}
        if src.get("annotation_agreement") is not None:
            lines.append(f"  verification : {100 * src['annotation_agreement']:.2f}% d'accord avec les "
                         f"annotations du modele d'origine")
        if src.get("ipsdk_values"):
            lines.append("  valeurs IPSDK: " + ", ".join(f"{k}->{v}" for k, v in src["ipsdk_values"].items()))
        if getattr(self.extractor, "kind", "") == "ipsdk":
            ex = self.extractor
            lines.append(f"  grille IPSDK : {len(ex.grid)} filtres, dimension {ex.dimension}, "
                         f"gaussienne normalisee={ex.gaussian_normalize}, zoom={ex.zoom_mode}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        from ..io.models import save_model
        save_model(self, path)

    @staticmethod
    def load(path: str, **kwargs) -> "PixelClassifier":
        from ..io.models import load_model
        return load_model(path, **kwargs)


# --------------------------------------------------------------------------
def labels_from_proba(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    classes = np.asarray(classes)
    dtype = np.uint8 if classes.max() < 256 else np.uint16
    return classes[np.argmax(proba, axis=-1)].astype(dtype)


def uncertainty(proba: np.ndarray) -> np.ndarray:
    """1 - (p_max - p_second) : proche de 1 la ou le classifieur hesite."""
    if proba.shape[-1] < 2:
        return np.zeros(proba.shape[:-1], np.float32)
    part = np.partition(proba, -2, axis=-1)
    return (1.0 - (part[..., -1] - part[..., -2])).astype(np.float32)


_SCALE_RE = re.compile(r"_s([0-9.]+)|\(σ=([0-9.]+)\)")


_IPSDK_SIZE_RE = re.compile(r"\s(\d+)x\d+")


def _group_key(name: str, group: str) -> str:
    m = _IPSDK_SIZE_RE.search(name)
    if m:  # nom IPSDK Explorer : "Gaussian 5x5 (x 1/2)"
        if group == "family":
            return name[:m.start()].strip()
        if group == "scale":
            return f"taille {m.group(1)}x{m.group(1)}"
    if group == "family":
        if "(" in name:  # nom ilastik
            return name.split(" (")[0]
        base = re.sub(r"^c\d+_", "", name)
        return re.split(r"_s[0-9.]", base)[0]
    if group == "scale":
        m = _SCALE_RE.search(name)
        return f"σ/r={m.group(1) or m.group(2)}" if m else "brut"
    raise ValueError("group doit etre 'family' ou 'scale'")
