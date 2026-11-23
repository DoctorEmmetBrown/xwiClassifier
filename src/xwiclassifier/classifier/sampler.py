"""Extraction des echantillons d'entrainement (voxels annotes), bloc par bloc.

Les features ne sont calculees que sur les blocs contenant des annotations
(plus un halo), jamais sur tout le volume : l'entrainement reste rapide sur de
gros volumes. Un cache optionnel (par bloc) evite de recalculer les features
quand on re-entraine apres quelques coups de pinceau (mode interactif napari).
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

from ..blocks import iter_blocks, with_halo


def default_training_block(ndim: int) -> tuple:
    return (64, 64, 64) if ndim == 3 else (256, 256)


class TrainingSampler:
    def __init__(self, extractor, block_shape: tuple | None = None,
                 cache_bytes: float = 2e9):
        self.extractor = extractor
        self.block_shape = block_shape
        self.cache_bytes = cache_bytes
        self._cache: OrderedDict = OrderedDict()
        self._cache_size = 0

    def clear(self) -> None:
        self._cache.clear()
        self._cache_size = 0

    def _store(self, key, feats):
        if self.cache_bytes <= 0:
            return
        self._cache[key] = feats
        self._cache_size += feats.nbytes
        while self._cache_size > self.cache_bytes and len(self._cache) > 1:
            _, old = self._cache.popitem(last=False)
            self._cache_size -= old.nbytes

    def samples(self, image: np.ndarray, labels: np.ndarray, norm=None,
                image_key=None, channel_axis: int | None = None, progress=None):
        """Renvoie (X, y) pour les voxels ``labels > 0``."""
        img = np.asarray(image)
        if channel_axis is not None:
            img = np.moveaxis(img, channel_axis, -1)
            spatial = img.shape[:-1]
        else:
            spatial = img.shape
        labels = np.asarray(labels)
        if labels.shape != spatial:
            raise ValueError(f"Labels {labels.shape} incompatibles avec l'image {spatial}")
        ndim = len(spatial)
        bs = self.block_shape or default_training_block(ndim)
        bs = tuple(min(b, s) for b, s in zip(bs, spatial))
        halo = self.extractor.halo(ndim)
        if norm is None:
            norm = self.extractor.fit_normalization(img, channel_axis=-1 if channel_axis is not None else None)
        ca = -1 if channel_axis is not None else None

        todo = [inner for inner in iter_blocks(spatial, bs) if labels[inner].any()]
        Xs, ys = [], []
        for k, inner in enumerate(todo):
            lab = labels[inner]
            mask = lab > 0
            key = None if image_key is None else (image_key, tuple((s.start, s.stop) for s in inner))
            feats = self._cache.get(key) if key is not None else None
            if feats is None:
                outer, local = with_halo(inner, halo, spatial)
                feats = self.extractor.compute(img[outer], norm=norm, channel_axis=ca)[local]
                if key is not None:
                    self._store(key, feats)
            else:
                self._cache.move_to_end(key)
            Xs.append(feats[mask])
            ys.append(lab[mask])
            if progress is not None:
                progress(k + 1, len(todo))
        if not Xs:
            raise ValueError("Aucun pixel annote.")
        return np.concatenate(Xs), np.concatenate(ys)


def subsample_per_class(X, y, max_per_class: int | None, random_state: int = 0):
    """Plafonne le nombre d'echantillons par classe (tirage aleatoire reproductible)."""
    if not max_per_class:
        return X, y
    rng = np.random.default_rng(random_state)
    keep = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        if idx.size > max_per_class:
            idx = rng.choice(idx, max_per_class, replace=False)
        keep.append(idx)
    keep = np.sort(np.concatenate(keep))
    return X[keep], y[keep]
