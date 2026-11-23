"""Extraction des echantillons d'entrainement (voxels annotes), bloc par bloc.

Les features ne sont calculees que la ou il y a des annotations (plus un halo).
La taille des blocs est choisie automatiquement par un modele de cout : petits
blocs pour quelques traits localises, image entiere (sans surcout de halo) pour
des annotations eparpillees partout. Un cache par bloc evite de recalculer les
features quand on re-entraine apres quelques coups de pinceau (mode napari), et
les features de l'image entiere, si elles ont ete calculees, sont gardees pour
la prediction qui suit (``last_full``).
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

from ..blocks import bytes_per_voxel, default_block_shape, iter_blocks, memory_budget, with_halo


def candidate_blocks(ndim: int) -> list:
    sides = (32, 64, 128, 256) if ndim == 3 else (128, 256, 512, 1024, 2048)
    return [(s,) * ndim for s in sides]


def image_fingerprint(img: np.ndarray):
    """Identifiant d'image (adresse memoire + forme + echantillon des valeurs)."""
    a = np.asarray(img)
    sample = a.reshape(-1)[:: max(1, a.size // 4096)]
    return (a.__array_interface__["data"][0], a.shape, a.dtype.str, float(np.sum(sample, dtype=np.float64)))


class TrainingSampler:
    def __init__(self, extractor, block_shape: tuple | None = None,
                 cache_bytes: float = 2e9, budget_bytes: float | None = None, keep_full_bytes: float | None = None):
        self.extractor = extractor
        self.block_shape = block_shape
        self.cache_bytes = cache_bytes
        self.budget_bytes = budget_bytes or memory_budget()
        self.keep_full_bytes = keep_full_bytes if keep_full_bytes is not None else self.budget_bytes
        self._cache: OrderedDict = OrderedDict()
        self._cache_size = 0
        self.last_full = None  # (empreinte image, features de l'image entiere)

    def clear(self) -> None:
        self._cache.clear()
        self._cache_size = 0
        self.last_full = None

    def _store(self, key, feats):
        if self.cache_bytes <= 0 or feats.nbytes > self.cache_bytes:
            return
        self._cache[key] = feats
        self._cache_size += feats.nbytes
        while self._cache_size > self.cache_bytes and len(self._cache) > 1:
            _, old = self._cache.popitem(last=False)
            self._cache_size -= old.nbytes

    def _plan(self, spatial, coords, halo, n_feat, image_key):
        """Choisit la grille de blocs minimisant le volume a calculer (hors cache)."""
        ndim = len(spatial)
        full_bs = default_block_shape(spatial, n_feat, halo, self.budget_bytes)
        cands = [self.block_shape] if self.block_shape else candidate_blocks(ndim) + [full_bs]
        best = None
        max_vox = self.budget_bytes / bytes_per_voxel(n_feat)
        for bs in cands:
            bs = tuple(min(b, s) for b, s in zip(bs, spatial))
            if np.prod([min(b + 2 * h, s) for b, h, s in zip(bs, halo, spatial)], dtype=float) > max_vox \
                    and bs != full_bs and not self.block_shape:
                continue
            idx = np.unique(coords // np.asarray(bs), axis=0)
            blocks = [tuple(slice(int(i) * b, min((int(i) + 1) * b, s)) for i, b, s in zip(row, bs, spatial))
                      for row in idx]
            cost = 0.0
            for inner in blocks:
                key = None if image_key is None else (image_key, tuple((x.start, x.stop) for x in inner))
                if key is not None and key in self._cache:
                    continue
                outer, _ = with_halo(inner, halo, spatial)
                cost += np.prod([o.stop - o.start for o in outer], dtype=float)
            if best is None or cost < best[0] * 0.98:
                best = (cost, bs, blocks)
        return best[1], best[2]

    def samples(self, image: np.ndarray, labels: np.ndarray, norm=None,
                image_key=None, channel_axis: int | None = None, progress=None):
        """Renvoie (X, y) pour les voxels ``labels > 0``."""
        img = np.asarray(image)
        if channel_axis is not None:
            img = np.moveaxis(img, channel_axis, -1)
            spatial = img.shape[:-1]
            n_ch = img.shape[-1]
        else:
            spatial, n_ch = img.shape, 1
        labels = np.asarray(labels)
        if labels.shape != spatial:
            raise ValueError(f"Labels {labels.shape} incompatibles avec l'image {spatial}")
        ndim = len(spatial)
        coords = np.argwhere(labels > 0)
        if coords.size == 0:
            raise ValueError("Aucun pixel annote.")
        halo = self.extractor.halo(ndim)
        n_feat = self.extractor.n_features(ndim, n_ch)
        ca = -1 if channel_axis is not None else None
        if norm is None:
            norm = self.extractor.fit_normalization(img, channel_axis=ca)
        bs, blocks = self._plan(spatial, coords, halo, n_feat, image_key)
        whole = len(blocks) == 1 and all(x.start == 0 and x.stop == s for x, s in zip(blocks[0], spatial))
        self.last_full = None

        Xs, ys = [], []
        for k, inner in enumerate(blocks):
            lab = labels[inner]
            mask = lab > 0
            key = None if image_key is None else (image_key, tuple((x.start, x.stop) for x in inner))
            feats = self._cache.get(key) if key is not None else None
            if feats is None:
                outer, local = with_halo(inner, halo, spatial)
                feats = self.extractor.compute(img[outer], norm=norm, channel_axis=ca)[local]
                if key is not None:
                    self._store(key, feats)
            else:
                self._cache.move_to_end(key)
            if whole and feats.nbytes <= self.keep_full_bytes:
                self.last_full = (image_fingerprint(image), feats)
            Xs.append(feats[mask])
            ys.append(lab[mask])
            if progress is not None:
                progress(k + 1, len(blocks))
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
