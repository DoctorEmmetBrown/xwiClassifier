"""Decoupage en blocs avec halo, pour traiter des volumes plus grands que la RAM."""
from __future__ import annotations

import itertools
import math

import numpy as np


def default_block_shape(spatial_shape: tuple, n_features: int,
                        budget_bytes: float = 1.5e9, n_threads: int = 4) -> tuple:
    """Taille de bloc (cubique) telle que features + intermediaires tiennent dans le budget."""
    ndim = len(spatial_shape)
    # features float32 + ~10 intermediaires par thread
    per_voxel = 4.0 * (n_features + 10 * n_threads + 4)
    voxels = max(budget_bytes / per_voxel, 32 ** ndim)
    side = int(voxels ** (1.0 / ndim))
    side = max(32, min(side, 4096))
    return tuple(min(side, s) for s in spatial_shape)


def iter_blocks(shape: tuple, block_shape: tuple):
    """Genere les tuples de slices (region interne) couvrant ``shape``."""
    ranges = [range(0, s, b) for s, b in zip(shape, block_shape)]
    for starts in itertools.product(*ranges):
        yield tuple(slice(st, min(st + b, s)) for st, b, s in zip(starts, block_shape, shape))


def n_blocks(shape: tuple, block_shape: tuple) -> int:
    return int(np.prod([math.ceil(s / b) for s, b in zip(shape, block_shape)]))


def with_halo(inner: tuple, halo: tuple, shape: tuple):
    """Region etendue (bornee a l'image) et position de la region interne dans celle-ci."""
    outer = tuple(slice(max(0, sl.start - h), min(s, sl.stop + h))
                  for sl, h, s in zip(inner, halo, shape))
    local = tuple(slice(sl.start - o.start, sl.stop - o.start) for sl, o in zip(inner, outer))
    return outer, local


def normalize_region(region, shape: tuple) -> tuple:
    """Convertit une region (slices / entiers / None) en tuple de slices bornees."""
    if region is None:
        return tuple(slice(0, s) for s in shape)
    region = tuple(region) + (slice(None),) * (len(shape) - len(tuple(region)))
    out = []
    for r, s in zip(region, shape):
        if isinstance(r, (int, np.integer)):
            r = slice(int(r), int(r) + 1)
        start, stop, _ = r.indices(s)
        out.append(slice(start, stop))
    return tuple(out)
