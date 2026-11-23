"""Decoupage en blocs avec halo, pour traiter des volumes plus grands que la RAM."""
from __future__ import annotations

import itertools
import math
import os

import numpy as np


def bytes_per_voxel(n_features: int, n_threads: int = 4) -> float:
    """Memoire par voxel du *domaine calcule* : features float32 + intermediaires par thread."""
    return 4.0 * (n_features + 20 * n_threads + 8)


def memory_budget() -> float:
    """Budget memoire par bloc : ``XWI_MEMORY_BUDGET_GB`` ou ~30 % de la RAM (1 a 16 Go)."""
    env = os.environ.get("XWI_MEMORY_BUDGET_GB")
    if env:
        return float(env) * 1e9
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        total = 8e9
    return float(min(max(0.3 * total, 1e9), 16e9))


_SIDES = (16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096)


def _axis_cost(size: int, block: int, halo: int) -> int:
    """Somme des etendues calculees (bloc + halo, bornees) le long d'un axe."""
    total = 0
    for start in range(0, size, block):
        stop = min(start + block, size)
        total += min(size, stop + halo) - max(0, start - halo)
    return total


def default_block_shape(spatial_shape: tuple, n_features: int, halo: tuple | None = None,
                        budget_bytes: float | None = None, n_threads: int = 4) -> tuple:
    """Forme de bloc minimisant le volume calcule (blocs + halos) sous contrainte memoire.

    Chaque axe est soit couvert entierement, soit decoupe ; on evalue le cout exact
    (produit des sommes par axe) de toutes les combinaisons. Si l'image entiere
    tient dans le budget, un seul bloc est renvoye.
    """
    shape = tuple(int(s) for s in spatial_shape)
    halo = tuple(int(h) for h in halo) if halo is not None else (0,) * len(shape)
    budget = budget_bytes if budget_bytes is not None else memory_budget()
    max_vox = budget / bytes_per_voxel(n_features, n_threads)
    if np.prod(shape, dtype=float) <= max_vox:
        return shape
    options = [sorted({s} | {b for b in _SIDES if b < s}) for s in shape]
    best = None
    for combo in itertools.product(*options):
        outer = np.prod([min(b + 2 * h, s) for b, h, s in zip(combo, halo, shape)], dtype=float)
        if outer > max_vox:
            continue
        cost = np.prod([_axis_cost(s, b, h) for s, b, h in zip(shape, combo, halo)], dtype=float)
        # a cout egal, on prefere moins de blocs
        key = (cost, np.prod([math.ceil(s / b) for s, b in zip(shape, combo)]))
        if best is None or key < best[0]:
            best = (key, combo)
    if best is None:  # budget tres faible : plus petits blocs possibles
        return tuple(min(16, s) for s in shape)
    return tuple(best[1])


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
