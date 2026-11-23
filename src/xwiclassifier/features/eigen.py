"""Valeurs propres analytiques de matrices symetriques 2x2 et 3x3, vectorisees.

Bien plus rapide et econome que ``numpy.linalg.eigvalsh`` sur des piles de
matrices (pas de copie (N, 3, 3)). Valeurs rendues par ordre **decroissant**,
comme vigra/ilastik et scikit-image.
"""
from __future__ import annotations

import numpy as np


def eig2_sym(a, b, c):
    """Matrices [[a, b], [b, c]] -> (l1 >= l2), float32."""
    a = np.asarray(a, np.float64)
    c = np.asarray(c, np.float64)
    half_tr = 0.5 * (a + c)
    disc = np.sqrt((0.5 * (a - c)) ** 2 + np.asarray(b, np.float64) ** 2)
    return [(half_tr + disc).astype(np.float32), (half_tr - disc).astype(np.float32)]


def eig3_sym(a11, a22, a33, a12, a13, a23):
    """Matrices symetriques 3x3 -> (l1 >= l2 >= l3), float32 (methode trigonometrique)."""
    a11 = np.asarray(a11, np.float64)
    a22 = np.asarray(a22, np.float64)
    a33 = np.asarray(a33, np.float64)
    a12 = np.asarray(a12, np.float64)
    a13 = np.asarray(a13, np.float64)
    a23 = np.asarray(a23, np.float64)

    q = (a11 + a22 + a33) / 3.0
    b11, b22, b33 = a11 - q, a22 - q, a33 - q
    p1 = a12 * a12 + a13 * a13 + a23 * a23
    p2 = b11 * b11 + b22 * b22 + b33 * b33 + 2.0 * p1
    p = np.sqrt(p2 / 6.0)
    safe = p > 1e-30
    inv_p = np.where(safe, 1.0 / np.where(safe, p, 1.0), 0.0)
    b11 *= inv_p
    b22 *= inv_p
    b33 *= inv_p
    c12, c13, c23 = a12 * inv_p, a13 * inv_p, a23 * inv_p
    det = (b11 * (b22 * b33 - c23 * c23)
           - c12 * (c12 * b33 - c23 * c13)
           + c13 * (c12 * c23 - b22 * c13))
    r = np.clip(0.5 * det, -1.0, 1.0)
    phi = np.arccos(r) / 3.0
    l1 = q + 2.0 * p * np.cos(phi)
    l3 = q + 2.0 * p * np.cos(phi + 2.0 * np.pi / 3.0)
    l2 = 3.0 * q - l1 - l3
    return [l1.astype(np.float32), l2.astype(np.float32), l3.astype(np.float32)]


def eig_sym(components: dict, axes: tuple):
    """Valeurs propres d'un tenseur symetrique donne par ses composantes.

    ``components[(i, j)]`` (i <= j) pour les axes ``axes`` (2 ou 3 axes).
    """
    if len(axes) == 2:
        i, j = axes
        return eig2_sym(components[(i, i)], components[(i, j)], components[(j, j)])
    if len(axes) == 3:
        i, j, k = axes
        return eig3_sym(components[(i, i)], components[(j, j)], components[(k, k)],
                        components[(i, j)], components[(i, k)], components[(j, k)])
    if len(axes) == 1:
        (i,) = axes
        return [np.asarray(components[(i, i)], np.float32)]
    raise ValueError("eig_sym : 1, 2 ou 3 axes attendus")
