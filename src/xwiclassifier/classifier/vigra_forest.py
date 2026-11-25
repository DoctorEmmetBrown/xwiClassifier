"""Lecture des forets aleatoires vigra (ilastik) et conversion en scikit-learn.

Format vigra (``random_forest_hdf5_impex.hxx`` / ``rf_nodeproxy.hxx``) :
  * groupe de foret : ``_ext_param/labels``, ``_options/...``, ``Tree_XX/topology``
    (int32) et ``Tree_XX/parameters`` (float64) ;
  * ``topology[0]`` = nb de features, ``topology[1]`` = nb de classes, racine a l'indice 2 ;
  * noeud seuil (type 0) : ``[type, adr_param, enfant_gauche, enfant_droit, colonne]``,
    ``parameters[adr] = poids``, ``parameters[adr+1] = seuil`` ; regle : x < seuil -> gauche ;
  * feuille (type 0x40000000) : ``[type, adr_param]`` ; ``parameters[adr]`` = poids
    (nb d'echantillons), puis une probabilite par classe ;
  * prediction : moyenne des probabilites des feuilles sur tous les arbres
    (``predict_weighted_ = 0``) ; ilastik combine plusieurs forets en ponderant
    par leur nombre d'arbres, ce qui revient a moyenner sur tous les arbres.

La conversion produit une ``RandomForestClassifier`` scikit-learn dont les
predictions sont identiques (seuils convertis de ``<`` vers ``<=`` en float32),
executee en C et en parallele, sans dependre de vigra.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .forest_build import forest_from_trees, tree_from_arrays

LEAF_TAG = 0x40000000
THRESHOLD_NODE = 0
CONST_PROB_LEAF = LEAF_TAG


@dataclass
class VigraTree:
    feature: np.ndarray     # -2 pour les feuilles
    threshold: np.ndarray   # seuil vigra (regle x < seuil -> gauche)
    left: np.ndarray        # -1 pour les feuilles
    right: np.ndarray
    proba: np.ndarray       # (n_noeuds, n_classes) ; significatif aux feuilles
    weight: np.ndarray      # poids des feuilles (nb d'echantillons)

    @property
    def n_nodes(self) -> int:
        return len(self.feature)


def parse_vigra_tree(topology: np.ndarray, parameters: np.ndarray) -> VigraTree:
    topo = np.asarray(topology, dtype=np.int64)
    par = np.asarray(parameters, dtype=np.float64)
    n_classes = int(topo[1])
    order, ids, stack = [], {}, [2]
    while stack:
        i = stack.pop()
        ids[i] = len(order)
        order.append(i)
        t = int(topo[i])
        if t & LEAF_TAG:
            if t != CONST_PROB_LEAF:
                raise NotImplementedError(f"Type de feuille vigra non supporte : {t:#x}")
            continue
        if t != THRESHOLD_NODE:
            raise NotImplementedError(f"Type de noeud vigra non supporte : {t}")
        stack.append(int(topo[i + 3]))
        stack.append(int(topo[i + 2]))
    n = len(order)
    feature = np.full(n, -2, np.int64)
    threshold = np.full(n, -2.0)
    left = np.full(n, -1, np.int64)
    right = np.full(n, -1, np.int64)
    proba = np.zeros((n, n_classes))
    weight = np.zeros(n)
    for k, i in enumerate(order):
        t, addr = int(topo[i]), int(topo[i + 1])
        if t & LEAF_TAG:
            weight[k] = par[addr]
            proba[k] = par[addr + 1: addr + 1 + n_classes]
        else:
            left[k], right[k] = ids[int(topo[i + 2])], ids[int(topo[i + 3])]
            feature[k] = int(topo[i + 4])
            threshold[k] = par[addr + 1]
    return VigraTree(feature, threshold, left, right, proba, weight)


def read_vigra_forest(group) -> dict:
    """Lit un groupe HDF5 de foret vigra (ex. ``ClassifierForests/Forest0000``)."""
    names = sorted(k for k in group.keys() if not k.startswith("_"))
    trees = [parse_vigra_tree(group[k]["topology"][()], group[k]["parameters"][()]) for k in names]
    ext = group["_ext_param"]
    labels = np.asarray(ext["labels"][()]).ravel() if "labels" in ext else None
    n_features = int(np.asarray(ext["column_count_"][()]).ravel()[0])
    n_classes = int(np.asarray(ext["class_count_"][()]).ravel()[0])
    weighted = False
    if "_options" in group and "predict_weighted_" in group["_options"]:
        weighted = bool(np.asarray(group["_options"]["predict_weighted_"][()]).ravel()[0])
    return {"trees": trees, "labels": labels, "n_features": n_features,
            "n_classes": n_classes, "predict_weighted": weighted}


# --------------------------------------------------------------------------
def predict_proba_reference(trees, X: np.ndarray) -> np.ndarray:
    """Prediction de reference (numpy pur, semantique vigra), pour les tests."""
    X = np.asarray(X, np.float32)
    acc = np.zeros((X.shape[0], trees[0].proba.shape[1]))
    rows = np.arange(X.shape[0])
    for t in trees:
        idx = np.zeros(X.shape[0], np.int64)
        active = t.left[idx] >= 0
        while active.any():
            a = idx[active]
            go_left = X[rows[active], t.feature[a]] < t.threshold[a]
            idx[active] = np.where(go_left, t.left[a], t.right[a])
            active = t.left[idx] >= 0
        acc += t.proba[idx]
    return (acc / acc.sum(1, keepdims=True)).astype(np.float32)


def _strict_to_le(threshold: np.ndarray) -> np.ndarray:
    """Seuil t' (float32 exact) tel que, pour tout x float32 : x <= t'  <=>  x < t."""
    t = np.asarray(threshold, np.float64)
    t32 = t.astype(np.float32)
    too_big = t32.astype(np.float64) >= t
    t32[too_big] = np.nextafter(t32[too_big], np.float32(-np.inf))
    return t32.astype(np.float64)


def _to_sklearn_tree(vt: VigraTree, n_features: int, n_classes: int):
    n = vt.n_nodes
    # effectifs par noeud : feuilles = poids x probas, noeuds internes = somme des enfants
    counts = vt.proba * vt.weight[:, None]
    for k in range(n - 1, -1, -1):  # ordre prefixe inverse : enfants avant parents
        if vt.left[k] >= 0:
            counts[k] = counts[vt.left[k]] + counts[vt.right[k]]
    leaf = vt.left < 0
    values = counts.copy()
    values[leaf] = vt.proba[leaf]
    thr = np.full(n, -2.0)
    thr[~leaf] = _strict_to_le(vt.threshold[~leaf])
    return tree_from_arrays(vt.left, vt.right, vt.feature, thr, counts, values, n_features, n_classes)


def to_sklearn_forest(trees, n_features: int, n_classes: int, n_jobs: int = -1):
    """Assemble tous les arbres (de toutes les forets) en une RandomForestClassifier."""
    return forest_from_trees([_to_sklearn_tree(t, n_features, n_classes) for t in trees],
                             n_features, n_classes, n_jobs)
