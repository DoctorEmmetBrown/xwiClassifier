"""Construction de forets scikit-learn a partir de tableaux de noeuds.

Utilise par l'import des forets vigra (ilastik) et IPSDK Explorer : on fabrique
des arbres ``sklearn.tree._tree.Tree`` pour la version de scikit-learn installee,
en recopiant le format interne d'un arbre de reference (robuste aux changements
de format entre versions).
"""
from __future__ import annotations

import numpy as np

_TEMPLATE = None


def _template_state():
    global _TEMPLATE
    if _TEMPLATE is None:
        from sklearn.tree import DecisionTreeClassifier

        tree = DecisionTreeClassifier(max_depth=1, random_state=0).fit([[0.0], [1.0]], [0, 1]).tree_
        _TEMPLATE = tree.__getstate__()
    return _TEMPLATE


def tree_from_arrays(left, right, feature, threshold, counts, values, n_features: int, n_classes: int):
    """DecisionTreeClassifier a partir des tableaux de noeuds (regle : x <= seuil -> gauche).

    ``counts`` : effectifs (ponderes) par classe et par noeud, pour impurete et importances ;
    ``values`` : valeurs des noeuds (normalisees ici en proportions).
    """
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.tree._tree import Tree

    left = np.asarray(left, np.int64)
    right = np.asarray(right, np.int64)
    n = len(left)
    counts = np.asarray(counts, np.float64).reshape(n, -1)
    values = np.asarray(values, np.float64).reshape(n, -1)
    tot = counts.sum(1)
    frac = counts / np.where(tot > 0, tot, 1.0)[:, None]
    impurity = 1.0 - (frac ** 2).sum(1)
    vsum = values.sum(1, keepdims=True)
    values = values / np.where(vsum > 0, vsum, 1.0)
    depth = np.zeros(n, np.int64)
    for k in range(n):
        if left[k] >= 0:
            depth[left[k]] = depth[right[k]] = depth[k] + 1

    state = dict(_template_state())
    nodes = np.zeros(n, dtype=state["nodes"].dtype)
    nodes["left_child"] = left
    nodes["right_child"] = right
    nodes["feature"] = np.where(left >= 0, np.asarray(feature, np.int64), -2)
    nodes["threshold"] = np.where(left >= 0, np.asarray(threshold, np.float64), -2.0)
    nodes["impurity"] = impurity
    nodes["n_node_samples"] = np.maximum(np.rint(tot), 1).astype(nodes["n_node_samples"].dtype)
    nodes["weighted_n_node_samples"] = np.maximum(tot, 1e-12)
    if "missing_go_to_left" in nodes.dtype.names:
        nodes["missing_go_to_left"] = 0
    state.update(max_depth=int(depth.max()), node_count=n, nodes=nodes,
                 values=np.ascontiguousarray(values.reshape(n, 1, n_classes)))
    tree = Tree(n_features, np.array([n_classes], dtype=np.intp), 1)
    tree.__setstate__(state)
    dt = DecisionTreeClassifier()
    dt.tree_ = tree
    dt.n_outputs_ = 1
    dt.n_classes_ = n_classes
    dt.classes_ = np.arange(n_classes)
    dt.n_features_in_ = n_features
    dt.max_features_ = n_features
    return dt


def forest_from_trees(trees, n_features: int, n_classes: int, n_jobs: int = -1):
    """Assemble des DecisionTreeClassifier en RandomForestClassifier (moyenne des probas)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    rf = RandomForestClassifier(n_estimators=len(trees), n_jobs=n_jobs)
    rf.estimators_ = list(trees)
    rf.estimator_ = DecisionTreeClassifier()
    rf.n_outputs_ = 1
    rf.n_classes_ = n_classes
    rf.classes_ = np.arange(n_classes)
    rf.n_features_in_ = n_features
    return rf
