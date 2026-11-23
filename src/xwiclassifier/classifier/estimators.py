"""Fabrique d'estimateurs (RandomForest, ExtraTrees, LightGBM, XGBoost)."""
from __future__ import annotations

import warnings

import numpy as np

CLASSIFIERS = ("rf", "extratrees", "lightgbm", "xgboost")


def available_classifiers() -> list[str]:
    """Classifieurs reellement utilisables sur cette machine."""
    kinds = ["rf", "extratrees"]
    for kind, mod in (("lightgbm", "lightgbm"), ("xgboost", "xgboost")):
        try:
            __import__(mod)
            kinds.append(kind)
        except Exception:
            pass
    return kinds


def default_params(kind: str) -> dict:
    if kind == "rf":
        return dict(n_estimators=100, max_features="sqrt", min_samples_leaf=1,
                    max_samples=0.5, class_weight="balanced", n_jobs=-1)
    if kind == "extratrees":
        return dict(n_estimators=100, max_features="sqrt", class_weight="balanced", n_jobs=-1)
    if kind == "lightgbm":
        return dict(n_estimators=200, num_leaves=63, learning_rate=0.1, subsample=0.8,
                    subsample_freq=1, colsample_bytree=0.8, class_weight="balanced",
                    n_jobs=-1, verbose=-1)
    if kind == "xgboost":
        # importance_type="gain" : importances comparables a sklearn
        return dict(n_estimators=300, tree_method="hist", max_depth=6, learning_rate=0.3,
                    subsample=0.8, colsample_bytree=0.8, importance_type="gain", n_jobs=-1)
    raise ValueError(f"Classifieur inconnu : {kind!r} (choix : {CLASSIFIERS})")


def make_estimator(kind: str = "rf", random_state: int | None = 0, **params):
    """Instancie un estimateur compatible scikit-learn avec des defauts adaptes."""
    p = default_params(kind)
    p.update(params)
    if random_state is not None:
        p.setdefault("random_state", random_state)
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**p)
    if kind == "extratrees":
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(**p)
    if kind == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError("LightGBM n'est pas installe : pip install lightgbm") from exc
        return LGBMClassifier(**p)
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("XGBoost n'est pas installe : pip install xgboost "
                          "(sur macOS : brew install libomp)") from exc
    return XGBClassifier(**p)


def fit_estimator(estimator, kind: str, X: np.ndarray, y_enc: np.ndarray) -> None:
    """Entraine en gerant l'equilibrage des classes selon la bibliotheque."""
    if kind == "xgboost":
        from sklearn.utils.class_weight import compute_sample_weight

        n_classes = int(y_enc.max()) + 1
        estimator.set_params(eval_metric="logloss" if n_classes <= 2 else "mlogloss")
        estimator.fit(X, y_enc, sample_weight=compute_sample_weight("balanced", y_enc))
    else:
        estimator.fit(X, y_enc)


def predict_proba(estimator, X: np.ndarray, chunk: int = 1_000_000) -> np.ndarray:
    """``predict_proba`` par paquets (memoire bornee), resultat float32."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    out = None
    with warnings.catch_warnings():
        # LightGBM se plaint des noms de colonnes quand on predit sur numpy
        warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
        for start in range(0, X.shape[0], chunk):
            p = np.asarray(estimator.predict_proba(X[start:start + chunk]), dtype=np.float32)
            if out is None:
                out = np.empty((X.shape[0], p.shape[1]), np.float32)
            out[start:start + chunk] = p
    if out is None:
        n = len(getattr(estimator, "classes_", [0, 1]))
        out = np.zeros((0, n), np.float32)
    return out


def split_count_importances(estimator, n_features: int) -> np.ndarray | None:
    """Importance = frequence d'utilisation des features dans les noeuds (repli)."""
    trees = getattr(estimator, "estimators_", None)
    if not trees:
        return None
    counts = np.zeros(n_features, float)
    for t in trees:
        f = t.tree_.feature
        f = f[f >= 0]
        counts += np.bincount(f, minlength=n_features)[:n_features]
    s = counts.sum()
    return counts / s if s > 0 else None
