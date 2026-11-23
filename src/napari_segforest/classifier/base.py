"""Classifieur de pixels pluggable.

Backend par defaut : RandomForest (sklearn), comme ilastik. On peut
basculer sur LightGBM / XGBoost si installes (extra "boosting"), via le
parametre `kind`. L'interface entraine sur les pixels labellises et
predit sur tout le volume (en blocs pour limiter la RAM).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from ..features.engine import FeatureEngine, FeatureSpec


def make_estimator(kind: str = "rf", **params):
    """Fabrique un estimateur scikit-learn-compatible."""
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier

        defaults = dict(n_estimators=200, max_depth=None, n_jobs=-1,
                        max_samples=0.5, class_weight="balanced")
        defaults.update(params)
        return RandomForestClassifier(**defaults)
    if kind == "extratrees":
        from sklearn.ensemble import ExtraTreesClassifier

        defaults = dict(n_estimators=200, n_jobs=-1, class_weight="balanced")
        defaults.update(params)
        return ExtraTreesClassifier(**defaults)
    if kind == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError(
                "LightGBM n'est pas installe. Installez-le avec :\n"
                "    pip install lightgbm\n"
                "ou choisissez 'rf'/'extratrees' (toujours disponibles)."
            ) from exc
        defaults = dict(n_estimators=300, num_leaves=63, n_jobs=-1, verbose=-1)
        defaults.update(params)
        return LGBMClassifier(**defaults)
    if kind == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "XGBoost n'est pas installe. Installez-le avec :\n"
                "    pip install xgboost\n"
                "ou choisissez 'rf'/'extratrees' (toujours disponibles)."
            ) from exc
        # importance_type="gain" : importance par gain moyen (comparable a sklearn),
        # bien plus informative que "weight" (defaut) qui concentre tout sur 1 feature.
        defaults = dict(n_estimators=300, tree_method="hist", n_jobs=-1,
                        max_depth=6, learning_rate=0.3, subsample=0.8,
                        colsample_bytree=0.8, importance_type="gain",
                        eval_metric="mlogloss")
        defaults.update(params)
        return XGBClassifier(**defaults)
    raise ValueError(f"Classifieur inconnu : {kind!r}")


def available_classifiers() -> list[str]:
    """Liste des classifieurs reellement utilisables sur cette machine."""
    kinds = ["rf", "extratrees"]
    try:
        import lightgbm  # noqa: F401
        kinds.append("lightgbm")
    except ImportError:
        pass
    try:
        import xgboost  # noqa: F401
        kinds.append("xgboost")
    except ImportError:
        pass
    return kinds


@dataclass
class PixelClassifier:
    kind: str = "rf"
    feature_spec: FeatureSpec = field(default_factory=FeatureSpec)
    params: dict = field(default_factory=dict)
    class_names: dict[int, str] = field(default_factory=dict)  # {valeur_label: nom}

    def __post_init__(self):
        self.engine = FeatureEngine(self.feature_spec)
        self.model = make_estimator(self.kind, **self.params)
        self._features = None
        self._image_id = None
        self.classes_ = None

    # ------------------------------------------------------------------
    def _ensure_features(self, image: np.ndarray) -> np.ndarray:
        if self._image_id != id(image):
            self._features = self.engine.compute(image)
            self._image_id = id(image)
        return self._features

    def fit(self, image: np.ndarray, labels: np.ndarray) -> "PixelClassifier":
        """labels : meme shape que image, 0 = non labellise, >=1 = classe."""
        feats = self._ensure_features(image)
        mask = labels > 0
        if mask.sum() == 0:
            raise ValueError("Aucun pixel labellise.")
        X = feats[mask]
        y = labels[mask]
        # Encodage interne 1,2,3.. -> 0,1,2.. (requis par XGBoost, neutre ailleurs)
        self.classes_ = np.unique(y)
        y_enc = np.searchsorted(self.classes_, y)
        self.model.fit(X, y_enc)
        return self

    def predict(self, image: np.ndarray) -> np.ndarray:
        feats = self._ensure_features(image)
        flat = np.ascontiguousarray(feats.reshape(-1, feats.shape[-1]))
        with warnings.catch_warnings():
            # L'ordre des canaux est garanti identique a l'entrainement (meme
            # FeatureEngine) : le warning "feature names" de LightGBM est sans objet.
            warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
            pred_enc = self.model.predict(flat)
        pred = self.classes_[pred_enc.astype(int)]  # decode vers labels d'origine
        return pred.reshape(image.shape)

    def predict_proba(self, image: np.ndarray) -> np.ndarray:
        feats = self._ensure_features(image)
        flat = np.ascontiguousarray(feats.reshape(-1, feats.shape[-1]))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
            proba = self.model.predict_proba(flat)
        return proba.reshape((*image.shape, proba.shape[-1]))

    def feature_importances(self) -> dict[str, float] | None:
        imp = getattr(self.model, "feature_importances_", None)
        if imp is None:
            return None
        names = self.engine.channel_names or [f"f{i}" for i in range(len(imp))]
        return dict(sorted(zip(names, imp.tolist()), key=lambda kv: -kv[1]))

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        import joblib

        joblib.dump({"kind": self.kind, "spec": self.feature_spec,
                     "model": self.model, "class_names": self.class_names,
                     "classes_": self.classes_}, path)

    @classmethod
    def load(cls, path: str) -> "PixelClassifier":
        import joblib

        d = joblib.load(path)
        obj = cls(kind=d["kind"], feature_spec=d["spec"],
                  class_names=d.get("class_names", {}))
        obj.model = d["model"]
        # classes_ d'origine (labels 1,2,3..) necessaires pour decoder les predictions
        obj.classes_ = d.get("classes_")
        return obj

    def class_label(self, value: int) -> str:
        """Nom lisible d'une valeur de classe (fallback generique)."""
        return self.class_names.get(value, f"classe {value}")
