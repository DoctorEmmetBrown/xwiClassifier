"""Sauvegarde / chargement des modeles (format xwi v2, anciens .joblib v0.1, ilastik, IPSDK).

Le format xwi est un fichier joblib contenant un dictionnaire versionne :
les features sont stockees en dict (et non en objets Python), ce qui rend les
modeles robustes aux refactorisations du code. Attention : joblib/pickle peut
executer du code au chargement, ne chargez que des modeles de confiance.
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
import sys
import types
import warnings
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

FORMAT = "xwiclassifier-model"
VERSION = 2


def _versions() -> dict:
    import sklearn

    from .. import __version__
    return {"xwiclassifier": __version__, "sklearn": sklearn.__version__, "numpy": np.__version__}


def save_model(model, path: str) -> None:
    import joblib

    if not model.is_fitted:
        raise RuntimeError("Impossible de sauver un modele non entraine.")
    payload = {
        "format": FORMAT, "version": VERSION,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "versions": _versions(),
        "features": model.extractor.to_dict(),
        "kind": model.kind, "params": model.params,
        "estimator": model.estimator,
        "classes": np.asarray(model.classes_).tolist(),
        "class_names": {int(k): v for k, v in model.class_names.items()},
        "class_colors": {int(k): list(v) for k, v in model.class_colors.items()},
        "ndim": model.ndim_, "n_channels": model.n_channels_, "n_features": model.n_features_,
        "source": model.source, "training_info": model.training_info,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    joblib.dump(payload, path, compress=3)


def load_model(path: str, **kwargs):
    """Charge un modele : xwi (.joblib), ancien v0.1, projet ilastik (.ilp).

    Les modeles IPSDK (.bin) ne sont pas lisibles directement (format proprietaire
    non documente) : voir :func:`xwiclassifier.io.ipsdk.distill_ipsdk_model`.
    """
    path = os.fspath(path)
    low = path.lower()
    from .ipsdk_explorer import is_ipsdk_model_folder
    if _is_nnunet(path):
        from ..nnunet.inference import NnUNetSegmenter
        return NnUNetSegmenter(path, **kwargs)
    if os.path.isdir(path) or low.endswith(".mho"):
        if is_ipsdk_model_folder(path):
            from .ipsdk_explorer import load_ipsdk_model
            return load_ipsdk_model(path, **kwargs)
        raise ValueError(f"{path} : dossier non reconnu (ni modele IPSDK Explorer, ni modele xwi).")
    if low.endswith(".ilp"):
        from .ilastik_project import load_ilastik_model
        return load_ilastik_model(path, **kwargs)
    if low.endswith(".bin"):
        folder = os.path.dirname(os.path.abspath(path))
        if is_ipsdk_model_folder(folder):  # ModelIPSDK.bin dans un dossier de modele Explorer
            from .ipsdk_explorer import load_ipsdk_model
            return load_ipsdk_model(folder, **kwargs)
        raise ValueError(
            "Fichier IPSDK .bin isole : format binaire proprietaire. Donnez plutot le dossier du "
            "modele IPSDK Explorer (Settings.mho + Model), ou utilisez `xwi ipsdk-distill`.")
    import joblib

    with _legacy_modules():
        payload = joblib.load(path)
    if isinstance(payload, dict) and payload.get("format") == FORMAT:
        return _from_payload(payload)
    if isinstance(payload, dict) and "model" in payload and "spec" in payload:
        return _from_legacy(payload)
    raise ValueError(f"{path} : format de modele non reconnu.")


def _is_nnunet(path: str) -> bool:
    """Dossier de modele nnU-Net, dossier de resultats d'un jeu, ou pointeur .json xwi-nnunet."""
    import json
    from pathlib import Path

    p = Path(path)
    if p.is_file() and p.suffix == ".json":
        try:
            return json.loads(p.read_text()).get("format") == "xwi-nnunet"
        except Exception:
            return False
    if p.is_dir():
        if (p / "plans.json").exists() and (p / "dataset.json").exists():
            return True
        return any((m / "plans.json").exists() for m in p.glob("*__*__*"))
    return False


def _from_payload(d: dict):
    from ..classifier.model import PixelClassifier

    v = d.get("versions", {})
    try:
        import sklearn
        if v.get("sklearn") and v["sklearn"] != sklearn.__version__:
            warnings.warn(f"Modele cree avec scikit-learn {v['sklearn']} (installe : "
                          f"{sklearn.__version__}) : verifiez les resultats.", stacklevel=3)
    except Exception:
        pass
    m = PixelClassifier(features=d["features"], kind=d.get("kind", "rf"), params=d.get("params"),
                        class_names=d.get("class_names"), class_colors=d.get("class_colors"))
    m.estimator = d["estimator"]
    m.classes_ = np.asarray(d["classes"])
    m.ndim_, m.n_channels_, m.n_features_ = d.get("ndim"), d.get("n_channels"), d.get("n_features")
    m.source = d.get("source", {"type": "xwi"})
    m.training_info = d.get("training_info", {})
    return m


# --------------------------------------------------------------------------
# Compatibilite avec les modeles du prototype v0.1 (paquet napari_segforest)
# --------------------------------------------------------------------------
@dataclass
class LegacyFeatureSpec:
    families: tuple = ()
    sigmas: tuple = ()
    extra: list = field(default_factory=list)
    pairs: tuple | None = None


class _legacy_modules:
    """Expose temporairement ``napari_segforest.*`` pour depickler les vieux modeles."""

    NAMES = ("napari_segforest", "napari_segforest.features", "napari_segforest.features.engine")

    def __enter__(self):
        self._added = []
        if "napari_segforest" in sys.modules:
            return self
        for name in self.NAMES:
            mod = types.ModuleType(name)
            mod.FeatureSpec = LegacyFeatureSpec
            sys.modules[name] = mod
            self._added.append(name)
        return self

    def __exit__(self, *exc):
        for name in self._added:
            sys.modules.pop(name, None)
        return False


def _from_legacy(d: dict):
    from ..classifier.model import PixelClassifier
    from ..features import FeatureSpec

    spec = d["spec"]
    if getattr(spec, "pairs", None):
        pairs = tuple(spec.pairs)
    else:
        pairs = tuple((f, 0.0) if f == "intensity" else (f, float(s))
                      for f in spec.families for s in (spec.sigmas if f != "intensity" else (0.0,)))
    warnings.warn("Modele v0.1 : les features sont recalculees avec le moteur v0.2 "
                  "(derivees gaussiennes exactes). Un re-entrainement est recommande.", stacklevel=3)
    m = PixelClassifier(features=FeatureSpec(pairs=pairs, normalize="minmax"),
                        kind=d.get("kind", "rf"), class_names=d.get("class_names"))
    m.estimator = d["model"]
    classes = d.get("classes_")
    m.classes_ = np.asarray(classes if classes is not None else m.estimator.classes_)
    m.n_features_ = int(getattr(m.estimator, "n_features_in_", 0)) or None
    m.n_channels_ = 1
    for nd in (2, 3):  # deduit la dimension compatible avec le nombre de features
        if m.extractor.n_features(nd) == m.n_features_:
            m.ndim_ = nd
            break
    m.source = {"type": "xwi-v0.1"}
    return m
