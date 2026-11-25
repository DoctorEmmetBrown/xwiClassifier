"""Import des modeles "Smart Segmentation" d'IPSDK Explorer (Reactiv'IP).

Un modele IPSDK Explorer est un **dossier** (sous Windows :
``%APPDATA%/ReactivIP/Explorer/Machine_Learning/Pixel_Classification/<modele>``) :

* ``Settings.mho``   XML : grille de features, classes (nom, couleur, valeur),
                     liste des images d'entrainement ;
* ``Model``          foret scikit-learn (``RandomForestClassifier``) serialisee
                     par joblib (scikit-learn 0.24 dans IPSDK Explorer 4.0) ;
* ``ModelIPSDK.bin`` la meme foret convertie au format IPSDK (inutile ici) ;
* ``<image>/Image.tif`` et ``<image>/ImageDraw.tif`` : images d'entrainement et
  annotations (0 = non annote, k+1 = classe k).

On relit la foret scikit-learn sans dependre de sa version (capture de l'etat
brut des arbres puis reconstruction pour la version installee), on reproduit
la grille de features IPSDK (:class:`~xwiclassifier.features.ipsdk.IPSDKFeatures`)
et on verifie l'ensemble sur les pixels annotes stockes dans le dossier.
"""
from __future__ import annotations

import contextlib
import logging
import os
import xml.etree.ElementTree as ET

import numpy as np

from ..features.ipsdk import IPSDK_ALL, IPSDKFeatures

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Settings.mho
# --------------------------------------------------------------------------
def _child(el, tag):
    if el is None:
        return None
    for c in el:
        if c.tag == tag:
            return c
    return None


def _text(el, tag):
    c = _child(el, tag)
    if c is None or c.text is None:
        return None
    t = c.text.strip()
    if c.get("ASCII") == "True" and t:
        try:
            t = "".join(chr(int(x)) for x in t.split("."))
        except ValueError:
            pass
    return t


def read_settings(folder: str) -> dict:
    """Lit Settings.mho : features, classes, images d'entrainement."""
    root = ET.parse(os.path.join(folder, "Settings.mho")).getroot()
    fe = _child(root, "Features")
    if fe is None:
        raise ValueError("Settings.mho sans section <Features> : ce n'est pas un modele de pixel classification.")
    sizes = [int(x) for x in (_text(fe, "All_Sizes") or "").split(",") if x.strip()]
    res = [int(x) for x in (_text(fe, "MultiRes") or "1").split(",") if x.strip()] or [1]
    dimension = _text(fe, "Dimension") or "2D"
    order, flags = [], {}
    for c in fe:  # ordre du fichier = ordre utilise par IPSDK Explorer a l'entrainement
        name = c.tag.replace("_", " ")
        if name in IPSDK_ALL:
            order.append(name)
            flags[name] = [v.strip() == "True" for v in (c.text or "").split(",")]
    classes = []
    lc = _child(root, "LabelClasses")
    n = int(_text(lc, "NumberLabels") or 0) if lc is not None else 0
    for i in range(n):
        el = _child(lc, f"Label_{i}")
        if el is None:
            continue
        col = _text(el, "Color")
        val = _text(el, "Value")
        classes.append({"index": i, "name": _text(el, "Name") or f"Label {i + 1}",
                        "color": tuple(int(float(x)) for x in col.split(",")) if col else None,
                        "value": int(val) if val not in (None, "") else i})
    images = []
    ae = _child(root, "AllElements")
    if ae is not None:
        for c in ae:
            if c.tag == "Image":
                nm = _text(c, "Name")
                if nm:
                    images.append(nm)
    return {"sizes": sizes, "resolutions": res, "dimension": dimension, "order": order,
            "flags": flags, "classes": classes, "images": images}


def ipsdk_features_from_settings(settings: dict, **kw) -> IPSDKFeatures:
    return IPSDKFeatures.from_settings(settings["sizes"], settings["flags"], settings["resolutions"],
                                       settings["dimension"], order=settings["order"], **kw)


# --------------------------------------------------------------------------
# Lecture de la foret scikit-learn, independante de la version
# --------------------------------------------------------------------------
class _Stub:
    """Remplace toute classe scikit-learn au depicklage : capture arguments et etat."""

    def __init__(self, *args, **kwargs):
        self._args = args

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        self._state = state


_STUBS: dict = {}


def _stub_class(module: str, name: str):
    key = (module, name)
    if key not in _STUBS:
        _STUBS[key] = type(name, (_Stub,), {"_sk_module": module})
    return _STUBS[key]


@contextlib.contextmanager
def _sklearn_stubs():
    import joblib.numpy_pickle as jnp

    original = jnp.NumpyUnpickler.find_class

    def find_class(self, module, name):
        if module.split(".")[0] == "sklearn":
            return _stub_class(module, name)
        return original(self, module, name)

    jnp.NumpyUnpickler.find_class = find_class
    try:
        yield
    finally:
        jnp.NumpyUnpickler.find_class = original


def load_sklearn_forest(path: str):
    """Charge un RandomForestClassifier picke (toute version) -> (foret moderne, classes)."""
    import joblib

    from ..classifier.forest_build import forest_from_trees, tree_from_arrays

    with _sklearn_stubs():
        obj = joblib.load(path)
    d = obj.__dict__
    if "estimators_" not in d:
        raise ValueError(f"{path} ne contient pas de foret d'arbres scikit-learn.")
    classes = np.asarray(d["classes_"])
    n_classes = len(classes)
    n_features = int(d.get("n_features_in_", d.get("n_features_", 0)))
    trees = []
    for est in d["estimators_"]:
        t = est.__dict__["tree_"]
        st = t._state
        nodes, values = st["nodes"], np.asarray(st["values"])
        nf = int(t._args[0]) if t._args else n_features
        n_features = n_features or nf
        nn = int(st["node_count"])
        nodes = nodes[:nn]
        vals = values[:nn, 0, :n_classes]
        counts = vals if "weighted_n_node_samples" not in nodes.dtype.names else \
            vals / np.maximum(vals.sum(1, keepdims=True), 1e-300) * nodes["weighted_n_node_samples"][:, None]
        trees.append(tree_from_arrays(nodes["left_child"], nodes["right_child"], nodes["feature"],
                                      nodes["threshold"], counts, vals, n_features, n_classes))
    return forest_from_trees(trees, n_features, n_classes), classes


# --------------------------------------------------------------------------
# Dossier de modele
# --------------------------------------------------------------------------
def _resolve_folder(path: str) -> str:
    path = os.fspath(path)
    if os.path.isfile(path):
        path = os.path.dirname(path)
    if not os.path.exists(os.path.join(path, "Settings.mho")):
        raise FileNotFoundError(f"{path} : pas de Settings.mho (dossier de modele IPSDK Explorer attendu).")
    return path


def is_ipsdk_model_folder(path: str) -> bool:
    try:
        _resolve_folder(path)
        return True
    except (FileNotFoundError, OSError):
        return False


def _canonical(arr: np.ndarray, dimension_hint: str | None = None):
    """Image IPSDK lue par tifffile -> (image spatiale [+ canal en dernier], channel_axis)."""
    a = np.squeeze(np.asarray(arr))
    if a.ndim == 3 and a.shape[-1] in (3, 4) and a.shape[0] > 4:
        return a, -1  # couleur (y, x, c)
    if a.ndim == 3 and a.shape[0] in (3, 4) and dimension_hint == "color":
        return np.moveaxis(a, 0, -1), -1
    return a, None


def ipsdk_annotations(path: str) -> list:
    """Images d'entrainement + annotations stockees dans le dossier du modele."""
    from .images import read_image

    folder = _resolve_folder(path)
    st = read_settings(folder)
    names = st["images"] or sorted(d for d in os.listdir(folder)
                                   if os.path.isfile(os.path.join(folder, d, "Image.tif")))
    out = []
    for nm in names:
        img_p = os.path.join(folder, nm, "Image.tif")
        if not os.path.exists(img_p):
            continue
        img, ch = _canonical(read_image(img_p))
        draw_p = os.path.join(folder, nm, "ImageDraw.tif")
        lab = np.squeeze(read_image(draw_p)).astype(np.uint16) if os.path.exists(draw_p) else None
        spatial = img.shape if ch is None else img.shape[:-1]
        if lab is not None and lab.shape != spatial:
            logger.warning("Annotations de %s ignorees (forme %s != %s)", nm, lab.shape, spatial)
            lab = None
        out.append({"name": nm, "image": img, "labels": lab, "channel_axis": ch})
    return out


def _agreement(model, anns, max_pixels: int = 200_000):
    """(exactitude, probabilite moyenne de la bonne classe) sur les pixels annotes."""
    from ..classifier import estimators as est_mod

    hits, total, psum = 0, 0, 0.0
    for a in anns:
        if a["labels"] is None or not a["labels"].any():
            continue
        X, y = _annotated_features(model.extractor, a)
        if len(y) > max_pixels:
            sel = np.random.default_rng(0).choice(len(y), max_pixels, replace=False)
            X, y = X[sel], y[sel]
        proba = est_mod.predict_proba(model.estimator, X)
        idx = np.searchsorted(model.classes_, y)
        ok = (idx < len(model.classes_)) & (model.classes_[np.minimum(idx, len(model.classes_) - 1)] == y)
        pred = model.classes_[np.argmax(proba, axis=1)]
        hits += int((pred == y).sum())
        psum += float(proba[np.arange(len(y))[ok], idx[ok]].sum())
        total += len(y)
    if not total:
        return None, None
    return hits / total, psum / total


def _annotated_features(extractor, ann):
    feats = extractor.compute(ann["image"], channel_axis=ann["channel_axis"])
    mask = ann["labels"] > 0
    return feats[mask], ann["labels"][mask].astype(np.int64)


def load_ipsdk_model(path: str, mode: str = "forest", calibrate: bool = True, features=None,
                     kind: str = "rf", params: dict | None = None):
    """Construit un PixelClassifier a partir d'un dossier de modele IPSDK Explorer.

    mode="forest"  : foret IPSDK (scikit-learn) reutilisee avec les features IPSDK reproduites ;
                     ``calibrate`` choisit automatiquement, d'apres les annotations du
                     dossier, les details non documentes (normalisation gaussienne, zoom).
    mode="retrain" : nouveau modele entraine sur les annotations du dossier
                     (``features`` : None = jeu xwi par defaut, "ipsdk" = grille IPSDK,
                     "equivalent" = familles xwi equivalentes, ou une FeatureSpec).
    """
    from ..classifier.model import PixelClassifier

    folder = _resolve_folder(path)
    st = read_settings(folder)
    by_index = {c["index"]: c for c in st["classes"]}
    anns = ipsdk_annotations(folder)
    src = {"type": "ipsdk-explorer", "path": os.path.abspath(folder), "mode": mode,
           "ipsdk_values": {c["index"] + 1: c["value"] for c in st["classes"]}}

    if mode == "retrain":
        usable = [a for a in anns if a["labels"] is not None and a["labels"].any()]
        if not usable:
            raise ValueError("Aucune annotation (ImageDraw.tif) dans le dossier du modele.")
        if isinstance(features, str) and features == "ipsdk":
            features = ipsdk_features_from_settings(st)
        elif isinstance(features, str) and features == "equivalent":
            features = to_xwi_spec(st)
        m = PixelClassifier(features=features, kind=kind, params=params,
                            class_names={i + 1: c["name"] for i, c in by_index.items()},
                            class_colors={i + 1: c["color"] for i, c in by_index.items() if c["color"]})
        chs = {a["channel_axis"] for a in usable}
        m.fit([a["image"] for a in usable], [a["labels"] for a in usable], channel_axis=chs.pop())
        m.source = src
        return m

    if mode != "forest":
        raise ValueError("mode doit etre 'forest' ou 'retrain'")
    model_path = os.path.join(folder, "Model")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Fichier 'Model' absent : le modele n'a pas ete entraine/sauve dans "
                                "IPSDK Explorer. Utilisez mode='retrain' avec les annotations.")
    forest, sk_classes = load_sklearn_forest(model_path)
    extractor = ipsdk_features_from_settings(st)
    sample = next((a for a in anns), None)
    if sample is not None:
        ndim = sample["image"].ndim - (0 if sample["channel_axis"] is None else 1)
        n_ch = 1 if sample["channel_axis"] is None else sample["image"].shape[-1]
    else:
        ndim, n_ch = (3 if st["dimension"] == "3D" else 2), 1
        for c in (1, 3):
            if extractor.n_features(2, c) == forest.n_features_in_:
                n_ch = c
    if extractor.n_features(ndim, n_ch) != forest.n_features_in_:
        raise ValueError(f"La foret attend {forest.n_features_in_} features, la grille IPSDK en "
                         f"produit {extractor.n_features(ndim, n_ch)} : Settings.mho et Model incoherents ?")
    classes = np.asarray(sk_classes, np.int64) + 1  # indice de classe IPSDK k -> label xwi k+1
    m = PixelClassifier(features=extractor, kind="rf",
                        class_names={int(c): by_index.get(int(c) - 1, {}).get("name", f"Label {c}") for c in classes},
                        class_colors={int(c): by_index[int(c) - 1]["color"] for c in classes
                                      if int(c) - 1 in by_index and by_index[int(c) - 1]["color"]})
    m.estimator = forest
    m.classes_ = classes
    m.ndim_, m.n_channels_, m.n_features_ = ndim, n_ch, forest.n_features_in_
    m.source = src

    usable = [a for a in anns if a["labels"] is not None and a["labels"].any()]
    if usable:
        variants = [{"gaussian_normalize": True, "zoom_mode": "corner"}]
        if calibrate:
            uses_gauss = any(n in ("Gaussian", "Laplacian Difference") for _, n, _ in extractor.grid)
            uses_zoom = extractor.requires_full_image
            norms = (True, False) if uses_gauss else (True,)
            zooms = ("corner", "center") if uses_zoom else ("corner",)
            variants = [{"gaussian_normalize": g, "zoom_mode": z} for g in norms for z in zooms]
        scores = []
        for v in variants:
            m.extractor = IPSDKFeatures(extractor.grid, extractor.dimension, **v)
            acc, ptrue = _agreement(m, usable)
            scores.append((acc or 0.0, ptrue or 0.0, v))
        # critere : exactitude puis probabilite moyenne de la bonne classe (plus fin)
        best_acc, best_p, best_v = max(scores, key=lambda t: (round(t[0], 4), t[1]))
        m.extractor = IPSDKFeatures(extractor.grid, extractor.dimension, **best_v)
        src["annotation_agreement"] = best_acc
        src["annotation_confidence"] = best_p
        src["calibration"] = [{"variant": v, "agreement": a, "confidence": p} for a, p, v in scores]
    return m


def to_xwi_spec(settings: dict):
    """FeatureSpec xwi equivalente a la grille IPSDK (resolution 1 uniquement)."""
    from ..features import FeatureSpec

    fam = {"Gaussian": "gaussian", "Mean": "mean", "Laplacian Difference": "dog",
           "High Pass": "high_pass", "Variance": "box_variance", "Erosion": "erosion",
           "Dilation": "dilation", "Opening": "opening", "Closing": "closing"}
    pairs = [("intensity", 0.0)]
    for name in settings["order"]:
        for j, s in enumerate(settings["sizes"]):
            if j < len(settings["flags"][name]) and settings["flags"][name][j]:
                pairs.append((fam[name], float(s)))
    if any(r != 1 for r in settings["resolutions"]):
        logger.warning("Multi-resolution IPSDK non transposee dans le jeu xwi equivalent.")
    return FeatureSpec(pairs=tuple(pairs), morpho_shape="exact",
                       compute_in_2d=settings["dimension"] == "2D")


def ipsdk_output_values(model, seg: np.ndarray) -> np.ndarray:
    """Remappe une segmentation xwi vers les valeurs de sortie d'IPSDK (champ Value)."""
    vals = model.source.get("ipsdk_values") or {}
    out = seg.astype(np.int32, copy=True)
    for k, v in vals.items():
        out[seg == int(k)] = int(v)
    return out
