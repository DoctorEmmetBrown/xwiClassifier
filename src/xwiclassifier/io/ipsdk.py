"""Recuperation des modeles IPSDK Explorer (Reactiv'IP).

Les modeles "Smart Segmentation" d'IPSDK sont des objets ``RandomForestModel``
serialises dans un format binaire proprietaire (``PyIPSDK.readFromBinaryFile``),
non documente publiquement : on ne peut pas le relire directement. Trois voies :

1. **Distillation** (recommande, une seule fois, avec votre licence actuelle) :
   PyIPSDK applique votre modele sur des images representatives ; on echantillonne
   les pixels classes (de preference les plus surs) et on entraine un modele xwi
   equivalent, ensuite utilisable sans IPSDK. L'accord est mesure sur des pixels
   mis de cote.
2. **Annotations** : exportez vos annotations IPSDK Explorer en image de labels
   (TIFF) et entrainez directement (``xwi train``).
3. **Inspection** : ``inspect_binary`` resume le contenu du fichier (en-tete,
   chaines lisibles) pour aider a documenter le format.

Les appels PyIPSDK utilises sont ceux de la documentation publique
(``readFromBinaryFile``, ``loadTiffImageFile``,
``pixelClassificationRFWithProbabilitiesImg``) ; ils n'ont pas pu etre testes
sans licence IPSDK.
"""
from __future__ import annotations

import os
import re
import tempfile

import numpy as np


def ipsdk_available() -> bool:
    try:
        import PyIPSDK  # noqa: F401
        import PyIPSDK.IPSDKIPLMachineLearning  # noqa: F401
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# Inspection d'un fichier inconnu
# --------------------------------------------------------------------------
_MAGICS = {
    b"\x89HDF\r\n\x1a\n": "HDF5",
    b"PK\x03\x04": "archive ZIP",
    b"<?xml": "XML",
    b"\x80\x04": "pickle Python (protocole 4)",
    b"\x80\x05": "pickle Python (protocole 5)",
}


def inspect_binary(path: str, max_strings: int = 80, min_len: int = 4) -> dict:
    """Resume un fichier binaire : taille, en-tete, format probable, chaines lisibles."""
    with open(path, "rb") as fh:
        data = fh.read()
    kind = next((name for magic, name in _MAGICS.items() if data.startswith(magic)), None)
    if kind is None and b"serialization::archive" in data[:64]:
        kind = "archive boost::serialization"
    strings = re.findall(rb"[\x20-\x7e]{%d,}" % min_len, data)
    uniq = list(dict.fromkeys(s.decode("ascii", "replace") for s in strings))
    keywords = [s for s in uniq if re.search(r"(?i)forest|tree|feature|class|model|gauss|sigma|"
                                            r"hessian|gradient|laplac|struct|label|ipsdk", s)]
    return {"path": os.path.abspath(path), "size": len(data), "header_hex": data[:32].hex(" "),
            "format": kind or "inconnu (binaire proprietaire probable)",
            "keywords": keywords[:max_strings], "strings": uniq[:max_strings]}


# --------------------------------------------------------------------------
# Pont PyIPSDK
# --------------------------------------------------------------------------
def _to_ipsdk(image: np.ndarray):
    import PyIPSDK

    arr = np.ascontiguousarray(image)
    if hasattr(PyIPSDK, "fromArray"):
        try:
            return PyIPSDK.fromArray(arr)
        except Exception:
            pass
    import tifffile

    tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
    tmp.close()
    tifffile.imwrite(tmp.name, arr)
    try:
        return PyIPSDK.loadTiffImageFile(tmp.name)
    finally:
        os.unlink(tmp.name)


def _from_ipsdk(img) -> np.ndarray:
    if hasattr(img, "array"):
        try:
            return np.array(img.array)
        except Exception:
            pass
    import PyIPSDK
    import tifffile

    tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
    tmp.close()
    try:
        PyIPSDK.saveTiffImageFile(tmp.name, img)
        return tifffile.imread(tmp.name)
    finally:
        os.unlink(tmp.name)


def run_ipsdk_model(model_path: str, image: np.ndarray):
    """Applique un modele IPSDK (.bin) a une image. Renvoie (labels, probabilites|None)."""
    if not ipsdk_available():
        raise ImportError("PyIPSDK n'est pas installe (necessaire pour lire un modele IPSDK).")
    import PyIPSDK
    import PyIPSDK.IPSDKIPLMachineLearning as ml

    model, ok = PyIPSDK.readFromBinaryFile(model_path)
    if not ok:
        raise IOError(f"PyIPSDK n'a pas pu lire {model_path}")
    out_img, out_real = ml.pixelClassificationRFWithProbabilitiesImg(_to_ipsdk(image), model)
    labels = _from_ipsdk(out_img)
    try:
        proba = _from_ipsdk(out_real).astype(np.float32)
    except Exception:
        proba = None
    return labels, proba


def distill_ipsdk_model(model_path: str, images, features=None, kind: str = "rf",
                        params: dict | None = None, samples_per_class: int = 50_000,
                        confident_fraction: float = 0.8, holdout: float = 0.2,
                        channel_axis: int | None = None, random_state: int = 0,
                        predictor=None):
    """Entraine un modele xwi qui reproduit un modele IPSDK.

    ``predictor`` (optionnel, pour tests) : fonction image -> (labels, probas)
    remplacant l'appel a PyIPSDK. ``confident_fraction`` : part des pixels de
    chaque classe a la plus forte confiance parmi lesquels on echantillonne.
    """
    from ..classifier.model import PixelClassifier

    if isinstance(images, np.ndarray):
        images = [images]
    predictor = predictor or (lambda im: run_ipsdk_model(model_path, im))
    rng = np.random.default_rng(random_state)
    train_labels, held = [], []
    values = set()
    for img in images:
        lab, proba = predictor(img)
        lab = np.asarray(lab)
        spatial = img.shape if channel_axis is None else np.delete(img.shape, channel_axis)
        lab = lab.reshape(tuple(spatial)) if lab.size == int(np.prod(spatial)) else lab
        conf = None
        if proba is not None:
            proba = np.asarray(proba)
            conf = proba if proba.shape == lab.shape else proba.max(axis=-1).reshape(lab.shape)
        tl = np.zeros(lab.shape, np.uint16)
        hl = np.zeros(lab.shape, np.uint16)
        for v in np.unique(lab):
            values.add(int(v))
            idx = np.flatnonzero(lab.ravel() == v)
            if conf is not None and 0 < confident_fraction < 1 and idx.size > 10:
                c = conf.ravel()[idx]
                idx = idx[c >= np.quantile(c, 1 - confident_fraction)]
            n = min(idx.size, int(samples_per_class / (1 - holdout)) + 1)
            pick = rng.choice(idx, n, replace=False)
            n_hold = int(round(holdout * n))
            target = int(v) + 1  # xwi : 0 = non annote
            hl.ravel()[pick[:n_hold]] = target
            tl.ravel()[pick[n_hold:]] = target
        train_labels.append(tl)
        held.append(hl)

    names = {v + 1: f"IPSDK classe {v}" for v in sorted(values)}
    m = PixelClassifier(features=features, kind=kind, params=params, class_names=names)
    m.fit(list(images), train_labels, channel_axis=channel_axis)
    agree, total = 0, 0
    for img, hl in zip(images, held):
        mask = hl > 0
        if mask.any():
            seg = m.predict(img, channel_axis=channel_axis)
            agree += int((seg[mask] == hl[mask]).sum())
            total += int(mask.sum())
    m.source = {"type": "ipsdk-distilled", "path": os.path.abspath(model_path),
                "holdout_agreement": agree / total if total else None,
                "label_offset": 1}
    return m
