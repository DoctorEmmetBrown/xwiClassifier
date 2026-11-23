"""Lecture des projets ilastik (.ilp, HDF5) : features, annotations, donnees, forets.

Deux facons de recuperer un projet ilastik :
  * ``mode="forest"`` : reutilise tel quel le classifieur entraine par ilastik
    (forets vigra converties en scikit-learn) avec les features ilastik
    reproduites a l'identique -> memes predictions qu'ilastik, sans ilastik ;
  * ``mode="retrain"`` : relit les annotations (et les images) du projet et
    entraine un nouveau modele xwi, avec le jeu de features de votre choix
    (par defaut le jeu xwi, plus riche ; ou ``features="ilastik"``).
"""
from __future__ import annotations

import glob
import json
import logging
import os
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)
CANONICAL = "tzyxc"


def _s(v) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8")
    if isinstance(v, np.ndarray) and v.shape == ():
        return _s(v[()])
    return str(v)


def _axes_from_tags(tags) -> str:
    try:
        return "".join(a["key"] for a in json.loads(_s(tags))["axes"])
    except Exception:
        return ""


def to_canonical(arr: np.ndarray, axes: str):
    """Reordonne en t,z,y,x,c (axes presents uniquement). Renvoie (tableau, axes)."""
    axes = axes.lower()
    if len(axes) != arr.ndim:
        raise ValueError(f"Axes '{axes}' incompatibles avec la forme {arr.shape}")
    present = [a for a in CANONICAL if a in axes]
    arr = np.transpose(arr, [axes.index(a) for a in present])
    return arr, "".join(present)


def _parse_slicing(s: str):
    out = []
    for part in s.strip("[]() ").split(","):
        a, b = part.split(":")[:2]
        out.append(slice(int(a) if a else None, int(b) if b else None))
    return tuple(out)


@dataclass
class LaneInfo:
    index: int
    nickname: str
    location: str
    file_path: str
    axes: str
    shape: tuple


class IlastikProject:
    def __init__(self, path: str):
        import h5py

        self.path = os.fspath(path)
        self._f = h5py.File(self.path, "r")

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------
    @property
    def ilastik_version(self) -> str:
        return _s(self._f["ilastikVersion"][()]) if "ilastikVersion" in self._f else "?"

    @property
    def workflow(self) -> str:
        return _s(self._f["workflowName"][()]) if "workflowName" in self._f else "?"

    def feature_selection(self) -> dict | None:
        g = self._f.get("FeatureSelections")
        if g is None or "SelectionMatrix" not in g:
            return None
        ids = [_s(v) for v in g["FeatureIds"][()]]
        scales = [float(v) for v in g["Scales"][()]]
        c2d = [bool(v) for v in g["ComputeIn2d"][()]] if "ComputeIn2d" in g else [False] * len(scales)
        return {"feature_ids": ids, "scales": scales,
                "selection_matrix": np.asarray(g["SelectionMatrix"][()], bool), "compute_in_2d": c2d}

    def ilastik_features(self):
        from ..features.ilastik import IlastikFeatures

        fs = self.feature_selection()
        if fs is None:
            raise ValueError("Pas de selection de features dans ce projet.")
        return IlastikFeatures(**fs)

    def _pc(self):
        g = self._f.get("PixelClassification")
        if g is None:
            raise ValueError(f"Pas de PixelClassification dans le projet ({self.workflow}).")
        return g

    def label_names(self) -> list:
        g = self._pc()
        return [_s(v) for v in g["LabelNames"][()]] if "LabelNames" in g else []

    def label_colors(self) -> list:
        g = self._pc()
        return [tuple(int(x) for x in c) for c in g["LabelColors"][()]] if "LabelColors" in g else []

    # ------------------------------------------------------------------
    def lanes(self) -> list:
        infos = self._f.get("Input Data/infos")
        lanes = []
        if infos is None:
            return lanes
        for k, name in enumerate(sorted(infos.keys())):
            raw = infos[name].get("Raw Data")
            if raw is None or "filePath" not in raw:
                continue
            axes = _axes_from_tags(raw["axistags"][()]) if "axistags" in raw else ""
            shape = tuple(int(v) for v in raw["shape"][()]) if "shape" in raw else ()
            lanes.append(LaneInfo(k, _s(raw["nickname"][()]) if "nickname" in raw else name,
                                  _s(raw["location"][()]) if "location" in raw else "?",
                                  _s(raw["filePath"][()]), axes, shape))
        return lanes

    def spatial_info(self, lane: int = 0):
        """(ndim spatial, nb de canaux) d'une lane, sans lire les donnees."""
        li = self.lanes()[lane]
        axes = li.axes or ("yxc" if len(li.shape) == 3 else "zyxc")
        ndim = sum(1 for a in axes if a in "zyx" and li.shape[axes.index(a)] > 1)
        n_ch = li.shape[axes.index("c")] if "c" in axes else 1
        return max(ndim, 2), n_ch

    def read_raw(self, lane: int = 0, raw_path: str | None = None):
        """Donnees brutes d'une lane, en ordre canonique (t,z,y,x,c). Renvoie (tableau, axes)."""
        li = self.lanes()[lane]
        if raw_path is None and li.location == "ProjectInternal":
            ds = self._f[li.file_path]
            data = ds[()]
            axes = _axes_from_tags(ds.attrs.get("axistags", "")) or li.axes
        else:
            from .images import read_image

            path = raw_path or self._resolve(li.file_path)
            files = sorted(glob.glob(path)) if any(ch in path for ch in "*?") else [path]
            if not files or not os.path.exists(files[0].split(".h5/")[0].split(".hdf5/")[0]):
                raise FileNotFoundError(
                    f"Donnees de la lane {lane} introuvables : {path}. "
                    "Passez le chemin via raw_paths=[...] (ou --raw en ligne de commande).")
            data = np.stack([read_image(p) for p in files]) if len(files) > 1 else read_image(files[0])
            axes = li.axes if len(li.axes) == data.ndim else _guess_axes(data)
        return to_canonical(np.asarray(data), axes)

    def _resolve(self, p: str) -> str:
        if os.path.isabs(p):
            return p
        return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(self.path)), p))

    def read_labels(self, lane: int = 0, canonical_shape: tuple | None = None,
                    canonical_axes: str | None = None) -> np.ndarray | None:
        """Annotations d'une lane (ordre canonique, sans axe c). None si aucune."""
        g = self._pc().get("LabelSets")
        if g is None:
            return None
        names = sorted(g.keys())
        if lane >= len(names) or len(g[names[lane]]) == 0:
            return None
        if canonical_shape is None:
            li = self.lanes()[lane]
            cshape = to_canonical(np.empty(li.shape, np.uint8), li.axes)
            canonical_shape, canonical_axes = cshape[0].shape, cshape[1]
        caxes = canonical_axes
        full = np.zeros(canonical_shape, np.uint8)
        for bname, ds in g[names[lane]].items():
            data = ds["data"][()] if hasattr(ds, "keys") else ds[()]
            attrs = ds.attrs
            baxes = _axes_from_tags(attrs.get("axistags", "")) or caxes
            sl = _parse_slicing(_s(attrs["blockSlice"]))
            block, bax = to_canonical(np.asarray(data), baxes)
            slc = dict(zip(baxes, sl))
            target = tuple(slc.get(a, slice(None)) for a in bax)
            # ajuste aux axes canoniques de la lane (axes singletons absents d'un cote)
            idx, blk = [], block
            for a in caxes:
                if a in bax:
                    idx.append(target[bax.index(a)])
                else:
                    idx.append(slice(0, 1))
                    blk = np.expand_dims(blk, caxes.index(a))
            region = full[tuple(idx)]
            m = blk > 0
            region[m] = blk[m]
        if "c" in caxes:
            full = full[..., 0]
        return full

    def forests(self) -> dict | None:
        from ..classifier.vigra_forest import read_vigra_forest

        g = self._pc().get("ClassifierForests")
        if g is None:
            return None
        trees, nf, nc, labels = [], None, None, None
        for k in sorted(g.keys()):
            if k.startswith("Forest"):
                fo = read_vigra_forest(g[k])
                trees += fo["trees"]
                nf, nc, labels = fo["n_features"], fo["n_classes"], fo["labels"]
        if not trees:
            return None
        known = np.asarray(g["known_labels"][()]).ravel() if "known_labels" in g else labels
        return {"trees": trees, "n_features": nf, "n_classes": nc,
                "known_labels": np.asarray(known if known is not None else np.arange(1, nc + 1))}

    def annotations(self, lane: int = 0, raw_path: str | None = None):
        """(image, labels, axes) prets a l'emploi : t retire si unique, canal en dernier."""
        raw, axes = self.read_raw(lane, raw_path)
        lab = self.read_labels(lane, raw.shape, axes)
        if "t" in axes and raw.shape[0] == 1:
            raw, axes = raw[0], axes[1:]
            lab = None if lab is None else lab[0]
        return raw, lab, axes


def _guess_axes(data: np.ndarray) -> str:
    return {2: "yx", 3: "zyx", 4: "zyxc"}.get(data.ndim, "tzyxc"[-data.ndim:])


def _squeeze_for_model(raw: np.ndarray, axes: str, lab=None):
    """Retire t (=1), z (=1) et c (=1) ; renvoie (image, labels, channel_axis)."""
    for a in ("t", "z"):
        if a in axes and raw.shape[axes.index(a)] == 1:
            i = axes.index(a)
            raw = np.take(raw, 0, axis=i)
            if lab is not None:
                lab = np.take(lab, 0, axis=i)
            axes = axes.replace(a, "")
    ch = None
    if "c" in axes:
        if raw.shape[-1] == 1:
            raw = raw[..., 0]
        else:
            ch = -1
    return raw, lab, ch


def ilastik_annotations(path: str, lane: int = 0, raw_path: str | None = None):
    """Image + annotations + noms/couleurs d'un projet ilastik (pour napari ou la CLI)."""
    with IlastikProject(path) as p:
        raw, lab, axes = p.annotations(lane, raw_path)
        raw, lab, ch = _squeeze_for_model(raw, axes, lab)
        return {"image": raw, "labels": lab, "channel_axis": ch,
                "names": p.label_names(), "colors": p.label_colors()}


def load_ilastik_model(path: str, mode: str = "forest", features=None, kind: str = "rf",
                       params: dict | None = None, raw_paths=None, lanes=None):
    """Construit un PixelClassifier a partir d'un projet ilastik.

    mode="forest"  : classifieur ilastik reutilise (exact) ;
    mode="retrain" : nouveau modele entraine sur les annotations du projet
                     (``features`` : None = jeu xwi par defaut, "ilastik" = memes
                     features qu'ilastik, ou une FeatureSpec).
    """
    from ..classifier.model import PixelClassifier
    from ..classifier.vigra_forest import to_sklearn_forest

    with IlastikProject(path) as p:
        names, colors = p.label_names(), p.label_colors()
        class_names = {i + 1: n for i, n in enumerate(names)}
        class_colors = {i + 1: tuple(c) for i, c in enumerate(colors)}
        src = {"type": "ilastik", "path": os.path.abspath(path),
               "ilastik_version": p.ilastik_version, "mode": mode}

        if mode == "forest":
            fo = p.forests()
            if fo is None:
                raise ValueError("Ce projet ne contient pas de classifieur entraine : "
                                 "utilisez mode='retrain' (annotations) ou entrainez-le dans ilastik.")
            feats = p.ilastik_features()
            ndim, n_ch = p.spatial_info(0) if p.lanes() else (None, 1)
            candidates = [(ndim, n_ch)] + [(d, n_ch) for d in (2, 3) if d != ndim]
            match = next(((d, c) for d, c in candidates
                          if d and feats.n_features(d, c) == fo["n_features"]), None)
            if match is None:
                raise ValueError(f"Nombre de features incoherent : la foret en attend {fo['n_features']}.")
            m = PixelClassifier(features=feats, kind="rf", class_names=class_names,
                                class_colors=class_colors)
            m.estimator = to_sklearn_forest(fo["trees"], fo["n_features"], fo["n_classes"])
            m.classes_ = fo["known_labels"].astype(np.int64)
            m.ndim_, m.n_channels_ = match
            m.n_features_ = fo["n_features"]
            m.source = src
            return m

        if mode != "retrain":
            raise ValueError("mode doit etre 'forest' ou 'retrain'")
        if isinstance(features, str) and features == "ilastik":
            features = p.ilastik_features()
        images, labels, ch = [], [], None
        lane_ids = lanes if lanes is not None else range(len(p.lanes()))
        for k, lane in enumerate(lane_ids):
            rp = raw_paths[k] if raw_paths else None
            raw, axes = p.read_raw(lane, rp)
            lab = p.read_labels(lane, raw.shape, axes)
            if lab is None or not lab.any():
                continue
            ts = range(raw.shape[0]) if "t" in axes else [None]
            for t in ts:
                r = raw if t is None else raw[t]
                lb = lab if t is None else lab[t]
                ax = axes if t is None else axes[1:]
                if not lb.any():
                    continue
                r, lb, ch = _squeeze_for_model(r, ax, lb)
                images.append(r)
                labels.append(lb)
        if not images:
            raise ValueError("Aucune annotation trouvee dans le projet.")
    m = PixelClassifier(features=features, kind=kind, params=params,
                        class_names=class_names, class_colors=class_colors)
    m.fit(images, labels, channel_axis=ch)
    m.source = src
    return m
