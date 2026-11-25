"""Espace de travail nnU-Net et jeux de donnees construits a partir des annotations xwi.

Un espace de travail regroupe les trois dossiers attendus par nnU-Net
(``nnUNet_raw``, ``nnUNet_preprocessed``, ``nnUNet_results``) et un jeu de donnees
par organe / structure (ex. ``Dataset501_ReinVaisseaux``).

Conventions :
  * labels xwi : 0 = non annote, 1 = background, 2.. = classes ;
  * labels nnU-Net : 0 = background, 1.. = classes, ``ignore`` = plus grande valeur.
    Les voxels non annotes deviennent ``ignore`` : nnU-Net s'entraine alors sur des
    annotations partielles (traits, quelques coupes) sans les compter comme du fond ;
  * deux facons d'ajouter des cas : ``sparse`` (annotations partielles, recadrees
    autour des traits) et ``dense`` (region validee par l'utilisateur, entierement
    etiquetee ; une marge de contexte autour est marquee ``ignore``) ;
  * normalisation calculee une fois sur l'image complete (``volume`` par defaut),
    ce qui permet ensuite une inference par blocs sans raccords.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NORMALIZATIONS = ("volume", "ct", "none")
TRAINERS_DIR = str(Path(__file__).parent / "trainers")


def default_root() -> Path:
    return Path(os.environ.get("XWI_NNUNET_ROOT", Path.home() / "xwi_nnunet"))


# --------------------------------------------------------------------------
# Normalisation (parametres calcules sur l'image complete)
# --------------------------------------------------------------------------
def fit_volume_normalization(image: np.ndarray, method: str = "volume", max_samples: int = 2_000_000) -> dict:
    """Parametres de normalisation d'un canal : z-score robuste (percentiles 0.5-99.5)."""
    if method != "volume":
        return {"method": method}
    flat = np.asarray(image).ravel()
    if flat.size > max_samples:
        flat = flat[np.random.default_rng(0).choice(flat.size, max_samples, replace=False)]
    flat = flat.astype(np.float64)
    lo, hi = np.percentile(flat, [0.5, 99.5])
    c = np.clip(flat, lo, hi)
    sd = float(c.std()) or 1.0
    return {"method": "volume", "lo": float(lo), "hi": float(hi), "mean": float(c.mean()), "std": sd}


def apply_volume_normalization(image: np.ndarray, params: dict) -> np.ndarray:
    a = np.asarray(image, dtype=np.float32)
    if params.get("method") != "volume":
        return a
    return ((np.clip(a, params["lo"], params["hi"]) - params["mean"]) / params["std"]).astype(np.float32)


# --------------------------------------------------------------------------
# Ecriture / lecture (SimpleITK, espacement inclus)
# --------------------------------------------------------------------------
def write_sitk(path: str, array: np.ndarray, spacing) -> None:
    import SimpleITK as sitk

    img = sitk.GetImageFromArray(np.ascontiguousarray(array))
    img.SetSpacing(tuple(float(s) for s in reversed(spacing)))  # SimpleITK : (x, y, z)
    sitk.WriteImage(img, str(path), useCompression=True)


def read_sitk(path: str):
    import SimpleITK as sitk

    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img), tuple(reversed(img.GetSpacing()))


def _safe(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "", name.title()) or "Data"
    return s[:40]


# --------------------------------------------------------------------------
@dataclass
class DatasetInfo:
    id: int
    name: str
    folder: Path
    n_cases: int
    classes: dict


class NnUNetWorkspace:
    def __init__(self, root: str | os.PathLike | None = None):
        self.root = Path(root) if root else default_root()
        self.raw = self.root / "nnUNet_raw"
        self.preprocessed = self.root / "nnUNet_preprocessed"
        self.results = self.root / "nnUNet_results"
        self.logs = self.root / "logs"
        for d in (self.raw, self.preprocessed, self.results, self.logs):
            d.mkdir(parents=True, exist_ok=True)

    def env(self, extra: dict | None = None) -> dict:
        env = dict(os.environ)
        env.update(nnUNet_raw=str(self.raw), nnUNet_preprocessed=str(self.preprocessed),
                   nnUNet_results=str(self.results))
        ext = env.get("nnUNet_extTrainer", "")
        env["nnUNet_extTrainer"] = os.pathsep.join([TRAINERS_DIR] + ([ext] if ext else []))
        env.update({k: str(v) for k, v in (extra or {}).items()})
        return env

    def activate(self) -> None:
        """Positionne les variables nnU-Net dans le processus courant (inference en Python)."""
        os.environ.update({k: v for k, v in self.env().items()
                           if k.startswith("nnUNet_")})

    # ------------------------------------------------------------------
    def datasets(self) -> list[DatasetInfo]:
        out = []
        for d in sorted(self.raw.glob("Dataset[0-9][0-9][0-9]_*")):
            try:
                ds = NnUNetDataset(self, d)
                out.append(DatasetInfo(ds.id, ds.name, d, ds.n_cases, ds.class_names))
            except Exception:
                continue
        return out

    def next_id(self, start: int = 501) -> int:
        used = {int(p.name[7:10]) for p in list(self.raw.glob("Dataset*")) + list(self.results.glob("Dataset*"))
                if p.name[7:10].isdigit()}
        i = start
        while i in used:
            i += 1
        if i > 999:
            raise RuntimeError("Plus d'identifiant de jeu de donnees libre (501-999).")
        return i

    def get_dataset(self, name_or_id) -> "NnUNetDataset":
        key = str(name_or_id)
        for d in self.raw.glob("Dataset[0-9][0-9][0-9]_*"):
            if key in (d.name, d.name[7:10], d.name[11:]) or (key.isdigit() and int(key) == int(d.name[7:10])):
                return NnUNetDataset(self, d)
        raise KeyError(f"Jeu de donnees introuvable : {name_or_id} (dans {self.raw})")

    def create_dataset(self, name: str, class_names: dict, class_colors: dict | None = None,
                       n_channels: int = 1, normalization: str = "volume", dataset_id: int | None = None,
                       file_ending: str = ".nii.gz", description: str = "") -> "NnUNetDataset":
        """Cree un jeu de donnees. ``class_names`` : {valeur xwi (>=1): nom}, 1 = background."""
        if normalization not in NORMALIZATIONS:
            raise ValueError(f"normalization parmi {NORMALIZATIONS}")
        values = sorted(int(v) for v in class_names)
        if values != list(range(1, len(values) + 1)) or len(values) < 2:
            raise ValueError("Classes xwi attendues : 1 (background), 2, 3... consecutives.")
        did = dataset_id or self.next_id()
        folder = self.raw / f"Dataset{did:03d}_{_safe(name)}"
        if folder.exists():
            raise FileExistsError(f"{folder} existe deja.")
        (folder / "imagesTr").mkdir(parents=True)
        (folder / "labelsTr").mkdir()
        labels, used = {"background": 0}, {"background", "ignore"}
        for v in values[1:]:
            nm = _safe(class_names[v]) or f"Class{v - 1}"
            while nm.lower() in used:
                nm += "_"
            used.add(nm.lower())
            labels[nm] = v - 1
        labels["ignore"] = len(values)
        chan = "CT" if normalization == "ct" else "noNorm"
        ds_json = {
            "name": name, "description": description,
            "channel_names": {str(c): chan for c in range(n_channels)},
            "labels": labels, "numTraining": 0, "file_ending": file_ending,
            "xwi": {"class_names": {str(int(v)): class_names[v] for v in values},
                    "class_colors": {str(int(k)): list(c) for k, c in (class_colors or {}).items()},
                    "normalization": normalization, "ndim": None, "n_channels": n_channels,
                    "cases": [], "created": _dt.datetime.now().isoformat(timespec="seconds")},
        }
        (folder / "dataset.json").write_text(json.dumps(ds_json, indent=2, ensure_ascii=False))
        return NnUNetDataset(self, folder)

    def trained_models(self, dataset=None) -> list[Path]:
        """Dossiers de modeles entraines (contenant plans.json et au moins un fold)."""
        roots = [self.results / dataset.folder.name] if dataset is not None else list(self.results.glob("Dataset*"))
        out = []
        for r in roots:
            for m in sorted(Path(r).glob("*__*__*")):
                if (m / "plans.json").exists() and any(m.glob("fold_*")):
                    out.append(m)
        return out


class NnUNetDataset:
    def __init__(self, workspace: NnUNetWorkspace, folder: os.PathLike):
        self.ws = workspace
        self.folder = Path(folder)
        self.json_path = self.folder / "dataset.json"
        self.meta = json.loads(self.json_path.read_text())
        self.id = int(self.folder.name[7:10])
        self.name = self.folder.name

    # ------------------------------------------------------------------
    @property
    def xwi(self) -> dict:
        return self.meta.setdefault("xwi", {})

    @property
    def class_names(self) -> dict:
        return {int(k): v for k, v in self.xwi.get("class_names", {}).items()}

    @property
    def n_cases(self) -> int:
        return int(self.meta.get("numTraining", 0))

    @property
    def ignore_label(self) -> int:
        return int(self.meta["labels"]["ignore"])

    @property
    def ndim(self):
        return self.xwi.get("ndim")

    @property
    def default_configuration(self) -> str:
        return "2d" if self.ndim == 2 else "3d_fullres"

    def save(self) -> None:
        self.json_path.write_text(json.dumps(self.meta, indent=2, ensure_ascii=False))

    def cases(self) -> list:
        return list(self.xwi.get("cases", []))

    # ------------------------------------------------------------------
    def add_case(self, image: np.ndarray, labels: np.ndarray, spacing=None, channel_axis: int | None = None,
                 roi=None, margin: int = 16, mode: str = "sparse", norm: list | None = None,
                 source: str = "", case_name: str | None = None) -> str:
        """Ajoute un cas d'entrainement a partir d'une image et d'annotations xwi.

        mode="sparse" : seuls les voxels annotes (>0) comptent ; recadrage automatique
                        autour des annotations (+ ``margin`` de contexte).
        mode="dense"  : ``roi`` (tuple de slices) est une region validee, entierement
                        etiquetee ; la marge de contexte autour est marquee ``ignore``.
        ``norm`` : parametres de normalisation deja calcules sur l'image complete.
        """
        img = np.asarray(image)
        img = img[..., None] if channel_axis is None else np.moveaxis(img, channel_axis, -1)
        n_ch, spatial = img.shape[-1], img.shape[:-1]
        ndim = len(spatial)
        lab = np.asarray(labels)
        if lab.shape != spatial:
            raise ValueError(f"Annotations {lab.shape} incompatibles avec l'image {spatial}.")
        if ndim not in (2, 3):
            raise ValueError("Images 2D ou 3D attendues.")
        if self.ndim is not None and self.ndim != ndim:
            raise ValueError(f"Ce jeu de donnees contient des images {self.ndim}D.")
        if n_ch != int(self.xwi.get("n_channels", 1)):
            raise ValueError(f"Ce jeu de donnees attend {self.xwi.get('n_channels', 1)} canal/canaux.")
        n_cls = len(self.class_names)
        if lab.max() > n_cls:
            raise ValueError(f"Valeur d'annotation {int(lab.max())} > nombre de classes ({n_cls}).")
        spacing = tuple(float(s) for s in (spacing or (1.0,) * ndim))
        if len(spacing) != ndim:
            raise ValueError("spacing doit avoir une valeur par axe spatial.")

        # region a exporter
        if roi is not None:
            inner = tuple(slice(*r.indices(s)[:2]) for r, s in zip(roi, spatial))
        elif mode == "sparse":
            idx = np.argwhere(lab > 0)
            if idx.size == 0:
                raise ValueError("Aucun voxel annote.")
            lo, hi = idx.min(0), idx.max(0) + 1
            inner = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        else:
            inner = tuple(slice(0, s) for s in spatial)
        outer = tuple(slice(max(0, r.start - margin), min(s, r.stop + margin)) for r, s in zip(inner, spatial))
        local = tuple(slice(r.start - o.start, r.stop - o.start) for r, o in zip(inner, outer))

        method = self.xwi.get("normalization", "volume")
        if norm is None:
            norm = [fit_volume_normalization(img[..., c], method) for c in range(n_ch)]
        crop_lab = lab[outer].astype(np.int32)
        out_lab = np.full(crop_lab.shape, self.ignore_label, np.uint8 if self.ignore_label < 255 else np.uint16)
        valid = crop_lab > 0
        if mode == "dense" and roi is not None:
            inside = np.zeros(crop_lab.shape, bool)
            inside[local] = True
            valid &= inside
        out_lab[valid] = crop_lab[valid] - 1
        if not valid.any():
            raise ValueError("La region exportee ne contient aucun voxel etiquete.")

        case = case_name or f"case_{len(self.cases()):04d}"
        ending = self.meta["file_ending"]
        for c in range(n_ch):
            data = img[outer + (c,)]
            data = apply_volume_normalization(data, norm[c]) if method == "volume" else data.astype(np.float32)
            write_sitk(self.folder / "imagesTr" / f"{case}_{c:04d}{ending}", data, spacing)
        write_sitk(self.folder / "labelsTr" / f"{case}{ending}", out_lab, spacing)

        self.xwi["ndim"] = ndim
        self.xwi.setdefault("cases", []).append({
            "case": case, "mode": mode, "source": source, "shape": list(out_lab.shape),
            "spacing": list(spacing), "origin": [int(o.start) for o in outer],
            "labeled_voxels": int(valid.sum()),
            "per_class": {self.class_names[int(k) + 1]: int((out_lab == k).sum()) for k in range(n_cls)},
            "added": _dt.datetime.now().isoformat(timespec="seconds")})
        self.meta["numTraining"] = len(self.xwi["cases"])
        self.save()
        return case

    def remove_case(self, case: str) -> None:
        ending = self.meta["file_ending"]
        for p in list((self.folder / "imagesTr").glob(f"{case}_*{ending}")) + [self.folder / "labelsTr" / f"{case}{ending}"]:
            if p.exists():
                p.unlink()
        self.xwi["cases"] = [c for c in self.cases() if c["case"] != case]
        self.meta["numTraining"] = len(self.xwi["cases"])
        self.save()

    def summary(self) -> str:
        lines = [f"{self.name} : {self.n_cases} cas, {self.ndim}D, normalisation {self.xwi.get('normalization')}",
                 "  classes : " + ", ".join(f"{k}={v}" for k, v in sorted(self.class_names.items()))]
        for c in self.cases():
            lines.append(f"  - {c['case']} [{c['mode']}] {c['shape']} {c['labeled_voxels']} voxels etiquetes "
                         f"{c.get('source', '')}")
        return "\n".join(lines)
