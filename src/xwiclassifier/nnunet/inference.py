"""Segmentation avec un modele nnU-Net entraine, meme interface que PixelClassifier.

* normalisation calculee une fois sur l'image complete (comme a l'export) ;
* prediction par blocs avec une marge d'une demi-fenetre nnU-Net : volumes de
  taille quelconque, sans raccords visibles ;
* labels nnU-Net 0..K-1 rendus en labels xwi 1..K (1 = background).
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

from ..blocks import default_block_shape, iter_blocks, normalize_region, with_halo
from .workspace import TRAINERS_DIR, apply_volume_normalization, fit_volume_normalization

POINTER_FORMAT = "xwi-nnunet"


def is_nnunet_model_folder(path) -> bool:
    p = Path(path)
    return p.is_dir() and (p / "plans.json").exists() and (p / "dataset.json").exists()


def find_model_folder(path) -> Path:
    """Accepte un dossier de modele, un dossier de resultats de jeu de donnees ou un pointeur .json."""
    p = Path(path)
    if p.is_file() and p.suffix == ".json":
        d = json.loads(p.read_text())
        if d.get("format") != POINTER_FORMAT:
            raise ValueError(f"{p} n'est pas un pointeur de modele nnU-Net xwi.")
        return Path(d["model_folder"])
    if is_nnunet_model_folder(p):
        return p
    cands = [m for m in p.glob("*__*__*") if is_nnunet_model_folder(m)] if p.is_dir() else []
    if not cands:
        raise FileNotFoundError(f"Aucun modele nnU-Net dans {p}")
    return max(cands, key=lambda m: max((f.stat().st_mtime for f in m.glob("fold_*/checkpoint_*.pth")), default=0))


class NnUNetSegmenter:
    """Modele nnU-Net utilisable comme un PixelClassifier (predict, predict_proba, describe)."""

    kind = "nnunet"
    extractor = None
    feature_spec = None

    def __init__(self, model_folder, folds=None, checkpoint: str = "checkpoint_final.pth",
                 device: str = "auto", tile_step_size: float = 0.5, use_mirroring: bool = True):
        self.model_folder = find_model_folder(model_folder)
        self.folds = folds
        self.checkpoint = checkpoint
        self.device = device
        self.tile_step_size = tile_step_size
        self.use_mirroring = use_mirroring
        self.dataset_json = json.loads((self.model_folder / "dataset.json").read_text())
        self.plans = json.loads((self.model_folder / "plans.json").read_text())
        self.configuration = self.model_folder.name.split("__")[-1]
        cfg = self.plans["configurations"][self.configuration]
        self.patch_size = list(cfg["patch_size"])
        self.target_spacing = list(cfg["spacing"])
        xwi = self.dataset_json.get("xwi", {})
        labels = {k: v for k, v in self.dataset_json["labels"].items() if k != "ignore"}
        k = len(labels)
        self.classes_ = np.arange(1, k + 1)
        names = {int(a): b for a, b in xwi.get("class_names", {}).items()}
        inv = {int(v): n for n, v in labels.items()}
        self.class_names = {i + 1: names.get(i + 1, inv.get(i, f"classe {i + 1}")) for i in range(k)}
        self.class_colors = {int(a): tuple(b) for a, b in xwi.get("class_colors", {}).items()}
        self.normalization = xwi.get("normalization", "volume")
        self.ndim_ = xwi.get("ndim") or (2 if self.configuration == "2d" else 3)
        self.n_channels_ = len(self.dataset_json["channel_names"])
        self.source = {"type": "nnunet", "model_folder": str(self.model_folder),
                       "configuration": self.configuration}
        self.training_info = {}
        self._predictor = None

    # ------------------------------------------------------------------
    @property
    def is_fitted(self) -> bool:
        return True

    def class_label(self, value: int) -> str:
        return self.class_names.get(int(value), f"classe {value}")

    def feature_importances(self, group=None):
        return None

    def available_folds(self) -> list:
        return sorted(f.name[5:] for f in self.model_folder.glob("fold_*") if (f / self.checkpoint).exists())

    def _get_predictor(self):
        if self._predictor is None:
            import torch
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

            from .training import resolve_device

            ext = os.environ.get("nnUNet_extTrainer", "")
            if TRAINERS_DIR not in ext.split(os.pathsep):
                os.environ["nnUNet_extTrainer"] = os.pathsep.join([TRAINERS_DIR] + ([ext] if ext else []))
            dev = torch.device(resolve_device(self.device))
            pred = nnUNetPredictor(tile_step_size=self.tile_step_size, use_gaussian=True,
                                   use_mirroring=self.use_mirroring, device=dev, verbose=False,
                                   verbose_preprocessing=False, allow_tqdm=False)
            folds = self.folds
            if folds is None:
                folds = self.available_folds()
                if not folds:
                    raise FileNotFoundError(f"Aucun fold avec {self.checkpoint} dans {self.model_folder} "
                                            "(entrainement non termine ? essayez checkpoint='checkpoint_best.pth').")
            folds = tuple(f if f == "all" else int(f) for f in folds)
            pred.initialize_from_trained_model_folder(str(self.model_folder), use_folds=folds,
                                                      checkpoint_name=self.checkpoint)
            self._predictor = pred
        return self._predictor

    # ------------------------------------------------------------------
    def _spacing3(self, spacing, ndim):
        if spacing is None:
            ts = self.target_spacing
            spacing = ts if len(ts) == ndim else ([1.0] + list(ts) if ndim == 3 else ts[-2:])
        spacing = [float(s) for s in spacing]
        return ([999.0] + spacing) if ndim == 2 else spacing

    def halo(self, ndim: int, spacing=None) -> tuple:
        sp = self._spacing3(spacing, ndim)
        ps, ts = list(self.patch_size), list(self.target_spacing)
        if len(ps) == 2:  # modele 2D : pas de contexte en z
            ps, ts = [1] + ps, [sp[0]] + ts
        h = [int(math.ceil(p * t / s / 2)) + 1 for p, t, s in zip(ps, ts, sp)]
        if len(self.patch_size) == 2:
            h[0] = 0
        return tuple(h[-ndim:])

    def predict_proba(self, image: np.ndarray, channel_axis: int | None = None, region=None,
                      block_shape: tuple | None = None, spacing=None, smooth: float = 0.0,
                      norm=None, progress=None) -> np.ndarray:
        img = np.asarray(image)
        img = img[..., None] if channel_axis is None else np.moveaxis(img, channel_axis, -1)
        spatial, n_ch = img.shape[:-1], img.shape[-1]
        ndim = len(spatial)
        if n_ch != self.n_channels_:
            raise ValueError(f"Le modele attend {self.n_channels_} canal/canaux, l'image en a {n_ch}.")
        if norm is None:
            norm = [fit_volume_normalization(img[..., c], self.normalization) for c in range(n_ch)]
        region = normalize_region(region, spatial)
        rshape = tuple(r.stop - r.start for r in region)
        halo = self.halo(ndim, spacing)
        K = len(self.classes_)
        bs = block_shape or default_block_shape(rshape, K + n_ch + 24, halo, n_threads=1)
        bs = tuple(min(b, s) for b, s in zip(bs, rshape))
        sp = self._spacing3(spacing, ndim)
        pred = self._get_predictor()
        out = np.empty(rshape + (K,), np.float32)
        blocks = list(iter_blocks(rshape, bs))
        for i, rel in enumerate(blocks):
            inner = tuple(slice(r.start + b.start, r.start + b.stop) for r, b in zip(region, rel))
            outer, local = with_halo(inner, halo, spatial)
            data = np.stack([apply_volume_normalization(img[outer + (c,)], norm[c])
                             if self.normalization == "volume" else img[outer + (c,)].astype(np.float32)
                             for c in range(n_ch)])
            if ndim == 2:
                data = data[:, None]
            _, probs = pred.predict_single_npy_array(data, {"spacing": sp}, None, None, True)
            probs = np.asarray(probs, np.float32)
            if ndim == 2:
                probs = probs[:, 0]
            out[rel] = np.moveaxis(probs, 0, -1)[local]
            if progress is not None:
                progress(i + 1, len(blocks))
        return out

    def predict(self, image: np.ndarray, channel_axis: int | None = None, region=None,
                block_shape: tuple | None = None, spacing=None, smooth: float = 0.0,
                return_proba: bool = False, return_uncertainty: bool = False, progress=None):
        from ..classifier.model import labels_from_proba, uncertainty

        proba = self.predict_proba(image, channel_axis=channel_axis, region=region,
                                   block_shape=block_shape, spacing=spacing, progress=progress)
        seg = labels_from_proba(proba, self.classes_)
        if not (return_proba or return_uncertainty):
            return seg
        res = [seg]
        if return_proba:
            res.append(proba)
        if return_uncertainty:
            res.append(uncertainty(proba))
        return tuple(res)

    # ------------------------------------------------------------------
    def describe(self) -> str:
        from .training import training_status

        st = training_status(self.model_folder)
        lines = [f"Modele nnU-Net : {self.model_folder}",
                 f"  configuration : {self.configuration}, fenetre {self.patch_size}, "
                 f"espacement cible {self.target_spacing}",
                 f"  normalisation : {self.normalization} ; donnees {self.ndim_}D, {self.n_channels_} canal/canaux",
                 "  classes       : " + ", ".join(f"{int(c)}={self.class_label(c)}" for c in self.classes_)]
        for f, info in st["folds"].items():
            lines.append(f"  fold {f:>3} : epoque {info['epoch']}, meilleur Dice EMA {info['best_ema_dice']}, "
                         f"{'termine' if info['finished'] else 'en cours / interrompu'}")
        return "\n".join(lines)

    def save(self, path: str) -> None:
        """Ecrit un pointeur .json vers le dossier du modele (les poids restent dans nnUNet_results)."""
        Path(path).write_text(json.dumps({"format": POINTER_FORMAT, "model_folder": str(self.model_folder),
                                          "folds": self.folds, "checkpoint": self.checkpoint}, indent=2))
