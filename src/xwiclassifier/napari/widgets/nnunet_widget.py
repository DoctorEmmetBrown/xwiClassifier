"""Widget napari nnU-Net : constituer un jeu de donnees, entrainer, predire.

Flux : choisir l'organe / la structure (jeu de donnees), ajouter des annotations
partielles ou des regions validees (segmentation corrigee), preparer, entrainer
(en arriere-plan, folds repartis sur les GPU), puis predire la vue ou le volume.
"""
from __future__ import annotations

import numpy as np
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QInputDialog, QLabel,
    QLineEdit, QPushButton, QRadioButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

from ..colors import apply_to_layer, palette


def _parse_floats(text):
    return [float(t) for t in text.replace(";", ",").split(",") if t.strip()]


class NnUNetWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.job = None
        self.ws = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll_job)
        self._build_ui()
        self._open_workspace()

    # ==================================================================
    def _build_ui(self):
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        lay = QVBoxLayout(body)
        scroll.setWidget(body)
        outer.addWidget(scroll)

        from ...nnunet.training import detect_gpus, nnunet_available
        from ...nnunet.workspace import default_root

        ok = nnunet_available()
        gpus = detect_gpus() if ok else []
        info = QLabel(("nnU-Net installe" if ok else "nnU-Net NON installe (pip install xwiclassifier[nnunet])")
                      + f" — GPU : {len(gpus) if gpus else 'aucun'}")
        info.setWordWrap(True)
        lay.addWidget(info)

        box = QGroupBox("Espace de travail et jeu de donnees")
        g = QGridLayout(box)
        self.root_edit = QLineEdit(str(default_root()))
        b = QPushButton("...")
        b.clicked.connect(self._browse_root)
        g.addWidget(QLabel("Dossier :"), 0, 0)
        g.addWidget(self.root_edit, 0, 1)
        g.addWidget(b, 0, 2)
        self.root_edit.editingFinished.connect(self._open_workspace)
        self.ds_combo = QComboBox()
        self.ds_combo.currentIndexChanged.connect(self._refresh_models)
        g.addWidget(QLabel("Organe / structure :"), 1, 0)
        g.addWidget(self.ds_combo, 1, 1)
        b = QPushButton("Nouveau...")
        b.clicked.connect(self._new_dataset)
        g.addWidget(b, 1, 2)
        self.ds_info = QLabel("")
        self.ds_info.setWordWrap(True)
        g.addWidget(self.ds_info, 2, 0, 1, 3)
        lay.addWidget(box)

        box = QGroupBox("Ajouter des donnees d'entrainement")
        g = QGridLayout(box)
        self.img_combo = QComboBox()
        self.lab_combo = QComboBox()
        g.addWidget(QLabel("Image :"), 0, 0)
        g.addWidget(self.img_combo, 0, 1, 1, 2)
        g.addWidget(QLabel("Labels :"), 1, 0)
        g.addWidget(self.lab_combo, 1, 1, 1, 2)
        self.spacing_edit = QLineEdit()
        self.spacing_edit.setPlaceholderText("auto (echelle du layer)")
        g.addWidget(QLabel("Voxel (z,y,x) :"), 2, 0)
        g.addWidget(self.spacing_edit, 2, 1, 1, 2)
        self.rb_sparse = QRadioButton("Annotations partielles (traits)")
        self.rb_dense = QRadioButton("Region validee (segmentation corrigee)")
        self.rb_sparse.setChecked(True)
        g.addWidget(self.rb_sparse, 3, 0, 1, 3)
        g.addWidget(self.rb_dense, 4, 0, 1, 3)
        row = QHBoxLayout()
        row.addWidget(QLabel("Coupes z de"))
        self.z0 = QSpinBox()
        self.z1 = QSpinBox()
        for sp in (self.z0, self.z1):
            sp.setRange(0, 100000)
        row.addWidget(self.z0)
        row.addWidget(QLabel("a"))
        row.addWidget(self.z1)
        b = QPushButton("Autour de la vue")
        b.clicked.connect(self._roi_around_view)
        row.addWidget(b)
        g.addLayout(row, 5, 0, 1, 3)
        self.margin_spin = QSpinBox()
        self.margin_spin.setRange(0, 256)
        self.margin_spin.setValue(16)
        g.addWidget(QLabel("Marge de contexte :"), 6, 0)
        g.addWidget(self.margin_spin, 6, 1)
        b = QPushButton("Ajouter au jeu de donnees")
        b.clicked.connect(self._add_case)
        g.addWidget(b, 7, 0, 1, 3)
        lay.addWidget(box)
        self.viewer.layers.events.inserted.connect(lambda e: self._refresh_layers())
        self.viewer.layers.events.removed.connect(lambda e: self._refresh_layers())

        box = QGroupBox("Entrainement")
        g = QGridLayout(box)
        self.planner_combo = QComboBox()
        self.planner_combo.addItems(["default", "resenc_m", "resenc_l"])
        self.planner_combo.setToolTip("default : U-Net classique (~8 Go VRAM) ; resenc_m ~10 Go ; resenc_l ~24 Go")
        g.addWidget(QLabel("Architecture :"), 0, 0)
        g.addWidget(self.planner_combo, 0, 1)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(250)
        g.addWidget(QLabel("Epoques :"), 1, 0)
        g.addWidget(self.epochs_spin, 1, 1)
        self.folds_combo = QComboBox()
        self.folds_combo.addItems(["all (un modele)", "5 folds (validation croisee)"])
        g.addWidget(QLabel("Folds :"), 2, 0)
        g.addWidget(self.folds_combo, 2, 1)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cuda", "mps", "cpu"])
        g.addWidget(QLabel("Calcul :"), 3, 0)
        g.addWidget(self.device_combo, 3, 1)
        self.prep_btn = QPushButton("1. Preparer")
        self.prep_btn.clicked.connect(self._plan)
        self.train_btn = QPushButton("2. Entrainer")
        self.train_btn.clicked.connect(self._train)
        self.stop_btn = QPushButton("Arreter")
        self.stop_btn.clicked.connect(self._stop)
        row = QHBoxLayout()
        for w in (self.prep_btn, self.train_btn, self.stop_btn):
            row.addWidget(w)
        g.addLayout(row, 4, 0, 1, 2)
        self.cont_check = QCheckBox("Reprendre un entrainement interrompu")
        g.addWidget(self.cont_check, 5, 0, 1, 2)
        lay.addWidget(box)

        box = QGroupBox("Prediction")
        g = QGridLayout(box)
        self.model_combo = QComboBox()
        g.addWidget(QLabel("Modele :"), 0, 0)
        g.addWidget(self.model_combo, 0, 1)
        self.tta_check = QCheckBox("Augmentation par miroirs (plus precis, 8x plus lent en 3D)")
        self.tta_check.setChecked(True)
        g.addWidget(self.tta_check, 1, 0, 1, 2)
        b1 = QPushButton("Predire la vue")
        b1.clicked.connect(lambda: self._predict(view=True))
        b2 = QPushButton("Predire tout")
        b2.clicked.connect(lambda: self._predict(view=False))
        row = QHBoxLayout()
        row.addWidget(b1)
        row.addWidget(b2)
        g.addLayout(row, 2, 0, 1, 2)
        lay.addWidget(box)

        self.status = QLabel("Pret.")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        lay.addStretch()
        self._refresh_layers()

    # ==================================================================
    def _browse_root(self):
        d = QFileDialog.getExistingDirectory(self, "Espace de travail nnU-Net", self.root_edit.text())
        if d:
            self.root_edit.setText(d)
            self._open_workspace()

    def _open_workspace(self):
        from ...nnunet.workspace import NnUNetWorkspace

        try:
            self.ws = NnUNetWorkspace(self.root_edit.text().strip() or None)
        except Exception as exc:
            self.status.setText(f"Espace de travail invalide : {exc}")
            return
        self._refresh_datasets()

    def _refresh_datasets(self, select=None):
        cur = select or self.ds_combo.currentText()
        self.ds_combo.blockSignals(True)
        self.ds_combo.clear()
        for d in self.ws.datasets() if self.ws else []:
            self.ds_combo.addItem(d.name)
        i = self.ds_combo.findText(cur) if cur else -1
        if i >= 0:
            self.ds_combo.setCurrentIndex(i)
        self.ds_combo.blockSignals(False)
        self._refresh_models()

    def _dataset(self):
        name = self.ds_combo.currentText()
        return self.ws.get_dataset(name) if (self.ws and name) else None

    def _refresh_models(self, *_):
        self.model_combo.clear()
        ds = self._dataset()
        if ds is None:
            self.ds_info.setText("Aucun jeu de donnees : cliquez Nouveau...")
            return
        self.ds_info.setText(f"{ds.n_cases} cas, classes : " +
                             ", ".join(f"{k}={v}" for k, v in sorted(ds.class_names.items())))
        for m in self.ws.trained_models(ds):
            self.model_combo.addItem(m.name, str(m))

    def _layers(self, kind):
        import napari
        cls = napari.layers.Image if kind == "image" else napari.layers.Labels
        return [lyr for lyr in self.viewer.layers if isinstance(lyr, cls)]

    def _refresh_layers(self):
        for combo, kind in ((self.img_combo, "image"), (self.lab_combo, "labels")):
            cur = combo.currentText()
            combo.clear()
            for lyr in self._layers(kind):
                combo.addItem(lyr.name)
            i = combo.findText(cur)
            if i >= 0:
                combo.setCurrentIndex(i)

    def _layer(self, name):
        return self.viewer.layers[name] if name and name in self.viewer.layers else None

    def _spacing(self, layer, ndim):
        vals = _parse_floats(self.spacing_edit.text()) if self.spacing_edit.text().strip() else None
        if vals and len(vals) >= ndim:
            return tuple(vals[-ndim:])
        return tuple(float(s) for s in layer.scale[-ndim:]) if layer is not None else None

    # ------------------------------------------------------------------
    def _new_dataset(self, *_, name=None, classes=None):
        lab = self._layer(self.lab_combo.currentText())
        meta = dict(lab.metadata.get("xwi_class_names", {})) if lab is not None else {}
        if name is None:
            name, ok = QInputDialog.getText(self, "Nouveau jeu de donnees", "Organe / structure :")
            if not ok or not name.strip():
                return None
        if classes is None:
            default = ",".join(f"{k}={v}" for k, v in sorted(meta.items())) or "1=fond,2=structure"
            txt, ok = QInputDialog.getText(self, "Classes", "Classes (1 = fond) :", text=default)
            if not ok:
                return None
            classes = {int(k): v.strip() for k, v in (p.split("=", 1) for p in txt.split(","))}
        colors = lab.metadata.get("xwi_class_colors") if lab is not None else None
        colors = {int(k): tuple(int(255 * x) for x in c[:3]) for k, c in (colors or {}).items()}
        try:
            ds = self.ws.create_dataset(name, classes, colors)
        except Exception as exc:
            self.status.setText(f"Creation impossible : {exc}")
            return None
        self._refresh_datasets(select=ds.name)
        self.status.setText(f"Jeu de donnees cree : {ds.name}")
        return ds

    def _roi_around_view(self):
        step = self.viewer.dims.current_step
        z = int(step[-3]) if len(step) >= 3 else 0
        self.z0.setValue(max(0, z - 16))
        self.z1.setValue(z + 16)

    def _add_case(self):
        ds = self._dataset()
        img_l, lab_l = self._layer(self.img_combo.currentText()), self._layer(self.lab_combo.currentText())
        if ds is None or img_l is None or lab_l is None:
            self.status.setText("Choisissez un jeu de donnees, une image et un layer de labels.")
            return
        img = np.asarray(img_l.data)
        lab = np.asarray(lab_l.data)
        ch = -1 if getattr(img_l, "rgb", False) else None
        ndim = lab.ndim
        mode = "dense" if self.rb_dense.isChecked() else "sparse"
        roi = None
        if mode == "dense":
            if ndim == 3:
                z0, z1 = sorted((self.z0.value(), self.z1.value()))
                roi = (slice(z0, min(z1 + 1, lab.shape[0])), slice(None), slice(None))
            else:
                roi = (slice(None), slice(None))
        try:
            case = ds.add_case(img, lab, spacing=self._spacing(img_l, ndim), channel_axis=ch, roi=roi,
                               margin=self.margin_spin.value(), mode=mode, source=f"napari:{img_l.name}")
        except Exception as exc:
            self.status.setText(f"Ajout impossible : {exc}")
            return
        self._refresh_models()
        self.status.setText(f"{case} ajoute ({'region validee' if mode == 'dense' else 'annotations partielles'}).")

    # ------------------------------------------------------------------
    def _plan(self):
        ds = self._dataset()
        if ds is None or ds.n_cases == 0:
            self.status.setText("Ajoutez d'abord des donnees au jeu.")
            return
        from ...nnunet.training import plan_and_preprocess

        self.job = plan_and_preprocess(ds, planner=self.planner_combo.currentText(), wait=False)
        self.status.setText("Preparation en cours (analyse + pretraitement)...")
        self._timer.start(3000)

    def _train(self):
        ds = self._dataset()
        if ds is None:
            return
        from ...nnunet.training import train

        folds = ("all",) if self.folds_combo.currentIndex() == 0 else (0, 1, 2, 3, 4)
        try:
            self.job = train(ds, folds=folds, epochs=self.epochs_spin.value(),
                             device=self.device_combo.currentText(), planner=self.planner_combo.currentText(),
                             continue_training=self.cont_check.isChecked())
        except Exception as exc:
            self.status.setText(f"Lancement impossible : {exc}")
            return
        self.status.setText(f"Entrainement lance ({len(self.job.processes)} processus). Journal : {self.job.logs[0]}")
        self._timer.start(10000)

    def _stop(self):
        if self.job is not None:
            self.job.terminate()
            self.status.setText("Arret demande (reprise possible avec 'Reprendre').")

    def _poll_job(self):
        if self.job is None:
            self._timer.stop()
            return
        running = self.job.running()
        if self.job.kind == "plan":
            if not running:
                self._timer.stop()
                code = self.job.wait()
                self.status.setText("Preparation terminee." if code == 0 else
                                    f"Preparation en echec, voir {self.job.logs[0]}")
            return
        st = self.job.status()
        parts = []
        for f, i in st["folds"].items():
            dice = f"{i['best_ema_dice']:.3f}" if i["best_ema_dice"] is not None else "-"
            eta = ""
            if i["epoch_time_s"] and i["epoch"] is not None and self.job.epochs:
                eta = f", reste ~{(self.job.epochs - i['epoch'] - 1) * i['epoch_time_s'] / 60:.0f} min"
            parts.append(f"fold {f} : epoque {i['epoch']}/{self.job.epochs}, Dice {dice}{eta}")
        self.status.setText(("En cours — " if running else "Termine — ") + " | ".join(parts or ["demarrage..."]))
        if not running:
            self._timer.stop()
            self._refresh_models()

    # ------------------------------------------------------------------
    def _predict(self, view: bool):
        img_l = self._layer(self.img_combo.currentText())
        path = self.model_combo.currentData()
        if img_l is None or not path:
            self.status.setText("Choisissez une image et un modele entraine.")
            return
        from napari.qt.threading import thread_worker

        from ...nnunet.inference import NnUNetSegmenter

        img = np.asarray(img_l.data)
        ch = -1 if getattr(img_l, "rgb", False) else None
        shape = img.shape if ch is None else img.shape[:-1]
        region = None
        if view and len(shape) == 3 and self.viewer.dims.ndisplay == 2:
            z = int(np.clip(self.viewer.dims.current_step[-3], 0, shape[0] - 1))
            region = (slice(z, z + 1), slice(None), slice(None))
        spacing = self._spacing(img_l, len(shape))

        def job():
            m = NnUNetSegmenter(path, device=self.device_combo.currentText(), use_mirroring=self.tta_check.isChecked())
            return m, m.predict(img, channel_axis=ch, region=region, spacing=spacing)

        worker = thread_worker(job)()
        worker.returned.connect(lambda res: self._show_prediction(res, shape, region, img_l))
        worker.errored.connect(lambda e: self.status.setText(f"Prediction en echec : {e}"))
        self.status.setText("Prediction nnU-Net en cours...")
        worker.start()

    def _show_prediction(self, res, shape, region, img_l):
        model, seg = res
        name = "nnU-Net"
        if name not in self.viewer.layers or self.viewer.layers[name].data.shape != shape:
            layer = self.viewer.add_labels(np.zeros(shape, np.uint8), name=name, opacity=0.6,
                                           scale=img_l.scale[-len(shape):])
        else:
            layer = self.viewer.layers[name]
        tgt = region if region is not None else tuple(slice(None) for _ in shape)
        layer.data[tgt] = seg
        layer.refresh()
        apply_to_layer(layer, palette(model.classes_, {k: v for k, v in model.class_colors.items()}))
        layer.metadata["xwi_class_names"] = dict(model.class_names)
        self.status.setText("Prediction nnU-Net affichee" + (" (coupe courante)." if region is not None else "."))
