"""Widget napari de classification interactive de pixels (style ilastik, en mieux).

Flux : choisir l'image -> definir les classes -> peindre -> Entrainer ->
Predire la vue (rapide) ou tout le volume. Les calculs tournent dans un thread
pour ne pas figer l'interface ; le mode live re-entraine apres chaque coup de
pinceau (anti-rebond) en ne recalculant que les blocs nouvellement annotes.
"""
from __future__ import annotations

import json

import numpy as np
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QMessageBox, QProgressBar, QPushButton,
    QScrollArea, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from ...backend import backend_name
from ...classifier import PixelClassifier, TrainingSampler, available_classifiers, uncertainty
from ...classifier.model import labels_from_proba
from ...features import DEFAULT_FAMILIES, DEFAULT_RADII, DEFAULT_SCALES, FAMILIES, FeatureSpec
from ..colors import BACKGROUND_LABEL, apply_to_layer, default_color, palette, to_rgba01

MAX_CLASSES = 9  # en plus du background


def _parse_floats(text: str) -> list:
    out = []
    for tok in text.replace(";", ",").split(","):
        tok = tok.strip()
        if tok:
            out.append(float(tok))
    return out


class PixelClassifierWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.model: PixelClassifier | None = None
        self._labels_layer = None
        self._result_layer = None
        self._unc_layer = None
        self._sampler: TrainingSampler | None = None
        self._sampler_key = None
        self._busy = False
        self._pending_live = False
        self._progress = (0, 0, "")
        self.class_names: dict[int, str] = {}
        self.class_colors: dict[int, tuple] = {}
        self._class_buttons: dict = {}
        self._feature_checks: dict = {}

        self._live_timer = QTimer(self)
        self._live_timer.setSingleShot(True)
        self._live_timer.timeout.connect(self._live_update)
        self._progress_timer = QTimer(self)
        self._progress_timer.timeout.connect(self._refresh_progress)
        self._build_ui()

    # ==================================================================
    # Construction de l'interface
    # ==================================================================
    def _build_ui(self):
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        lay = QVBoxLayout(body)
        scroll.setWidget(body)
        outer.addWidget(scroll)

        # ---- Donnees ---------------------------------------------------
        box = QGroupBox("Donnees")
        g = QGridLayout(box)
        g.addWidget(QLabel(f"Calcul : {backend_name()}"), 0, 0, 1, 2)
        self.image_combo = QComboBox()
        g.addWidget(QLabel("Image :"), 1, 0)
        g.addWidget(self.image_combo, 1, 1)
        self.spacing_edit = QLineEdit()
        self.spacing_edit.setPlaceholderText("auto (echelle du layer) ex. 2,1,1")
        self.spacing_edit.setToolTip("Taille de voxel (z,y,x) : les filtres s'adaptent a l'anisotropie.")
        g.addWidget(QLabel("Voxel (z,y,x) :"), 2, 0)
        g.addWidget(self.spacing_edit, 2, 1)
        self.in2d_check = QCheckBox("Filtres 2D par coupe (volumes 3D)")
        g.addWidget(self.in2d_check, 3, 0, 1, 2)
        lay.addWidget(box)
        self.viewer.layers.events.inserted.connect(lambda e: self._refresh_image_combo())
        self.viewer.layers.events.removed.connect(lambda e: self._refresh_image_combo())
        self._refresh_image_combo()

        # ---- Classes ---------------------------------------------------
        box = QGroupBox("Classes et annotation")
        v = QVBoxLayout(box)
        row = QHBoxLayout()
        row.addWidget(QLabel("Classes (hors background) :"))
        self.n_classes_spin = QSpinBox()
        self.n_classes_spin.setRange(1, MAX_CLASSES)
        self.n_classes_spin.setValue(2)
        self.n_classes_spin.valueChanged.connect(self._rebuild_class_buttons)
        row.addWidget(self.n_classes_spin)
        v.addLayout(row)
        self.class_grid = QGridLayout()
        v.addLayout(self.class_grid)
        self.class_group = QButtonGroup(self)
        self.class_group.setExclusive(True)
        row = QHBoxLayout()
        row.addWidget(QLabel("Pinceau :"))
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setRange(1, 50)
        self.brush_spin = QSpinBox()
        self.brush_spin.setRange(1, 200)
        self.brush_slider.setValue(5)
        self.brush_spin.setValue(5)
        self.brush_slider.valueChanged.connect(self.brush_spin.setValue)
        self.brush_spin.valueChanged.connect(self._on_brush_size)
        row.addWidget(self.brush_slider)
        row.addWidget(self.brush_spin)
        v.addLayout(row)
        btn = QPushButton("Creer / selectionner le layer d'annotation")
        btn.clicked.connect(self._ensure_labels_layer)
        v.addWidget(btn)
        lay.addWidget(box)
        self._rebuild_class_buttons()

        # ---- Features --------------------------------------------------
        box = QGroupBox("Features")
        v = QVBoxLayout(box)
        g = QGridLayout()
        self.scales_edit = QLineEdit(", ".join(f"{s:g}" for s in DEFAULT_SCALES))
        self.radii_edit = QLineEdit(", ".join(str(r) for r in DEFAULT_RADII))
        g.addWidget(QLabel("Sigmas :"), 0, 0)
        g.addWidget(self.scales_edit, 0, 1)
        g.addWidget(QLabel("Rayons morpho :"), 1, 0)
        g.addWidget(self.radii_edit, 1, 1)
        apply_btn = QPushButton("Appliquer")
        apply_btn.clicked.connect(lambda: self._rebuild_feature_grid(keep=True))
        g.addWidget(apply_btn, 0, 2, 2, 1)
        v.addLayout(g)
        row = QHBoxLayout()
        for label, mode in (("Defaut", "default"), ("Tout", "all"), ("Aucun", "none")):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, m=mode: self._preset(m))
            row.addWidget(b)
        v.addLayout(row)
        self.grid_holder = QWidget()
        self.feature_grid = QGridLayout(self.grid_holder)
        self.feature_grid.setHorizontalSpacing(4)
        self.feature_grid.setVerticalSpacing(1)
        fscroll = QScrollArea()
        fscroll.setWidgetResizable(True)
        fscroll.setWidget(self.grid_holder)
        fscroll.setMinimumHeight(200)
        fscroll.setMaximumHeight(320)
        v.addWidget(fscroll)
        row = QHBoxLayout()
        row.addWidget(QLabel("Colonne :"))
        self.col_combo = QComboBox()
        row.addWidget(self.col_combo)
        for label, val in (("Cocher", True), ("Decocher", False)):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, x=val: self._toggle_column(x))
            row.addWidget(b)
        v.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Normalisation :"))
        self.norm_combo = QComboBox()
        self.norm_combo.addItems(["none", "percentile", "minmax", "zscore"])
        self.norm_combo.setToolTip("none : intensites brutes (recommande si l'acquisition est stable) ; "
                                   "percentile : robuste aux variations d'intensite entre images.")
        row.addWidget(self.norm_combo)
        v.addLayout(row)
        self.feature_info = QLabel("")
        self.feature_info.setWordWrap(True)
        v.addWidget(self.feature_info)
        lay.addWidget(box)
        self._rebuild_feature_grid()

        # ---- Classifieur et actions -----------------------------------
        box = QGroupBox("Apprentissage et prediction")
        g = QGridLayout(box)
        self.kind_combo = QComboBox()
        avail = available_classifiers()
        self.kind_combo.addItems(avail)
        missing = [k for k in ("lightgbm", "xgboost") if k not in avail]
        if missing:
            self.kind_combo.setToolTip("Non installes : " + ", ".join(missing))
        g.addWidget(QLabel("Classifieur :"), 0, 0)
        g.addWidget(self.kind_combo, 0, 1)
        self.trees_spin = QSpinBox()
        self.trees_spin.setRange(10, 2000)
        self.trees_spin.setValue(100)
        g.addWidget(QLabel("Arbres :"), 1, 0)
        g.addWidget(self.trees_spin, 1, 1)
        self.smooth_spin = QDoubleSpinBox()
        self.smooth_spin.setRange(0.0, 10.0)
        self.smooth_spin.setSingleStep(0.5)
        self.smooth_spin.setToolTip("Lissage des probabilites avant decision (reduit le bruit).")
        g.addWidget(QLabel("Lissage proba :"), 2, 0)
        g.addWidget(self.smooth_spin, 2, 1)
        self.train_btn = QPushButton("Entrainer")
        self.train_btn.clicked.connect(lambda: self._start_job(predict="view"))
        g.addWidget(self.train_btn, 3, 0, 1, 2)
        self.view_btn = QPushButton("Predire la vue")
        self.view_btn.setToolTip("Coupe affichee pour un volume 3D (rapide)")
        self.view_btn.clicked.connect(lambda: self._start_job(train=False, predict="view"))
        self.full_btn = QPushButton("Predire tout")
        self.full_btn.clicked.connect(lambda: self._start_job(train=False, predict="full"))
        g.addWidget(self.view_btn, 4, 0)
        g.addWidget(self.full_btn, 4, 1)
        self.live_check = QCheckBox("Mode live (re-entraine apres chaque trait)")
        self.live_check.stateChanged.connect(self._toggle_live)
        g.addWidget(self.live_check, 5, 0, 1, 2)
        self.unc_check = QCheckBox("Afficher l'incertitude (ou annoter ensuite)")
        g.addWidget(self.unc_check, 6, 0, 1, 2)
        lay.addWidget(box)

        # ---- Modeles ---------------------------------------------------
        box = QGroupBox("Modeles")
        g = QGridLayout(box)
        b = QPushButton("Sauver...")
        b.clicked.connect(self._save_model)
        g.addWidget(b, 0, 0)
        b = QPushButton("Charger (.joblib / .ilp)...")
        b.clicked.connect(self._load_model)
        g.addWidget(b, 0, 1)
        b = QPushButton("Dossier de modele IPSDK Explorer...")
        b.setToolTip("Dossier contenant Settings.mho et Model (AppData/Roaming/ReactivIP/Explorer/"
                     "Machine_Learning/Pixel_Classification/<modele>)")
        b.clicked.connect(self._load_ipsdk_folder)
        g.addWidget(b, 2, 0, 1, 2)
        self.ipsdk_btn = QPushButton("Distiller un modele IPSDK (.bin)...")
        self.ipsdk_btn.clicked.connect(self._distill_ipsdk)
        from ...io.ipsdk import ipsdk_available
        if not ipsdk_available():
            self.ipsdk_btn.setEnabled(False)
            self.ipsdk_btn.setToolTip("Necessite PyIPSDK (licence Reactiv'IP) sur cette machine.")
        g.addWidget(self.ipsdk_btn, 1, 0, 1, 2)
        lay.addWidget(box)

        # ---- Etat ------------------------------------------------------
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        lay.addWidget(self.progress)
        self.status = QLabel("Pret.")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        row = QHBoxLayout()
        row.addWidget(QLabel("Contributions :"))
        self.imp_group = QComboBox()
        self.imp_group.addItems(["par famille", "par echelle", "par canal"])
        self.imp_group.currentIndexChanged.connect(self._show_importances)
        row.addWidget(self.imp_group)
        lay.addLayout(row)
        self.importance_list = QListWidget()
        self.importance_list.setMaximumHeight(180)
        lay.addWidget(self.importance_list)
        lay.addStretch()

    # ==================================================================
    # Image et geometrie
    # ==================================================================
    def _image_layers(self):
        import napari
        return [lyr for lyr in self.viewer.layers if isinstance(lyr, napari.layers.Image)]

    def _refresh_image_combo(self):
        current = self.image_combo.currentText()
        self.image_combo.blockSignals(True)
        self.image_combo.clear()
        for lyr in self._image_layers():
            self.image_combo.addItem(lyr.name)
        idx = self.image_combo.findText(current)
        if idx >= 0:
            self.image_combo.setCurrentIndex(idx)
        self.image_combo.blockSignals(False)

    def _image_layer(self):
        name = self.image_combo.currentText()
        return self.viewer.layers[name] if name and name in self.viewer.layers else None

    def _image(self):
        lyr = self._image_layer()
        if lyr is None:
            return None, None
        data = np.asarray(lyr.data)
        return data, (-1 if getattr(lyr, "rgb", False) else None)

    def _spatial_shape(self, data, ch):
        return data.shape if ch is None else data.shape[:-1]

    def _spacing(self, ndim):
        txt = self.spacing_edit.text().strip()
        if txt:
            vals = _parse_floats(txt)
            return tuple(vals[-ndim:]) if len(vals) >= ndim else None
        lyr = self._image_layer()
        if lyr is not None:
            sc = tuple(float(s) for s in lyr.scale[-ndim:])
            if any(abs(s - sc[-1]) > 1e-6 for s in sc):
                return sc
        return None

    # ==================================================================
    # Classes
    # ==================================================================
    def _default_name(self, value):
        return "Background" if value == BACKGROUND_LABEL else f"Classe {value - BACKGROUND_LABEL}"

    def _class_values(self):
        return [BACKGROUND_LABEL + k for k in range(self.n_classes_spin.value() + 1)]

    def _rebuild_class_buttons(self):
        for btn in list(self.class_group.buttons()):
            self.class_group.removeButton(btn)
        while self.class_grid.count():
            w = self.class_grid.takeAt(0).widget()
            if w is not None:
                w.setParent(None)
        self._class_buttons = {}
        values = self._class_values()
        for row, value in enumerate(values):
            name = self.class_names.get(value, self._default_name(value))
            self.class_names[value] = name
            r, g, b, _ = self.class_colors.get(value, default_color(value))
            btn = QPushButton(name)
            btn.setCheckable(True)
            light = (0.299 * r + 0.587 * g + 0.114 * b) > 0.55
            btn.setStyleSheet(
                f"QPushButton {{ background-color: rgb({int(r * 255)},{int(g * 255)},{int(b * 255)});"
                f" color: {'black' if light else 'white'}; padding: 4px; }}"
                "QPushButton:checked { border: 3px solid black; font-weight: bold; }")
            self.class_group.addButton(btn, value)
            btn.clicked.connect(lambda _=False, v=value: self._select_class(v))
            edit = QLineEdit(name)
            edit.textChanged.connect(lambda txt, v=value: self._rename_class(v, txt))
            self._class_buttons[value] = btn
            self.class_grid.addWidget(btn, row, 0)
            self.class_grid.addWidget(edit, row, 1)
        for v in list(self.class_names):
            if v not in values:
                self.class_names.pop(v)
        first = self._class_buttons.get(BACKGROUND_LABEL)
        if first is not None:
            first.setChecked(True)
        self._apply_palettes()

    def _rename_class(self, value, txt):
        self.class_names[value] = txt
        if value in self._class_buttons:
            self._class_buttons[value].setText(txt)
        if self.model is not None:
            self.model.class_names[value] = txt

    def _select_class(self, value):
        try:
            btn = self._class_buttons.get(value)
            if btn is not None and not btn.isChecked():
                btn.setChecked(True)
            if not self._has_layer(self._labels_layer):
                if self._image()[0] is None:
                    self.status.setText("Selectionnez d'abord une image.")
                    return
                self._ensure_labels_layer()
            self.viewer.layers.selection.active = self._labels_layer
            self._labels_layer.selected_label = int(value)
            self._labels_layer.brush_size = self.brush_spin.value()
            self._labels_layer.mode = "paint"
            self.status.setText(f"Pinceau : « {self.class_names.get(value, value)} » (valeur {value}).")
        except Exception as exc:  # pragma: no cover - depend de la version de napari
            self.status.setText(f"Erreur de selection de classe : {exc}")

    def _on_brush_size(self, size):
        if self.brush_slider.value() != size and size <= self.brush_slider.maximum():
            self.brush_slider.setValue(size)
        if self._has_layer(self._labels_layer):
            self._labels_layer.brush_size = size

    def _has_layer(self, layer):
        return layer is not None and layer in self.viewer.layers

    def _apply_palettes(self):
        cols = palette(self._class_values(), self.class_colors)
        for lyr in (self._labels_layer, self._result_layer):
            if self._has_layer(lyr):
                apply_to_layer(lyr, cols)

    def _ensure_labels_layer(self):
        data, ch = self._image()
        if data is None:
            self.status.setText("Aucune image selectionnee.")
            return None
        shape = self._spatial_shape(data, ch)
        if not self._has_layer(self._labels_layer) or self._labels_layer.data.shape != shape:
            self._labels_layer = self.viewer.add_labels(
                np.zeros(shape, np.uint8), name="annotations", scale=self._image_layer().scale[-len(shape):])
            if hasattr(self._labels_layer.events, "paint"):
                self._labels_layer.events.paint.connect(self._on_paint)
            self._sampler = None
        self._apply_palettes()
        self.viewer.layers.selection.active = self._labels_layer
        self._labels_layer.selected_label = BACKGROUND_LABEL
        self._labels_layer.brush_size = self.brush_spin.value()
        self._labels_layer.mode = "paint"
        if BACKGROUND_LABEL in self._class_buttons:
            self._class_buttons[BACKGROUND_LABEL].setChecked(True)
        self.status.setText("Choisissez une classe puis peignez.")
        return self._labels_layer

    # ==================================================================
    # Features
    # ==================================================================
    def _rebuild_feature_grid(self, keep: bool = False):
        old = {k: cb.isChecked() for k, cb in self._feature_checks.items()} if keep else None
        while self.feature_grid.count():
            w = self.feature_grid.takeAt(0).widget()
            if w is not None:
                w.setParent(None)
        try:
            scales = sorted(set(_parse_floats(self.scales_edit.text()))) or list(DEFAULT_SCALES)
            radii = sorted({int(round(r)) for r in _parse_floats(self.radii_edit.text()) if r >= 1}) \
                or list(DEFAULT_RADII)
        except ValueError:
            self.status.setText("Sigmas / rayons invalides.")
            return
        self._scales, self._radii = scales, radii
        self._feature_checks = {}
        cols = max(len(scales), len(radii))
        r = 0
        self.feature_grid.addWidget(QLabel("<b>sigma</b>"), r, 0)
        for c, s in enumerate(scales):
            self.feature_grid.addWidget(QLabel(f"{s:g}"), r, c + 1, Qt.AlignCenter)
        r += 1
        for fam, info in FAMILIES.items():
            if info.kind not in ("raw", "smooth", "custom"):
                continue
            lbl = QLabel(fam)
            lbl.setToolTip(info.description)
            self.feature_grid.addWidget(lbl, r, 0)
            params = [0.0] if info.kind == "raw" else [s for s in scales if s >= info.min_scale]
            for p in params:
                cb = QCheckBox()
                key = (fam, float(p))
                default = fam in DEFAULT_FAMILIES
                cb.setChecked(old.get(key, default) if old is not None else default)
                cb.stateChanged.connect(self._update_feature_info)
                col = 1 if info.kind == "raw" else scales.index(p) + 1
                self.feature_grid.addWidget(cb, r, col, Qt.AlignCenter)
                self._feature_checks[key] = cb
            r += 1
        self.feature_grid.addWidget(QLabel("<b>rayon</b>"), r, 0)
        for c, rad in enumerate(radii):
            self.feature_grid.addWidget(QLabel(str(rad)), r, c + 1, Qt.AlignCenter)
        r += 1
        for fam, info in FAMILIES.items():
            if info.kind != "morpho":
                continue
            lbl = QLabel(fam)
            lbl.setToolTip(info.description)
            self.feature_grid.addWidget(lbl, r, 0)
            for c, rad in enumerate(radii):
                cb = QCheckBox()
                key = (fam, float(rad))
                cb.setChecked(old.get(key, False) if old is not None else False)
                cb.stateChanged.connect(self._update_feature_info)
                self.feature_grid.addWidget(cb, r, c + 1, Qt.AlignCenter)
                self._feature_checks[key] = cb
            r += 1
        self.col_combo.clear()
        self.col_combo.addItems([f"sigma {s:g}" for s in scales] + [f"rayon {x}" for x in radii])
        _ = cols
        self._update_feature_info()

    def _preset(self, mode):
        for (fam, _p), cb in self._feature_checks.items():
            if mode == "all":
                cb.setChecked(True)
            elif mode == "none":
                cb.setChecked(False)
            else:
                cb.setChecked(fam in DEFAULT_FAMILIES)

    def _toggle_column(self, checked):
        txt = self.col_combo.currentText()
        if not txt:
            return
        kind, val = txt.split(" ")
        val = float(val)
        for (fam, p), cb in self._feature_checks.items():
            is_morpho = FAMILIES[fam].kind == "morpho"
            if p == val and is_morpho == (kind == "rayon"):
                cb.setChecked(checked)

    def _build_spec(self) -> FeatureSpec:
        pairs = tuple(k for k, cb in self._feature_checks.items() if cb.isChecked()) or (("gaussian", 1.0),)
        data, ch = self._image()
        ndim = len(self._spatial_shape(data, ch)) if data is not None else 2
        return FeatureSpec(pairs=pairs, spacing=self._spacing(ndim),
                           compute_in_2d=self.in2d_check.isChecked(),
                           normalize=self.norm_combo.currentText())

    def _update_feature_info(self, *_):
        n = sum(cb.isChecked() for cb in self._feature_checks.values())
        try:
            data, ch = self._image()
            ndim = len(self._spatial_shape(data, ch)) if data is not None else 2
            from ...features import FeatureEngine
            nf = FeatureEngine(self._build_spec()).n_features(ndim)
            self.feature_info.setText(f"{n} filtres coches -> {nf} canaux de features ({ndim}D).")
        except Exception:
            self.feature_info.setText(f"{n} filtres coches.")

    # ==================================================================
    # Entrainement / prediction (thread)
    # ==================================================================
    def _set_busy(self, busy: bool):
        self._busy = busy
        for b in (self.train_btn, self.view_btn, self.full_btn):
            b.setEnabled(not busy)
        self.progress.setVisible(busy)
        if busy:
            self._progress = (0, 0, "")
            self._progress_timer.start(200)
        else:
            self._progress_timer.stop()

    def _refresh_progress(self):
        i, n, label = self._progress
        if n:
            self.progress.setMaximum(n)
            self.progress.setValue(i)
            self.progress.setFormat(f"{label} %v/%m")

    def _cb(self, label):
        def cb(i, n):
            self._progress = (i, n, label)
        return cb

    def _view_region(self, shape):
        if len(shape) == 3 and self.viewer.dims.ndisplay == 2:
            step = self.viewer.dims.current_step
            z = int(np.clip(step[-3] if len(step) >= 3 else 0, 0, shape[0] - 1))
            return (slice(z, z + 1), slice(None), slice(None)), z
        return None, None

    def _start_job(self, train: bool = True, predict: str | None = "view"):
        if self._busy:
            self._pending_live = True
            return
        data, ch = self._image()
        if data is None:
            self.status.setText("Aucune image selectionnee.")
            return
        labels = None
        if train:
            if not self._has_layer(self._labels_layer) or not np.any(self._labels_layer.data):
                self.status.setText("Peignez d'abord quelques pixels de chaque classe.")
                return
            labels = np.asarray(self._labels_layer.data)
            spec = self._build_spec()
            params = {"n_estimators": self.trees_spin.value()} \
                if self.kind_combo.currentText() in ("rf", "extratrees", "lightgbm", "xgboost") else {}
            key = (id(self._image_layer()), json.dumps(spec.to_dict(), sort_keys=True))
            if self._sampler is None or self._sampler_key != key:
                from ...features import FeatureEngine
                self._sampler = TrainingSampler(FeatureEngine(spec), cache_bytes=2e9)
                self._sampler_key = key
            new_model = PixelClassifier(spec, kind=self.kind_combo.currentText(), params=params,
                                        class_names=dict(self.class_names),
                                        class_colors=dict(self.class_colors))
        elif self.model is None:
            self.status.setText("Entrainez ou chargez d'abord un modele.")
            return
        shape = self._spatial_shape(data, ch)
        region, z = self._view_region(shape) if predict == "view" else (None, None)
        smooth = self.smooth_spin.value()
        want_unc = self.unc_check.isChecked()
        sampler = self._sampler
        image_key = id(self._image_layer())

        def job():
            model = self.model
            if train:
                model = new_model
                model.extractor = sampler.extractor
                model.fit(data, labels, channel_axis=ch, sampler=sampler, image_keys=image_key,
                          progress=self._cb("features"))
            proba = None
            if predict:
                proba = model.predict_proba(data, channel_axis=ch, region=region, smooth=smooth,
                                            progress=self._cb("prediction"))
            return model, proba

        from napari.qt.threading import thread_worker

        worker = thread_worker(job)()
        worker.returned.connect(lambda res: self._job_done(res, shape, region, want_unc, train))
        worker.errored.connect(self._job_failed)
        self._set_busy(True)
        self.status.setText("Entrainement..." if train else "Prediction...")
        worker.start()

    def _job_done(self, res, shape, region, want_unc, trained):
        model, proba = res
        self.model = model
        self._set_busy(False)
        if proba is not None:
            seg = labels_from_proba(proba, model.classes_)
            if not self._has_layer(self._result_layer) or self._result_layer.data.shape != shape:
                self._result_layer = self.viewer.add_labels(
                    np.zeros(shape, np.uint8), name="segmentation", opacity=0.6,
                    scale=self._image_layer().scale[-len(shape):])
            tgt = region if region is not None else tuple(slice(None) for _ in shape)
            self._result_layer.data[tgt] = seg
            self._result_layer.refresh()
            if want_unc:
                unc = uncertainty(proba)
                if not self._has_layer(self._unc_layer) or self._unc_layer.data.shape != shape:
                    self._unc_layer = self.viewer.add_image(
                        np.zeros(shape, np.float32), name="incertitude", colormap="magma",
                        blending="additive", contrast_limits=(0, 1),
                        scale=self._image_layer().scale[-len(shape):])
                self._unc_layer.data[tgt] = unc
                self._unc_layer.refresh()
            self._apply_palettes()
        info = model.training_info if trained else {}
        msg = "Entraine" if trained else "Predit"
        if info:
            msg += f" sur {info.get('n_used')} pixels ({info.get('per_class')})"
        if region is not None:
            msg += " — coupe affichee uniquement"
        self.status.setText(msg + ".")
        self._show_importances()
        if self._pending_live:
            self._pending_live = False
            self._live_timer.start(100)

    def _job_failed(self, exc):
        self._set_busy(False)
        self.status.setText(f"Echec : {exc}")

    def _toggle_live(self, state):
        if state and not self._has_layer(self._labels_layer):
            self._ensure_labels_layer()

    def _on_paint(self, event=None):
        if self.live_check.isChecked():
            self._live_timer.start(400)

    def _live_update(self):
        if self._busy:
            self._pending_live = True
            return
        self._start_job(train=True, predict="view")

    def _show_importances(self, *_):
        self.importance_list.clear()
        if self.model is None:
            return
        group = {0: "family", 1: "scale", 2: None}[self.imp_group.currentIndex()]
        imp = self.model.feature_importances(group=group)
        if not imp:
            self.importance_list.addItem("(importances indisponibles)")
            return
        for name, val in list(imp.items())[:60]:
            self.importance_list.addItem(f"{100 * val:5.1f} %  {'█' * int(round(50 * val))}  {name}")

    # ==================================================================
    # Modeles
    # ==================================================================
    def _save_model(self):
        if self.model is None or not self.model.is_fitted:
            self.status.setText("Rien a sauver.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Sauver le modele", "modele.joblib", "Modele (*.joblib)")
        if path:
            self.model.class_names.update(self.class_names)
            self.model.save(path)
            self.status.setText(f"Modele sauve : {path}")

    def _load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Charger un modele", "",
                                              "Modeles (*.joblib *.ilp Settings.mho);;Tous (*)")
        if not path:
            return
        try:
            if path.lower().endswith(".mho"):
                self._load_ipsdk(path)
            elif path.lower().endswith(".ilp"):
                self._load_ilastik(path)
            else:
                from ...io import load_model
                self._adopt_model(load_model(path))
                self.status.setText(f"Modele charge : {path}")
        except Exception as exc:
            self.status.setText(f"Chargement impossible : {exc}")

    def _load_ilastik(self, path):
        box = QMessageBox(self)
        box.setWindowTitle("Projet ilastik")
        box.setText("Que faire de ce projet ilastik ?")
        b_forest = box.addButton("Utiliser son classifieur", QMessageBox.AcceptRole)
        b_ann = box.addButton("Importer image + annotations", QMessageBox.ActionRole)
        box.addButton("Annuler", QMessageBox.RejectRole)
        box.exec_() if hasattr(box, "exec_") else box.exec()
        clicked = box.clickedButton()
        if clicked is b_forest:
            from ...io.ilastik_project import load_ilastik_model
            self._adopt_model(load_ilastik_model(path, mode="forest"))
            self.status.setText("Classifieur ilastik charge (features ilastik reproduites). "
                                "Entrainer ici creera un nouveau modele xwi.")
        elif clicked is b_ann:
            from ...io.ilastik_project import ilastik_annotations
            ann = ilastik_annotations(path)
            name = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            self.viewer.add_image(ann["image"], name=name, rgb=ann["channel_axis"] == -1 and
                                  ann["image"].shape[-1] in (3, 4))
            self._refresh_image_combo()
            self.image_combo.setCurrentText(name)
            names = {i + 1: n for i, n in enumerate(ann["names"])}
            colors = {i + 1: to_rgba01(c) for i, c in enumerate(ann["colors"])}
            self._set_classes(names, colors)
            if ann["labels"] is not None:
                self._labels_layer = self.viewer.add_labels(ann["labels"].astype(np.uint8),
                                                            name="annotations")
                if hasattr(self._labels_layer.events, "paint"):
                    self._labels_layer.events.paint.connect(self._on_paint)
                self._sampler = None
                self._apply_palettes()
            self.status.setText(f"Annotations ilastik importees ({len(names)} classes). "
                                "Cliquez Entrainer pour apprendre avec les features xwi.")

    def _load_ipsdk_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Dossier de modele IPSDK Explorer")
        if folder:
            try:
                self._load_ipsdk(folder)
            except Exception as exc:
                self.status.setText(f"Chargement impossible : {exc}")

    def _load_ipsdk(self, path):
        from ...io.ipsdk_explorer import ipsdk_annotations, load_ipsdk_model, read_settings, _resolve_folder

        folder = _resolve_folder(path)
        box = QMessageBox(self)
        box.setWindowTitle("Modele IPSDK Explorer")
        box.setText("Que faire de ce modele IPSDK Explorer ?")
        b_forest = box.addButton("Utiliser sa foret", QMessageBox.AcceptRole)
        b_ann = box.addButton("Importer images + annotations", QMessageBox.ActionRole)
        box.addButton("Annuler", QMessageBox.RejectRole)
        box.exec_() if hasattr(box, "exec_") else box.exec()
        clicked = box.clickedButton()
        if clicked is b_forest:
            m = load_ipsdk_model(folder)
            self._adopt_model(m)
            agr = m.source.get("annotation_agreement")
            self.status.setText("Modele IPSDK charge (features IPSDK reproduites)"
                                + (f" : {100 * agr:.1f}% d'accord avec ses annotations." if agr is not None else "."))
        elif clicked is b_ann:
            st = read_settings(folder)
            names = {c["index"] + 1: c["name"] for c in st["classes"]}
            colors = {c["index"] + 1: to_rgba01(c["color"]) for c in st["classes"] if c["color"]}
            anns = ipsdk_annotations(folder)
            for a in anns:
                self.viewer.add_image(a["image"], name=a["name"], rgb=a["channel_axis"] == -1)
            self._refresh_image_combo()
            if anns:
                self.image_combo.setCurrentText(anns[0]["name"])
            self._set_classes(names, colors)
            if anns and anns[0]["labels"] is not None:
                self._labels_layer = self.viewer.add_labels(anns[0]["labels"].astype(np.uint8), name="annotations")
                if hasattr(self._labels_layer.events, "paint"):
                    self._labels_layer.events.paint.connect(self._on_paint)
                self._sampler = None
                self._apply_palettes()
            self.status.setText(f"{len(anns)} image(s) IPSDK importee(s), {len(names)} classes. "
                                "Cliquez Entrainer pour apprendre avec les features xwi.")

    def _set_classes(self, names: dict, colors: dict):
        n = max(1, min(MAX_CLASSES, max(names) - BACKGROUND_LABEL if names else 1))
        self.class_names = dict(names)
        self.class_colors = dict(colors)
        self.n_classes_spin.blockSignals(True)
        self.n_classes_spin.setValue(n)
        self.n_classes_spin.blockSignals(False)
        self._rebuild_class_buttons()

    def _adopt_model(self, model):
        self.model = model
        colors = {k: to_rgba01(v) for k, v in model.class_colors.items()}
        self._set_classes({int(c): model.class_label(c) for c in model.classes_}, colors)
        spec = model.feature_spec
        if spec is not None:
            self.scales_edit.setText(", ".join(f"{s:g}" for s in sorted({p for f, p in spec.pairs
                                                                         if FAMILIES[f].kind != "morpho"
                                                                         and p > 0}) or DEFAULT_SCALES))
            rad = sorted({int(p) for f, p in spec.pairs if FAMILIES[f].kind == "morpho"})
            if rad:
                self.radii_edit.setText(", ".join(map(str, rad)))
            self._rebuild_feature_grid()
            sel = set(spec.pairs)
            for k, cb in self._feature_checks.items():
                cb.setChecked(k in sel)
            self.norm_combo.setCurrentText(spec.normalize)
            self.in2d_check.setChecked(spec.compute_in_2d)
            if spec.spacing:
                self.spacing_edit.setText(", ".join(f"{s:g}" for s in spec.spacing))
        self._show_importances()

    def _distill_ipsdk(self):
        data, ch = self._image()
        if data is None:
            self.status.setText("Selectionnez une image representative.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Modele IPSDK", "", "IPSDK (*.bin)")
        if not path:
            return
        from napari.qt.threading import thread_worker

        from ...io.ipsdk import distill_ipsdk_model

        worker = thread_worker(lambda: distill_ipsdk_model(path, data, features=self._build_spec(),
                                                            channel_axis=ch))()
        worker.returned.connect(lambda m: (self._set_busy(False), self._adopt_model(m), self.status.setText(
            f"Modele IPSDK distille : accord {100 * m.source['holdout_agreement']:.1f}% (pixels mis de cote).")))
        worker.errored.connect(self._job_failed)
        self._set_busy(True)
        self.status.setText("Distillation du modele IPSDK...")
        worker.start()
