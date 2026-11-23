"""Widget napari de classification de pixels interactive (style ilastik).

Flux d'usage :
  1. L'utilisateur charge une image (2D ou 3D) dans napari.
  2. Le widget cree un layer Labels "annotations" pour peindre les classes.
  3. On choisit les familles de features + sigmas (catalogue ouvert).
  4. "Entrainer" -> RF/boosting sur les pixels peints.
  5. "Predire" -> ajoute un layer Labels avec la segmentation complete.
  6. "Live" -> reentraine/predit a chaque coup de pinceau.
"""
from __future__ import annotations

import numpy as np
from qtpy.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QPushButton, QScrollArea, QSlider,
    QSpinBox, QVBoxLayout, QWidget,
)
from qtpy.QtCore import Qt

# Convention de labels : 0 = non peint, 1 = Background, 2..6 = classes 1..5.
BACKGROUND_LABEL = 1
MAX_CLASSES = 5
# Couleurs lisibles pour le layer Labels (index = valeur de label).
CLASS_COLORS = {
    1: (0.6, 0.6, 0.6, 1.0),   # background gris
    2: (0.90, 0.10, 0.10, 1.0),
    3: (0.10, 0.60, 0.90, 1.0),
    4: (0.20, 0.80, 0.20, 1.0),
    5: (0.95, 0.75, 0.10, 1.0),
    6: (0.70, 0.20, 0.80, 1.0),
}

from ..backend import backend_name
from ..classifier.base import PixelClassifier
from ..features.engine import (
    DEFAULT_FAMILIES, DEFAULT_SIGMAS, FEATURE_FAMILIES, MORPHO_FAMILIES, FeatureSpec,
)


class PixelClassifierWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.clf: PixelClassifier | None = None
        self._labels_layer = None
        self._result_layer = None
        self.class_names: dict[int, str] = {}   # {valeur_label: nom}
        self._name_edits: dict[int, object] = {}
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        layout.addWidget(QLabel(f"Backend : {backend_name()}"))

        # Image source
        self.image_combo = QComboBox()
        self._refresh_image_combo()
        self.viewer.layers.events.inserted.connect(lambda e: self._refresh_image_combo())
        self.viewer.layers.events.removed.connect(lambda e: self._refresh_image_combo())
        row = QHBoxLayout()
        row.addWidget(QLabel("Image :"))
        row.addWidget(self.image_combo)
        layout.addLayout(row)

        # ---- Gestion des classes -------------------------------------
        row = QHBoxLayout()
        row.addWidget(QLabel("Nombre de classes (hors background) :"))
        self.n_classes_spin = QSpinBox()
        self.n_classes_spin.setRange(1, MAX_CLASSES)
        self.n_classes_spin.setValue(2)
        self.n_classes_spin.valueChanged.connect(self._rebuild_class_buttons)
        row.addWidget(self.n_classes_spin)
        layout.addLayout(row)

        layout.addWidget(QLabel("Classe active (clic = couleur du pinceau) :"))
        self.class_btn_container = QGridLayout()
        layout.addLayout(self.class_btn_container)
        self.class_group = QButtonGroup(self)
        self.class_group.setExclusive(True)
        self._rebuild_class_buttons()

        # ---- Taille du pinceau ---------------------------------------
        row = QHBoxLayout()
        row.addWidget(QLabel("Taille du pinceau :"))
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setRange(1, 50)
        self.brush_slider.setValue(5)
        self.brush_spin = QSpinBox()
        self.brush_spin.setRange(1, 200)
        self.brush_spin.setValue(5)
        # synchronise slider <-> spinbox <-> layer
        self.brush_slider.valueChanged.connect(self.brush_spin.setValue)
        self.brush_spin.valueChanged.connect(self._on_brush_size)
        row.addWidget(self.brush_slider)
        row.addWidget(self.brush_spin)
        layout.addLayout(row)

        # Matrice de features : lignes = familles, colonnes = tailles (sigmas).
        # Chaque case cochee = couple (famille, sigma) calcule. Tout coche par defaut.
        layout.addWidget(QLabel(
            "Features (famille x taille) — decochez pour exclure.\n"
            "Lisses : taille = sigma (px). Morpho : taille = rayon (px)."
        ))
        self.feature_grid = QGridLayout()
        self.feature_grid.setHorizontalSpacing(4)
        self.feature_grid.setVerticalSpacing(2)
        self._feature_checks: dict[tuple[str, float], QCheckBox] = {}
        # en-tetes de colonnes (sigma / rayon)
        self.feature_grid.addWidget(QLabel("σ/r:"), 0, 0)
        for c, s in enumerate(DEFAULT_SIGMAS):
            self.feature_grid.addWidget(QLabel(f"{s:g}"), 0, c + 1, Qt.AlignCenter)
        for r, fam in enumerate(FEATURE_FAMILIES):
            self.feature_grid.addWidget(QLabel(fam), r + 1, 0)
            for c, s in enumerate(DEFAULT_SIGMAS):
                # intensity n'a de sens qu'a sigma 0 ; morpho seulement a rayon >= 1
                if fam == "intensity" and s != 0:
                    continue
                if fam in MORPHO_FAMILIES and s == 0:
                    continue
                cb = QCheckBox()
                cb.setChecked(fam in DEFAULT_FAMILIES)
                self.feature_grid.addWidget(cb, r + 1, c + 1, Qt.AlignCenter)
                self._feature_checks[(fam, float(s))] = cb
        # zone defilante pour la matrice (plus compacte visuellement)
        grid_holder = QWidget()
        grid_holder.setLayout(self.feature_grid)
        feat_scroll = QScrollArea()
        feat_scroll.setWidgetResizable(True)
        feat_scroll.setWidget(grid_holder)
        feat_scroll.setMinimumHeight(180)
        feat_scroll.setMaximumHeight(260)
        layout.addWidget(feat_scroll)

        # raccourcis : (de)cocher une colonne (taille) entiere
        row = QHBoxLayout()
        row.addWidget(QLabel("Cocher/decocher une taille :"))
        self.col_combo = QComboBox()
        self.col_combo.addItems([f"{s:g}" for s in DEFAULT_SIGMAS])
        row.addWidget(self.col_combo)
        b_on = QPushButton("Cocher")
        b_off = QPushButton("Decocher")
        b_on.clicked.connect(lambda: self._toggle_sigma_column(True))
        b_off.clicked.connect(lambda: self._toggle_sigma_column(False))
        row.addWidget(b_on)
        row.addWidget(b_off)
        layout.addLayout(row)

        # Classifieur
        row = QHBoxLayout()
        row.addWidget(QLabel("Classifieur :"))
        self.kind_combo = QComboBox()
        from ..classifier.base import available_classifiers

        avail = available_classifiers()
        self.kind_combo.addItems(avail)
        missing = [k for k in ("lightgbm", "xgboost") if k not in avail]
        if missing:
            self.kind_combo.setToolTip(
                "Non installes : " + ", ".join(missing)
                + " (pip install " + " ".join(missing) + ")"
            )
        row.addWidget(self.kind_combo)
        layout.addLayout(row)

        # Boutons
        self.new_labels_btn = QPushButton("Creer/selectionner layer d'annotation")
        self.new_labels_btn.clicked.connect(self._ensure_labels_layer)
        layout.addWidget(self.new_labels_btn)

        self.train_btn = QPushButton("Entrainer")
        self.train_btn.clicked.connect(self._train)
        layout.addWidget(self.train_btn)

        self.predict_btn = QPushButton("Predire (volume complet)")
        self.predict_btn.clicked.connect(self._predict)
        layout.addWidget(self.predict_btn)

        self.live_check = QCheckBox("Mode live (reentraine au pinceau)")
        self.live_check.stateChanged.connect(self._toggle_live)
        layout.addWidget(self.live_check)

        row = QHBoxLayout()
        self.save_btn = QPushButton("Sauver modele")
        self.save_btn.clicked.connect(self._save_model)
        self.load_btn = QPushButton("Charger modele")
        self.load_btn.clicked.connect(self._load_model)
        row.addWidget(self.save_btn)
        row.addWidget(self.load_btn)
        layout.addLayout(row)

        self.status = QLabel("Pret.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        # Importances des features (% de contribution a la decision)
        layout.addWidget(QLabel("Contribution des features a la decision (%) :"))
        self.importance_list = QListWidget()
        self.importance_list.setMaximumHeight(160)
        layout.addWidget(self.importance_list)
        layout.addStretch()

    # ------------------------------------------------------------------
    def _default_class_name(self, value: int) -> str:
        if value == BACKGROUND_LABEL:
            return "Background"
        return f"Classe {value - BACKGROUND_LABEL}"

    def _rebuild_class_buttons(self):
        # purge l'ancienne grille (boutons + champs noms)
        for btn in list(self.class_group.buttons()):
            self.class_group.removeButton(btn)
        while self.class_btn_container.count():
            item = self.class_btn_container.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        self._name_edits = {}

        values = [BACKGROUND_LABEL]
        values += [BACKGROUND_LABEL + 1 + c for c in range(self.n_classes_spin.value())]
        self._class_buttons: dict[int, object] = {}
        for row, value in enumerate(values):
            name0 = self.class_names.get(value, self._default_class_name(value))
            # bouton colore portant le nom de la classe = selection du pinceau
            btn = QPushButton(name0)
            btn.setCheckable(True)
            r, g, b, _ = CLASS_COLORS[value]
            txt_col = "black" if value == BACKGROUND_LABEL else "white"
            btn.setStyleSheet(
                f"QPushButton {{ background-color: rgb({int(r*255)},{int(g*255)},{int(b*255)});"
                f" color: {txt_col}; padding: 4px; }}"
                f"QPushButton:checked {{ border: 3px solid black; font-weight: bold; }}"
            )
            self.class_group.addButton(btn, value)
            btn.clicked.connect(lambda _checked=False, v=value: self._on_class_selected(v))
            self._class_buttons[value] = btn
            # champ nom editable -> met a jour le texte du bouton
            edit = QLineEdit(name0)
            edit.textChanged.connect(
                lambda txt, v=value: self._on_name_changed(v, txt)
            )
            self._name_edits[value] = edit
            self.class_names.setdefault(value, name0)
            self.class_btn_container.addWidget(btn, row, 0)
            self.class_btn_container.addWidget(edit, row, 1)

        # purge les noms des classes supprimees
        for v in list(self.class_names):
            if v not in values:
                self.class_names.pop(v, None)

        first = self.class_group.button(BACKGROUND_LABEL)
        if first:
            first.setChecked(True)
        self._apply_palette()

    def _on_brush_size(self, size):
        # garde le slider synchro et applique au layer si present
        if self.brush_slider.value() != size and size <= self.brush_slider.maximum():
            self.brush_slider.setValue(size)
        if self._labels_layer is not None and self._labels_layer in self.viewer.layers:
            self._labels_layer.brush_size = size

    def _on_name_changed(self, value, txt):
        self.class_names[value] = txt
        btn = self._class_buttons.get(value)
        if btn is not None:
            btn.setText(txt)

    def _on_class_selected(self, value):
        try:
            btn = self.class_group.button(value)
            if btn is not None and not btn.isChecked():
                btn.setChecked(True)
            # cree le layer d'annotation au besoin
            if self._labels_layer is None or self._labels_layer not in self.viewer.layers:
                if self._get_image() is None:
                    self.status.setText("Selectionnez d'abord une image.")
                    return
                self._ensure_labels_layer()
            self.viewer.layers.selection.active = self._labels_layer
            self._labels_layer.selected_label = int(value)
            self._labels_layer.brush_size = self.brush_spin.value()
            self._labels_layer.mode = "paint"
            self._labels_layer.refresh()
            self.status.setText(
                f"Pinceau actif : « {self.class_names.get(value, value)} » "
                f"(valeur {value}, taille {self.brush_spin.value()})."
            )
        except Exception as exc:
            self.status.setText(f"Erreur selection classe : {exc}")

    def _palette_dict(self) -> dict:
        n = self.n_classes_spin.value()
        return {v: CLASS_COLORS[v] for v in range(1, BACKGROUND_LABEL + 1 + n)}

    def _set_layer_colors(self, layer) -> None:
        """Force les couleurs CLASS_COLORS sur un layer Labels, tous napari confondus."""
        if layer is None or layer not in self.viewer.layers:
            return
        color = self._palette_dict()
        # napari >= 0.5 : DirectLabelColormap (chemin d'import correct)
        try:
            from napari.utils.colormaps import DirectLabelColormap

            cd = {int(k): list(v) for k, v in color.items()}
            cd[None] = [0.0, 0.0, 0.0, 0.0]   # fond / non labellise transparent
            layer.colormap = DirectLabelColormap(color_dict=cd)
            return
        except Exception:
            pass
        # napari < 0.5 : mode direct + dict color
        try:
            layer.color_mode = "direct"
            layer.color = color
        except Exception:
            pass

    def _apply_palette(self):
        """Applique les couleurs fixes aux annotations (et a la segmentation)."""
        self._set_layer_colors(self._labels_layer)
        self._set_layer_colors(self._result_layer)

    def _refresh_image_combo(self):
        import napari

        current = self.image_combo.currentText() if self.image_combo.count() else None
        self.image_combo.clear()
        for lyr in self.viewer.layers:
            if isinstance(lyr, napari.layers.Image):
                self.image_combo.addItem(lyr.name)
        if current:
            idx = self.image_combo.findText(current)
            if idx >= 0:
                self.image_combo.setCurrentIndex(idx)

    def _get_image(self) -> np.ndarray | None:
        name = self.image_combo.currentText()
        if not name or name not in self.viewer.layers:
            return None
        return np.asarray(self.viewer.layers[name].data)

    def _ensure_labels_layer(self):
        img = self._get_image()
        if img is None:
            self.status.setText("Aucune image selectionnee.")
            return
        if self._labels_layer is None or self._labels_layer not in self.viewer.layers:
            self._labels_layer = self.viewer.add_labels(
                np.zeros(img.shape, dtype=np.uint8), name="annotations"
            )
        self._apply_palette()
        self.viewer.layers.selection.active = self._labels_layer
        self._labels_layer.selected_label = BACKGROUND_LABEL
        self._labels_layer.brush_size = self.brush_spin.value()
        self._labels_layer.mode = "paint"
        # selectionne le bouton Background dans l'UI
        bg_btn = self.class_group.button(BACKGROUND_LABEL)
        if bg_btn:
            bg_btn.setChecked(True)
        self.status.setText(
            "Background pre-selectionne. Choisissez une classe puis peignez."
        )

    def _toggle_sigma_column(self, checked: bool):
        s = float(self.col_combo.currentText())
        for (fam, sig), cb in self._feature_checks.items():
            if sig == s:
                cb.setChecked(checked)

    def _build_spec(self) -> FeatureSpec:
        pairs = tuple((fam, sig) for (fam, sig), cb in self._feature_checks.items()
                      if cb.isChecked())
        if not pairs:
            # garde-fou : au moins l'intensite brute
            pairs = (("intensity", 0.0),)
        return FeatureSpec(pairs=pairs)

    # ------------------------------------------------------------------
    def _train(self):
        img = self._get_image()
        if img is None or self._labels_layer is None:
            self.status.setText("Image et annotations requises.")
            return
        labels = np.asarray(self._labels_layer.data)
        if (labels > 0).sum() == 0:
            self.status.setText("Aucun pixel peint.")
            return
        self.status.setText("Calcul des features + entrainement...")
        try:
            self.clf = PixelClassifier(kind=self.kind_combo.currentText(),
                                       feature_spec=self._build_spec(),
                                       class_names=dict(self.class_names))
            self.clf.fit(img, labels)
        except ImportError as exc:
            self.clf = None
            self.status.setText(str(exc).splitlines()[0])
            return
        except Exception as exc:  # surface toute autre erreur sans crasher napari
            self.clf = None
            self.status.setText(f"Echec entrainement : {exc}")
            return
        self._show_importances()
        named = ", ".join(self.class_names[v] for v in sorted(self.class_names))
        self.status.setText(f"Entraine [{named}]. Voir contributions ci-dessous.")

    def _show_importances(self):
        self.importance_list.clear()
        imp = self.clf.feature_importances() if self.clf else None
        if not imp:
            self.importance_list.addItem("(classifieur sans importances)")
            return
        total = sum(imp.values()) or 1.0
        for name, val in imp.items():
            pct = 100.0 * val / total
            bar = "█" * int(round(pct / 2))  # 1 bloc = 2 %
            self.importance_list.addItem(f"{pct:5.1f} %  {bar}  {name}")

    def _predict(self):
        img = self._get_image()
        if self.clf is None or img is None:
            self.status.setText("Entrainez d'abord un modele.")
            return
        self.status.setText("Prediction...")
        pred = self.clf.predict(img).astype(np.uint8)
        if self._result_layer is None or self._result_layer not in self.viewer.layers:
            self._result_layer = self.viewer.add_labels(pred, name="segmentation")
        else:
            self._result_layer.data = pred
        self._set_layer_colors(self._result_layer)  # meme palette que les annotations
        self.status.setText("Segmentation mise a jour.")

    def _toggle_live(self, state):
        if self._labels_layer is None:
            return
        if state == Qt.Checked:
            self._labels_layer.events.paint.connect(self._on_paint)
            self.status.setText("Live ON.")
        else:
            try:
                self._labels_layer.events.paint.disconnect(self._on_paint)
            except (TypeError, ValueError):
                pass
            self.status.setText("Live OFF.")

    def _on_paint(self, event=None):
        self._train()
        self._predict()

    def _save_model(self):
        if self.clf is None:
            self.status.setText("Rien a sauver.")
            return
        from qtpy.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(self, "Sauver", "model.joblib",
                                              "Joblib (*.joblib)")
        if path:
            self.clf.save(path)
            self.status.setText(f"Sauve : {path}")

    def _load_model(self):
        from qtpy.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(self, "Charger", "", "Joblib (*.joblib)")
        if path:
            self.clf = PixelClassifier.load(path)
            # restaure les noms et le nombre de classes dans l'UI
            if self.clf.class_names:
                self.class_names = dict(self.clf.class_names)
                n = max(0, max(self.class_names) - BACKGROUND_LABEL)
                self.n_classes_spin.setValue(min(max(n, 1), MAX_CLASSES))
                self._rebuild_class_buttons()
            self.status.setText(f"Charge : {path}")
