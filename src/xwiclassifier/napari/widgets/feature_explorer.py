"""Explorateur de features : affiche les canaux de features de la vue courante."""
from __future__ import annotations

import numpy as np
from qtpy.QtWidgets import QComboBox, QHBoxLayout, QLabel, QListWidget, QPushButton, QVBoxLayout, QWidget

from ...features import FeatureEngine, FeatureSpec


class FeatureExplorerWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("Image :"))
        self.image_combo = QComboBox()
        row.addWidget(self.image_combo)
        lay.addLayout(row)
        btn = QPushButton("Calculer les features (coupe affichee)")
        btn.clicked.connect(self._compute)
        lay.addWidget(btn)
        self.status = QLabel("Jeu de features par defaut ; une pile (feature, y, x) est ajoutee.")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.names = QListWidget()
        lay.addWidget(self.names)
        self.viewer.layers.events.inserted.connect(lambda e: self._refresh())
        self.viewer.layers.events.removed.connect(lambda e: self._refresh())
        self._refresh()

    def _refresh(self):
        import napari
        cur = self.image_combo.currentText()
        self.image_combo.clear()
        for lyr in self.viewer.layers:
            if isinstance(lyr, napari.layers.Image) and not lyr.name.startswith("features"):
                self.image_combo.addItem(lyr.name)
        i = self.image_combo.findText(cur)
        if i >= 0:
            self.image_combo.setCurrentIndex(i)

    def _compute(self):
        name = self.image_combo.currentText()
        if not name:
            return
        data = np.asarray(self.viewer.layers[name].data)
        eng = FeatureEngine(FeatureSpec())
        if data.ndim == 3:
            z = int(np.clip(self.viewer.dims.current_step[-3], 0, data.shape[0] - 1))
            halo = eng.halo(3)[0]
            lo, hi = max(0, z - halo), min(data.shape[0], z + halo + 1)
            feats = eng.compute(data[lo:hi])[z - lo]
            label = f"features (z={z})"
            names = eng.channel_names(3)
        else:
            feats = eng.compute(data)
            label = "features"
            names = eng.channel_names(2)
        self.viewer.add_image(np.moveaxis(feats, -1, 0), name=label, colormap="viridis")
        self.names.clear()
        self.names.addItems([f"{k}: {n}" for k, n in enumerate(names)])
        self.status.setText(f"{len(names)} canaux ajoutes (curseur 'feature' = indice ci-dessous).")
