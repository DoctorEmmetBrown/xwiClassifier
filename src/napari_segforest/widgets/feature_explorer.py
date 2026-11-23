"""Widget d'exploration : visualise chaque canal de feature comme un layer Image.

Utile pour comprendre quelles features apportent du signal avant d'entrainer.
"""
from __future__ import annotations

import numpy as np
from qtpy.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ..features.engine import FeatureEngine, FeatureSpec


class FeatureExplorerWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.image_combo = QComboBox()
        self._refresh()
        self.viewer.layers.events.inserted.connect(lambda e: self._refresh())
        row = QHBoxLayout()
        row.addWidget(QLabel("Image :"))
        row.addWidget(self.image_combo)
        layout.addLayout(row)
        btn = QPushButton("Calculer et afficher les features")
        btn.clicked.connect(self._compute)
        layout.addWidget(btn)
        self.status = QLabel("")
        layout.addWidget(self.status)
        layout.addStretch()

    def _refresh(self):
        import napari

        self.image_combo.clear()
        for lyr in self.viewer.layers:
            if isinstance(lyr, napari.layers.Image):
                self.image_combo.addItem(lyr.name)

    def _compute(self):
        name = self.image_combo.currentText()
        if not name:
            return
        img = np.asarray(self.viewer.layers[name].data)
        engine = FeatureEngine(FeatureSpec())
        stack = engine.compute(img)
        for i, cname in enumerate(engine.channel_names):
            self.viewer.add_image(stack[..., i], name=f"feat_{cname}",
                                  visible=False, colormap="viridis")
        self.status.setText(f"{len(engine.channel_names)} canaux ajoutes.")
