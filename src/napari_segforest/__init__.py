"""napari-segforest : classification interactive d'images 2D/3D type ilastik."""

import os as _os

# Coupe le message d'info OpenMP de LightGBM/libomp (purement informatif).
_os.environ.setdefault("KMP_WARNINGS", "0")

__version__ = "0.1.0"

from .features.engine import FeatureEngine, FeatureSpec  # noqa: F401
from .classifier.base import PixelClassifier  # noqa: F401
