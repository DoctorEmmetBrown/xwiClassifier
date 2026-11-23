"""xwiClassifier : classification de pixels/voxels 2D/3D (alternative libre a ilastik / IPSDK Explorer).

Utilisation sans napari ::

    import xwiclassifier as xwi

    model = xwi.load_model("modele.joblib")      # ou "projet.ilp" (ilastik)
    seg = model.predict(volume)                   # calcul par blocs, memoire bornee
    seg, proba = model.predict(volume, return_proba=True)

    # entrainement
    model = xwi.PixelClassifier(xwi.FeatureSpec(spacing=(2.0, 1.0, 1.0)))
    model.fit(volume, annotations)                # 0 = non annote
    model.save("modele.joblib")
"""
import os as _os

_os.environ.setdefault("KMP_WARNINGS", "0")  # message OpenMP informatif de LightGBM

__version__ = "0.2.0"

from .backend import backend_name, gpu_available, set_device  # noqa: E402,F401
from .classifier import (  # noqa: E402,F401
    PixelClassifier, TrainingSampler, available_classifiers, labels_from_proba, uncertainty,
)
from .features import (  # noqa: E402,F401
    DEFAULT_FAMILIES, DEFAULT_RADII, DEFAULT_SCALES, FAMILIES, FeatureEngine, FeatureSpec,
    register_family,
)
from .features.ilastik import IlastikFeatures  # noqa: E402,F401
from .io import load_model, read_image, save_model, write_image  # noqa: E402,F401
from .io.ilastik_project import IlastikProject, ilastik_annotations, load_ilastik_model  # noqa: E402,F401
from .io.ipsdk import distill_ipsdk_model, inspect_binary  # noqa: E402,F401


def predict(model, image, **kwargs):
    """Raccourci : ``model`` peut etre un chemin (.joblib, .ilp) ou un PixelClassifier."""
    if isinstance(model, (str, _os.PathLike)):
        model = load_model(model)
    return model.predict(image, **kwargs)
