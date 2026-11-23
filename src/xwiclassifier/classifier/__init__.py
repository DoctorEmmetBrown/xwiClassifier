"""Classifieurs de pixels."""
from .estimators import available_classifiers, make_estimator  # noqa: F401
from .model import PixelClassifier, labels_from_proba, uncertainty  # noqa: F401
from .sampler import TrainingSampler  # noqa: F401
