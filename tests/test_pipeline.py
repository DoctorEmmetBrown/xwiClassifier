"""Tests headless (sans GPU ni Qt) du moteur de features et du classifieur."""
import numpy as np
import pytest

from napari_segforest.features.engine import FeatureEngine, FeatureSpec
from napari_segforest.classifier.base import PixelClassifier


def _toy_2d():
    img = np.zeros((64, 64), np.float32)
    img[16:48, 16:48] = 1.0
    rng = np.random.default_rng(0)
    img += 0.2 * rng.standard_normal(img.shape).astype(np.float32)
    return img


def test_feature_engine_shape():
    img = _toy_2d()
    eng = FeatureEngine(FeatureSpec(families=("intensity", "gaussian", "gradient"),
                                    sigmas=(0, 1, 2)))
    stack = eng.compute(img)
    assert stack.shape[:2] == img.shape
    assert stack.shape[-1] == len(eng.channel_names)
    assert stack.dtype == np.float32


def test_classifier_learns_square():
    img = _toy_2d()
    labels = np.zeros(img.shape, np.uint8)
    labels[24:40, 24:40] = 1   # interieur
    labels[:8, :8] = 2         # fond
    clf = PixelClassifier(kind="rf", feature_spec=FeatureSpec(sigmas=(0, 1, 2)))
    clf.fit(img, labels)
    pred = clf.predict(img)
    # le centre doit etre classe 1, un coin classe 2
    assert pred[32, 32] == 1
    assert pred[2, 2] == 2
    assert clf.feature_importances() is not None


def test_save_load(tmp_path):
    img = _toy_2d()
    labels = np.zeros(img.shape, np.uint8)
    labels[24:40, 24:40] = 1
    labels[:8, :8] = 2
    clf = PixelClassifier(feature_spec=FeatureSpec(sigmas=(0, 1)))
    clf.fit(img, labels)
    p = tmp_path / "m.joblib"
    clf.save(str(p))
    clf2 = PixelClassifier.load(str(p))
    assert np.array_equal(clf2.predict(img), clf.predict(img))
