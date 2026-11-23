import os

import numpy as np
import pytest

from conftest import make_fake_ilp, toy_2d, toy_3d
from xwiclassifier import load_model
from xwiclassifier.classifier.vigra_forest import (
    _strict_to_le, predict_proba_reference, read_vigra_forest, to_sklearn_forest,
)
from xwiclassifier.io.ilastik_project import IlastikProject, ilastik_annotations, load_ilastik_model


@pytest.mark.parametrize("maker", [toy_2d, toy_3d])
def test_fake_project_forest_reproduces_training_forest(tmp_path, maker):
    img, lab = maker()
    p = str(tmp_path / "p.ilp")
    rf, feats = make_fake_ilp(p, img, lab)
    m = load_ilastik_model(p, mode="forest")
    X = feats.compute(img).reshape(-1, m.n_features_)
    ref = rf.predict_proba(X)
    got = m.predict_proba(img).reshape(-1, 2)
    assert np.abs(ref - got).max() < 1e-5
    assert m.class_names == {1: "fond", 2: "objet"}


def test_labels_and_annotations_roundtrip(tmp_path):
    img, lab = toy_3d()
    p = str(tmp_path / "p.ilp")
    make_fake_ilp(p, img, lab, train_forest=False)
    ann = ilastik_annotations(p)
    assert np.array_equal(ann["labels"], lab) and np.array_equal(ann["image"], img)
    with pytest.raises(ValueError, match="retrain"):
        load_ilastik_model(p, mode="forest")
    m = load_ilastik_model(p, mode="retrain")
    assert m.predict(img)[10, 24, 27] == 2


def test_external_raw_data(tmp_path):
    from xwiclassifier.io import write_image

    img, lab = toy_2d()
    raw = tmp_path / "raw.tif"
    write_image(raw, img)
    p = str(tmp_path / "p.ilp")
    make_fake_ilp(p, img, lab, train_forest=False, internal=False, raw_file="raw.tif")
    with IlastikProject(p) as prj:
        r, axes = prj.read_raw(0)
    assert np.array_equal(r, img) and axes == "yx"


def test_threshold_conversion_strict_vs_le():
    t = np.array([0.1, 1.0, -2.5, 3.14159265358979, 1e-8])
    tp = _strict_to_le(t)
    for ti, tpi in zip(t, tp):
        xs = np.array([np.float32(ti), np.nextafter(np.float32(ti), np.float32(-np.inf)),
                       np.nextafter(np.float32(ti), np.float32(np.inf))], np.float32)
        assert np.array_equal(xs < ti, xs.astype(np.float64) <= tpi)


def test_sklearn_conversion_matches_reference(tmp_path):
    import h5py

    img, lab = toy_2d()
    p = str(tmp_path / "p.ilp")
    make_fake_ilp(p, img, lab)
    with h5py.File(p, "r") as f:
        fo = read_vigra_forest(f["PixelClassification/ClassifierForests/Forest0000"])
    rf = to_sklearn_forest(fo["trees"], fo["n_features"], fo["n_classes"])
    X = np.random.default_rng(0).normal(0, 3, size=(5000, fo["n_features"])).astype(np.float32)
    assert np.abs(rf.predict_proba(X) - predict_proba_reference(fo["trees"], X)).max() < 1e-6


REAL = os.environ.get("XWI_ILASTIK_TEST_PROJECT")


@pytest.mark.skipif(not REAL, reason="XWI_ILASTIK_TEST_PROJECT non defini (projet .ilp reel)")
def test_real_ilastik_project_agrees_with_annotations():
    m = load_model(REAL)
    ann = ilastik_annotations(REAL)
    seg = m.predict(ann["image"], channel_axis=ann["channel_axis"])
    mask = ann["labels"] > 0
    assert (seg[mask] == ann["labels"][mask]).mean() > 0.98
