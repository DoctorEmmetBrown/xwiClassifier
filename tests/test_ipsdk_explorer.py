import warnings

import joblib
import numpy as np
import pytest

from conftest import make_fake_ipsdk_model, toy_2d, toy_3d
from xwiclassifier import FeatureEngine, FeatureSpec, load_model
from xwiclassifier.features.ipsdk import IPSDKFeatures, ipsdk_gaussian_kernel
from xwiclassifier.io.ipsdk_explorer import (
    ipsdk_annotations, ipsdk_output_values, load_ipsdk_model, read_settings, to_xwi_spec,
)


def _ref(rf, feats, img):
    return rf.predict_proba(feats.compute(img).reshape(-1, rf.n_features_in_)).reshape(img.shape + (-1,))


def test_settings_parsing_and_ascii_names(tmp_path):
    img, lab = toy_2d()
    make_fake_ipsdk_model(str(tmp_path / "m"), [img], [lab], names=("Fond é", "Fils"))
    st = read_settings(str(tmp_path / "m"))
    assert [c["name"] for c in st["classes"]] == ["Fond é", "Fils"]
    assert st["order"][0] == "Gaussian" and st["sizes"] == [1, 2, 3]


@pytest.mark.parametrize("old", [False, True])
def test_forest_reproduced_including_sklearn_024_pickles(tmp_path, old):
    img, lab = toy_2d()
    rf, feats = make_fake_ipsdk_model(str(tmp_path / "m"), [img], [lab], old_sklearn=old)
    if old:
        with pytest.raises(ValueError, match="incompatible"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            joblib.load(tmp_path / "m" / "Model")  # chargement direct impossible : ancien format
    m = load_model(str(tmp_path / "m"))
    assert np.abs(m.predict_proba(img) - _ref(rf, feats, img)).max() < 1e-5
    assert m.source["annotation_agreement"] == 1.0
    assert m.class_names == {1: "Fond", 2: "Objet"}
    assert set(np.unique(ipsdk_output_values(m, m.predict(img)))) <= {0, 1}


@pytest.mark.parametrize("dim", ["3D", "2D"])
def test_volumes_blockwise(tmp_path, dim):
    v, lab = toy_3d()
    rf, feats = make_fake_ipsdk_model(str(tmp_path / "m"), [v], [lab], dimension=dim)
    m = load_model(str(tmp_path / "m"))
    assert np.abs(m.predict_proba(v, block_shape=(8, 20, 20)) - _ref(rf, feats, v)).max() < 1e-5


def test_calibration_finds_zoom_convention(tmp_path):
    img, lab = toy_2d()
    rf, feats = make_fake_ipsdk_model(str(tmp_path / "m"), [img], [lab], multires=(1, 2, 4), zoom_mode="center")
    m = load_ipsdk_model(str(tmp_path / "m"))
    assert m.extractor.zoom_mode == "center"
    assert np.abs(m.predict_proba(img) - _ref(rf, feats, img)).max() < 1e-5


def test_retrain_modes_and_annotations(tmp_path):
    img, lab = toy_2d()
    make_fake_ipsdk_model(str(tmp_path / "m"), [img], [lab])
    anns = ipsdk_annotations(str(tmp_path / "m"))
    assert np.array_equal(anns[0]["labels"], lab)
    for feats in (None, "ipsdk", "equivalent"):
        m = load_ipsdk_model(str(tmp_path / "m"), mode="retrain", features=feats)
        assert m.predict(img)[48, 56] == 2
    spec = to_xwi_spec(read_settings(str(tmp_path / "m")))
    assert spec.morpho_shape == "exact" and ("high_pass", 2.0) in spec.pairs


def test_ipsdk_filters_transcribed_in_xwi_engine():
    rng = np.random.default_rng(0)
    img = (rng.random((50, 60)) * 100).astype(np.float32)
    pairs = [("mean", 2.0), ("high_pass", 2.0), ("box_variance", 3.0), ("erosion", 4.0),
             ("dilation", 4.0), ("opening", 3.0), ("closing", 3.0)]
    xwi = FeatureEngine(FeatureSpec(pairs=tuple(pairs), morpho_shape="exact")).compute(img)
    ip = IPSDKFeatures([(1, "Mean", 2), (1, "High Pass", 2), (1, "Variance", 3), (1, "Erosion", 4),
                        (1, "Dilation", 4), (1, "Opening", 3), (1, "Closing", 3)]).compute(img)[..., 1:]
    assert np.abs(xwi - ip).max() < 1e-3
    g = IPSDKFeatures([(1, "Gaussian", 2)]).compute(img)[15:-15, 15:-15, 1]
    gx = FeatureEngine(FeatureSpec(pairs=(("gaussian", 2.0),))).compute(img)[15:-15, 15:-15, 0]
    assert np.abs(g - gx).max() < 1e-3  # meme noyau a l'interieur (bords : miroir vs reflexion)


def test_gaussian_coverage_kernel():
    for s in (1, 2, 5):
        k = ipsdk_gaussian_kernel(float(s))
        assert len(k) // 2 == max(2, 3 * s) and abs(k.sum() - 1) < 1e-9
    const = np.full((30, 30), 7.0, np.float32)
    f = IPSDKFeatures([(1, "High Pass", 2), (1, "Variance", 2)]).compute(const)
    assert np.allclose(f[..., 1], 7.0) and np.allclose(f[..., 2], 0.0)
