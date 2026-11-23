import numpy as np
import pytest

from xwiclassifier import PixelClassifier, available_classifiers, load_model
from xwiclassifier.classifier import TrainingSampler
from xwiclassifier.features import FeatureSpec


def test_learns_and_blockwise_identical(data2d):
    img, lab = data2d
    clf = PixelClassifier(class_names={1: "fond", 2: "objet"}).fit(img, lab)
    full = clf.predict_proba(img)
    assert np.abs(full - clf.predict_proba(img, block_shape=(30, 41))).max() == 0
    seg = clf.predict(img)
    assert seg[48, 56] == 2 and seg[3, 3] == 1


def test_region_and_smoothing_and_uncertainty(data3d):
    v, lab = data3d
    clf = PixelClassifier(FeatureSpec(spacing=(2, 1, 1))).fit(v, lab)
    full = clf.predict_proba(v)
    reg = clf.predict_proba(v, region=(slice(8, 9), slice(None), slice(None)))
    assert np.allclose(reg, full[8:9])
    seg, unc = clf.predict(v, smooth=1.0, return_uncertainty=True)
    assert seg.shape == v.shape and unc.min() >= 0 and unc.max() <= 1


def test_multi_image_and_subsampling(data2d):
    img, lab = data2d
    clf = PixelClassifier(max_samples_per_class=50).fit([img, img[::-1]], [lab, lab[::-1]])
    assert clf.training_info["n_images"] == 2
    assert all(n <= 50 for n in clf.training_info["per_class"].values())


def test_sampler_cache_reuses_blocks(data3d):
    v, lab = data3d
    clf = PixelClassifier()
    sampler = TrainingSampler(clf.extractor, block_shape=(16, 16, 16))
    X1, _ = sampler.samples(v, lab, image_key="vol")
    n_cached = len(sampler._cache)
    X2, _ = sampler.samples(v, lab, image_key="vol")
    assert n_cached > 0 and len(sampler._cache) == n_cached and np.array_equal(X1, X2)


def test_save_load_roundtrip(tmp_path, data2d):
    img, lab = data2d
    clf = PixelClassifier(class_names={1: "fond", 2: "objet"}).fit(img, lab)
    p = tmp_path / "m.joblib"
    clf.save(p)
    m = load_model(p)
    assert np.array_equal(m.predict(img), clf.predict(img)) and m.class_names[2] == "objet"


def test_wrong_dimension_is_explained(data2d, data3d):
    clf = PixelClassifier().fit(*data2d)
    with pytest.raises(ValueError, match="features"):
        clf.predict(data3d[0])


def test_importances_grouped(data2d):
    clf = PixelClassifier().fit(*data2d)
    for g in (None, "family", "scale"):
        imp = clf.feature_importances(group=g)
        assert abs(sum(imp.values()) - 1) < 1e-6


@pytest.mark.parametrize("kind", [k for k in available_classifiers()])
def test_all_available_classifiers(kind, data2d):
    img, lab = data2d
    lab = lab.copy()
    lab[60:66, 90:100] = 3  # 3 classes, labels non contigus apres encodage
    clf = PixelClassifier(kind=kind, params={"n_estimators": 30}).fit(img, lab)
    seg = clf.predict(img)
    assert set(np.unique(seg)) <= {1, 2, 3}
    assert clf.feature_importances() is not None


def test_block_optimizer_respects_budget():
    from xwiclassifier.blocks import bytes_per_voxel, default_block_shape

    assert default_block_shape((40, 50), 10, (5, 5), budget_bytes=1e9) == (40, 50)
    budget = 1e9  # assez pour un bloc + halo, pas pour tout le volume
    bs = default_block_shape((96, 384, 384), 67, (23, 46, 46), budget_bytes=budget)
    outer = np.prod([min(b + 2 * h, s) for b, h, s in zip(bs, (23, 46, 46), (96, 384, 384))])
    assert outer * bytes_per_voxel(67) <= budget


def test_training_features_reused_for_prediction(data2d):
    img, lab = data2d
    clf = PixelClassifier().fit(img, lab)
    assert clf._memo is not None
    a = clf.predict_proba(img)
    clf.forget_features()
    assert np.array_equal(a, clf.predict_proba(img))
    assert np.array_equal(clf.predict_proba(img + 1.0)[..., 0].shape, img.shape)  # autre image : recalcul
