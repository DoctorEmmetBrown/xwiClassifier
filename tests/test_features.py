import numpy as np
import pytest

from xwiclassifier.blocks import iter_blocks, with_halo
from xwiclassifier.features import FAMILIES, FeatureEngine, FeatureSpec
from xwiclassifier.features.eigen import eig2_sym, eig3_sym

ALL = tuple(FAMILIES)


def test_default_spec_shapes(data2d):
    img, _ = data2d
    eng = FeatureEngine()
    st = eng.compute(img)
    assert st.shape == img.shape + (len(eng.channel_names(2)),)
    assert st.dtype == np.float32 and np.isfinite(st).all()


@pytest.mark.parametrize("spacing,in2d", [(None, False), ((2.0, 1.0, 1.0), False), (None, True)])
def test_blockwise_equals_global_all_families(spacing, in2d):
    rng = np.random.default_rng(3)
    img = rng.random((22, 40, 38)).astype(np.float32)
    spec = FeatureSpec.from_grid(families=ALL, scales=(0.7, 1.6, 3.5), radii=(1, 3),
                                 spacing=spacing, compute_in_2d=in2d)
    eng = FeatureEngine(spec)
    full = eng.compute(img)
    halo = eng.halo(3)
    blk = np.empty_like(full)
    for inner in iter_blocks(img.shape, (9, 15, 14)):
        outer, local = with_halo(inner, halo, img.shape)
        blk[inner] = eng.compute(img[outer])[local]
    assert np.abs(full - blk).max() < 1e-5


def test_derivatives_exact_on_ramp_and_constant():
    ramp = (2.0 * np.arange(64, dtype=np.float32))[None, :].repeat(64, 0)
    eng = FeatureEngine(FeatureSpec(pairs=(("gradient", 1.0), ("log", 1.0), ("gradient", 0.7))))
    f = eng.compute(ramp)
    assert f[32, 32, 0] == pytest.approx(2.0, abs=1e-4)
    assert abs(f[32, 32, 1]) < 1e-4
    const = eng.compute(np.full((32, 32), 67.0, np.float32))
    assert np.abs(const[..., 1]).max() < 1e-4


def test_analytic_eigenvalues_match_numpy():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(5000, 3, 3))
    a = a + a.transpose(0, 2, 1)
    ref = np.sort(np.linalg.eigvalsh(a), axis=1)[:, ::-1]
    got = np.stack(eig3_sym(a[:, 0, 0], a[:, 1, 1], a[:, 2, 2], a[:, 0, 1], a[:, 0, 2], a[:, 1, 2]), 1)
    assert np.abs(ref - got).max() < 1e-4
    b = a[:, :2, :2]
    ref2 = np.sort(np.linalg.eigvalsh(b), axis=1)[:, ::-1]
    assert np.abs(ref2 - np.stack(eig2_sym(b[:, 0, 0], b[:, 0, 1], b[:, 1, 1]), 1)).max() < 1e-4


def test_anisotropic_spacing_reduces_z_smoothing():
    img = np.zeros((21, 21, 21), np.float32)
    img[10, 10, 10] = 1.0
    iso = FeatureEngine(FeatureSpec(pairs=(("gaussian", 2.0),))).compute(img)[..., 0]
    ani = FeatureEngine(FeatureSpec(pairs=(("gaussian", 2.0),), spacing=(4.0, 1.0, 1.0))).compute(img)[..., 0]
    assert ani[10, 10, 10] > iso[10, 10, 10]  # moins etale en z
    assert ani[13, 10, 10] < iso[13, 10, 10]


def test_normalization_is_global_not_per_block():
    rng = np.random.default_rng(0)
    img = (rng.random((64, 64)) * 1000).astype(np.float32)
    eng = FeatureEngine(FeatureSpec(pairs=(("intensity", 0.0),), normalize="percentile"))
    norm = eng.fit_normalization(img)
    a = eng.compute(img[:20], norm=norm)
    b = eng.compute(img, norm=norm)[:20]
    assert np.allclose(a, b)


def test_spec_roundtrip_and_validation():
    spec = FeatureSpec.from_grid(spacing=(2, 1, 1), normalize="zscore")
    assert FeatureSpec.from_dict(spec.to_dict()) == spec
    with pytest.raises(ValueError):
        FeatureSpec(pairs=(("inconnue", 1.0),))


def test_multichannel_input(data2d):
    img, _ = data2d
    rgb = np.stack([img, 2 * img, -img], axis=-1)
    eng = FeatureEngine(FeatureSpec(pairs=(("gaussian", 1.0), ("hessian_eigen", 1.0))))
    st = eng.compute(rgb, channel_axis=-1)
    assert st.shape[-1] == eng.n_features(2, 3) == 9
