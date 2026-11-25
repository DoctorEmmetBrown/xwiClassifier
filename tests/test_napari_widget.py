"""Tests du widget napari (ignores si napari / Qt ne sont pas disponibles)."""
import numpy as np
import pytest

napari = pytest.importorskip("napari")
pytest.importorskip("pytestqt")

from xwiclassifier.napari.colors import default_color  # noqa: E402
from xwiclassifier.napari.widgets.pixel_classifier import PixelClassifierWidget  # noqa: E402


def _volume():
    rng = np.random.default_rng(0)
    v = np.zeros((12, 64, 64), np.float32)
    v[3:9, 16:48, 16:48] = 1.0
    return v + 0.25 * rng.standard_normal(v.shape).astype(np.float32)


def _color_of(layer, value):
    try:
        return tuple(np.round(layer.colormap.map(np.array([value]))[0][:3], 3))
    except Exception:
        return tuple(np.round(layer.get_color(value)[:3], 3))


@pytest.fixture
def widget(qtbot):
    # ViewerModel : meme API (layers, dims) sans canevas OpenGL -> testable sans ecran
    from napari.components import ViewerModel

    viewer = ViewerModel()
    viewer.add_image(_volume(), name="vol", scale=(2.0, 1.0, 1.0))
    w = PixelClassifierWidget(viewer)
    qtbot.addWidget(w)
    return viewer, w


def test_class_buttons_set_brush_label_and_color(widget):
    viewer, w = widget
    w._ensure_labels_layer()
    lab = w._labels_layer
    assert lab.data.shape == (12, 64, 64)
    for value in (2, 3, 1):
        w._class_buttons[value].click()
        assert lab.selected_label == value
        assert lab.mode == "paint"
        assert np.allclose(_color_of(lab, value), default_color(value)[:3], atol=1e-3)
    w.brush_spin.setValue(9)
    assert lab.brush_size == 9


def test_rename_and_resize_classes(widget):
    _, w = widget
    w.n_classes_spin.setValue(4)
    assert len(w._class_buttons) == 5
    edit = w.class_grid.itemAtPosition(2, 1).widget()
    edit.setText("noyau")
    assert w._class_buttons[3].text() == "noyau"
    w.n_classes_spin.setValue(1)
    assert set(w._class_buttons) == {1, 2}


def test_spacing_from_layer_scale_and_feature_grid(widget):
    _, w = widget
    spec = w._build_spec()
    assert spec.spacing == (2.0, 1.0, 1.0)
    w._preset("none")
    w._feature_checks[("gaussian", 1.0)].setChecked(True)
    w.col_combo.setCurrentText("rayon 2")
    w._toggle_column(True)
    fams = {f for f, _ in w._build_spec().pairs}
    assert "gaussian" in fams and "erosion" in fams and "log" not in fams


def test_train_predict_view_full_and_save(widget, qtbot, tmp_path):
    viewer, w = widget
    w._ensure_labels_layer()
    lab = w._labels_layer
    lab.data[5, 26:38, 26:38] = 2
    lab.data[5, :6, :6] = 1
    lab.data[0, -6:, -6:] = 1
    viewer.dims.set_current_step(0, 5)
    w.unc_check.setChecked(True)
    w._start_job(train=True, predict="view")
    qtbot.waitUntil(lambda: not w._busy, timeout=60000)
    assert w.model is not None and w.model.is_fitted, w.status.text()
    seg = w._result_layer.data
    assert seg[5].any() and not seg[2].any()  # seule la coupe affichee est predite
    assert seg[5, 32, 32] == 2 and seg[5, 2, 2] == 1
    assert w._unc_layer is not None
    assert w.importance_list.count() > 0
    w._start_job(train=False, predict="full")
    qtbot.waitUntil(lambda: not w._busy, timeout=60000)
    assert w._result_layer.data[2].any()
    path = tmp_path / "m.joblib"
    w.model.save(str(path))
    from xwiclassifier.io import load_model
    w._adopt_model(load_model(str(path)))
    assert w._build_spec().pairs == w.model.feature_spec.pairs


def test_live_mode_retrains_after_paint(widget, qtbot):
    viewer, w = widget
    w._ensure_labels_layer()
    lab = w._labels_layer
    lab.data[5, 26:38, 26:38] = 2
    lab.data[5, :6, :6] = 1
    viewer.dims.set_current_step(0, 5)
    w.live_check.setChecked(True)
    w._on_paint()
    qtbot.waitUntil(lambda: w.model is not None and not w._busy, timeout=60000)
    assert w._result_layer is not None


def test_feature_explorer_on_current_slice(qtbot):
    from napari.components import ViewerModel

    from xwiclassifier.napari.widgets.feature_explorer import FeatureExplorerWidget

    viewer = ViewerModel()
    viewer.add_image(_volume(), name="vol")
    viewer.dims.set_current_step(0, 4)
    w = FeatureExplorerWidget(viewer)
    qtbot.addWidget(w)
    w._compute()
    lyr = viewer.layers["features (z=4)"]
    assert lyr.data.shape[1:] == (64, 64) and lyr.data.shape[0] == w.names.count()


def test_adopt_ilastik_model(widget):
    import os

    path = os.environ.get("XWI_ILASTIK_TEST_PROJECT")
    if not path:
        pytest.skip("XWI_ILASTIK_TEST_PROJECT non defini")
    from xwiclassifier.io.ilastik_project import load_ilastik_model

    _, w = widget
    w._adopt_model(load_ilastik_model(path))
    assert w.model.source["type"] == "ilastik"
    assert w._class_buttons[1].text() == w.model.class_label(1)


def test_adopt_ipsdk_model(widget, tmp_path):
    from conftest import make_fake_ipsdk_model, toy_2d
    from xwiclassifier.io.ipsdk_explorer import load_ipsdk_model

    img, lab = toy_2d()
    make_fake_ipsdk_model(str(tmp_path / "m"), [img], [lab], names=("Fond", "Fils"))
    _, w = widget
    w._adopt_model(load_ipsdk_model(str(tmp_path / "m")))
    assert w._class_buttons[2].text() == "Fils"
    assert w.importance_list.count() > 0


def test_nnunet_widget_builds_dataset(qtbot, tmp_path):
    pytest.importorskip("SimpleITK")
    from napari.components import ViewerModel

    from conftest import toy_3d
    from xwiclassifier.napari.widgets.nnunet_widget import NnUNetWidget

    viewer = ViewerModel()
    v, lab = toy_3d()
    viewer.add_image(v, name="vol", scale=(2.0, 1.0, 1.0))
    lyr = viewer.add_labels(lab, name="annotations")
    lyr.metadata["xwi_class_names"] = {1: "fond", 2: "objet"}
    w = NnUNetWidget(viewer)
    qtbot.addWidget(w)
    w.root_edit.setText(str(tmp_path / "ws"))
    w._open_workspace()
    ds = w._new_dataset(name="Rein test", classes={1: "fond", 2: "objet"})
    assert ds is not None and w.ds_combo.currentText() == ds.name
    w.img_combo.setCurrentText("vol")
    w.lab_combo.setCurrentText("annotations")
    w._add_case()
    w.rb_dense.setChecked(True)
    w.z0.setValue(5)
    w.z1.setValue(14)
    w._add_case()
    ds = w._dataset()
    assert ds.n_cases == 2 and [c["mode"] for c in ds.cases()] == ["sparse", "dense"]
    assert ds.cases()[0]["spacing"] == [2.0, 1.0, 1.0]
