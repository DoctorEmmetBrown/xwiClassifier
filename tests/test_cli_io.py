import json
import subprocess
import sys

import numpy as np

from conftest import make_fake_ilp, toy_3d
from xwiclassifier.cli import main
from xwiclassifier.io import read_image, write_image
from xwiclassifier.io.ipsdk import distill_ipsdk_model, inspect_binary


def test_io_formats(tmp_path, data2d):
    img, _ = data2d
    for ext in (".tif", ".npy", ".h5", ".npz"):
        p = tmp_path / f"a{ext}"
        if ext == ".npz":
            np.savez(p, x=img)
        else:
            write_image(p, img)
        assert np.array_equal(read_image(p), img)
    write_image(str(tmp_path / "b.h5") + "/grp/vol", img)
    assert np.array_equal(read_image(str(tmp_path / "b.h5") + "/grp/vol"), img)


def test_cli_train_predict_convert(tmp_path, capsys):
    v, lab = toy_3d()
    write_image(tmp_path / "v.tif", v)
    write_image(tmp_path / "l.tif", lab)
    m = str(tmp_path / "m.joblib")
    assert main(["train", "--image", str(tmp_path / "v.tif"), "--labels", str(tmp_path / "l.tif"),
                 "-o", m, "--spacing", "2", "1", "1", "--class-names", "1=fond,2=cube",
                 "--n-estimators", "30"]) == 0
    out = tmp_path / "out"
    assert main(["predict", m, str(tmp_path / "v.tif"), "-o", str(out), "--proba", "--uncertainty"]) == 0
    seg = read_image(out / "v_seg.tif")
    assert seg.shape == v.shape and seg[10, 24, 27] == 2
    assert read_image(out / "v_proba.tif").shape == (2,) + v.shape
    assert main(["info", m]) == 0
    p = str(tmp_path / "p.ilp")
    make_fake_ilp(p, v, lab)
    assert main(["convert", p, "-o", str(tmp_path / "c.joblib")]) == 0
    assert main(["ilastik-export", p, "-o", str(tmp_path / "exp")]) == 0
    assert np.array_equal(read_image(tmp_path / "exp" / "img_labels.tif"), lab)
    capsys.readouterr()
    assert main(["inspect", p]) == 0
    assert json.loads(capsys.readouterr().out)["foret"]["arbres"] == 12


def test_cli_entry_point_module(tmp_path):
    r = subprocess.run([sys.executable, "-m", "xwiclassifier", "features"], capture_output=True, text=True)
    assert r.returncode == 0 and "white_tophat" in r.stdout


def test_ipsdk_distillation_with_simulated_predictor(data2d):
    from scipy import ndimage as ndi

    img = ndi.gaussian_filter(np.random.default_rng(0).random((120, 140)), 5).astype(np.float32)

    def fake(im):
        sm = ndi.gaussian_filter(im, 2)
        return (sm > np.median(sm)).astype(np.uint8), np.abs(sm - np.median(sm))

    m = distill_ipsdk_model("faux.bin", img, predictor=fake, samples_per_class=2000)
    assert m.source["holdout_agreement"] > 0.95


def test_inspect_binary(tmp_path):
    p = tmp_path / "m.bin"
    p.write_bytes(b"\x16\x00serialization::archive\x00RandomForestModel\x00" + bytes(range(256)))
    info = inspect_binary(str(p))
    assert info["format"].startswith("archive boost") and "RandomForestModel" in info["keywords"]
