"""Extension nnU-Net : jeux de donnees (sans torch) et chaine complete (optionnelle)."""
import json
import os

import numpy as np
import pytest

from conftest import toy_3d
from xwiclassifier.nnunet.training import _numbers, training_status
from xwiclassifier.nnunet.workspace import NnUNetWorkspace, apply_volume_normalization, read_sitk

sitk = pytest.importorskip("SimpleITK")


def _ws(tmp_path):
    return NnUNetWorkspace(tmp_path / "ws")


def test_dataset_creation_and_label_mapping(tmp_path):
    ws = _ws(tmp_path)
    ds = ws.create_dataset("Rein vaisseaux", {1: "fond", 2: "vaisseaux", 3: "glomerules"})
    assert ds.id == 501 and ds.meta["labels"] == {"background": 0, "Vaisseaux": 1, "Glomerules": 2, "ignore": 3}
    v, lab = toy_3d()
    lab[12, 30, 30] = 3
    case = ds.add_case(v, lab, spacing=(2.0, 1.0, 1.0), margin=4)
    seg, sp = read_sitk(ds.folder / "labelsTr" / f"{case}.nii.gz")
    assert sp == (2.0, 1.0, 1.0)
    vals = set(np.unique(seg))
    assert vals == {0, 1, 2, 3}  # fond, vaisseaux, glomerules, ignore
    assert (seg == 3).mean() > 0.9  # annotations partielles : la majorite est ignoree
    img, _ = read_sitk(ds.folder / "imagesTr" / f"{case}_0000.nii.gz")
    assert img.shape == seg.shape and abs(float(img.mean())) < 2  # normalise


def test_dense_region_with_context_margin(tmp_path):
    ds = _ws(tmp_path).create_dataset("Blobs", {1: "fond", 2: "objet"})
    v, _ = toy_3d()
    truth = (v > 0.5).astype(np.uint8) + 1
    case = ds.add_case(v, truth, roi=(slice(5, 15), slice(10, 40), slice(10, 40)), mode="dense", margin=3)
    seg, _ = read_sitk(ds.folder / "labelsTr" / f"{case}.nii.gz")
    assert seg.shape == (16, 36, 36)
    assert (seg[3:-3, 3:-3, 3:-3] != 2).all()       # region validee : tout est etiquete
    assert (seg[0] == 2).all() and (seg[:, 0] == 2).all()  # marge de contexte : ignore
    assert ds.n_cases == 1 and ds.cases()[0]["mode"] == "dense"


def test_volume_normalization_global():
    rng = np.random.default_rng(0)
    img = (rng.random((40, 40)) * 1000).astype(np.float32)
    from xwiclassifier.nnunet.workspace import fit_volume_normalization
    p = fit_volume_normalization(img)
    a = apply_volume_normalization(img[:10], p)
    b = apply_volume_normalization(img, p)[:10]
    assert np.allclose(a, b)


def test_dataset_passes_nnunet_integrity_check(tmp_path):
    pytest.importorskip("nnunetv2")
    from nnunetv2.experiment_planning.verify_dataset_integrity import verify_dataset_integrity

    ds = _ws(tmp_path).create_dataset("Blobs", {1: "fond", 2: "objet"})
    for s in range(2):
        v, lab = toy_3d(seed=s)
        ds.add_case(v, lab, spacing=(2, 1, 1))
    verify_dataset_integrity(str(ds.folder), num_processes=1)


def test_status_parser(tmp_path):
    fold = tmp_path / "m" / "fold_all"
    fold.mkdir(parents=True)
    (fold / "training_log_2026_1_1_00_00_00.txt").write_text(
        "2026-01-01 00:00:01: Epoch 7 \n2026-01-01 00:00:02: train_loss -0.61 \n"
        "2026-01-01 00:00:02: Pseudo dice [np.float32(0.81), np.float32(0.7)] \n"
        "2026-01-01 00:00:02: Epoch time: 41.5 s \n"
        "2026-01-01 00:00:02: Yayy! New best EMA pseudo Dice: 0.7912 \n")
    st = training_status(tmp_path / "m")["folds"]["all"]
    assert st["epoch"] == 7 and st["pseudo_dice"] == [0.81, 0.7] and st["best_ema_dice"] == 0.7912
    assert st["train_loss"] == -0.61 and st["epoch_time_s"] == 41.5 and not st["finished"]
    assert _numbers("[0.5, 0.25]") == [0.5, 0.25]


def test_train_command_building(tmp_path, monkeypatch):
    """Repartition des folds sur 2 GPU sans lancer nnU-Net."""
    from xwiclassifier.nnunet import training

    ds = _ws(tmp_path).create_dataset("Blobs", {1: "fond", 2: "objet"})
    ds.xwi["ndim"] = 3
    calls = []

    class FakeProc:
        def poll(self):
            return 0

        def wait(self):
            return 0

    monkeypatch.setattr(training, "_spawn", lambda cmd, env, log: calls.append((cmd, env)) or FakeProc())
    job = training.train(ds, folds=(0, 1, 2, 3, 4), epochs=100, gpus=[0, 1], device="cuda")
    assert len(calls) == 2
    assert calls[0][1]["CUDA_VISIBLE_DEVICES"] == "0" and calls[0][0][-3:] == ["0", "2", "4"]
    assert calls[1][1]["CUDA_VISIBLE_DEVICES"] == "1" and calls[1][0][-2:] == ["1", "3"]
    assert calls[0][1]["XWI_NNUNET_EPOCHS"] == "100" and "trainers" in calls[0][1]["nnUNet_extTrainer"]
    calls.clear()
    training.train(ds, folds=("all",), epochs=50, gpus=[0, 1], device="cuda", ddp=True)
    assert len(calls) == 1 and calls[0][1]["CUDA_VISIBLE_DEVICES"] == "0,1" and calls[0][0][7] == "2"
    assert job.model_folder.name == "nnUNetTrainerXwi__nnUNetPlans__3d_fullres"


def test_load_model_detects_nnunet_folders(tmp_path):
    from xwiclassifier.io.models import _is_nnunet

    m = tmp_path / "res" / "Dataset501_X" / "nnUNetTrainerXwi__nnUNetPlans__3d_fullres"
    (m / "fold_all").mkdir(parents=True)
    (m / "plans.json").write_text("{}")
    (m / "dataset.json").write_text("{}")
    assert _is_nnunet(str(m)) and _is_nnunet(str(m.parent))
    ptr = tmp_path / "p.json"
    ptr.write_text(json.dumps({"format": "xwi-nnunet", "model_folder": str(m)}))
    assert _is_nnunet(str(ptr)) and not _is_nnunet(str(tmp_path))


@pytest.mark.skipif(not os.environ.get("XWI_NNUNET_E2E"), reason="XWI_NNUNET_E2E non defini (entrainement CPU ~3 min)")
def test_end_to_end_train_and_predict(tmp_path):
    from xwiclassifier import load_model
    from xwiclassifier.nnunet import plan_and_preprocess, train

    ds = _ws(tmp_path).create_dataset("Blobs", {1: "fond", 2: "objet"})
    for s in range(3):
        v, lab = toy_3d(seed=s)
        truth = (v > 0.5).astype(np.uint8) + 1
        ds.add_case(v * 100 + 500, lab, spacing=(2, 1, 1))
        ds.add_case(v * 100 + 500, truth, roi=(slice(4, 16), slice(8, 40), slice(8, 44)), mode="dense", margin=4)
    plan_and_preprocess(ds, num_processes=1)
    job = train(ds, epochs=2, iterations_per_epoch=10, val_iterations=2, device="cpu")
    assert job.wait() == 0
    m = load_model(str(job.model_folder), device="cpu", use_mirroring=False)
    v, _ = toy_3d(seed=9)
    seg = m.predict(v * 100 + 500, spacing=(2, 1, 1))
    truth = (v > 0.5) + 1
    assert seg.shape == v.shape and (seg == truth).mean() > 0.8
