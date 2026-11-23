"""Benchmark xwiClassifier sur un volume synthetique (blobs + fils + fond texture).

    python benchmarks/bench.py                 # v0.2 (cette version)
    python benchmarks/bench.py --v01 /chemin/vers/src_v01   # compare avec le prototype v0.1

Le code v0.1 s'extrait de l'historique git :
    mkdir /tmp/v01 && git archive 4aea611 src/napari_segforest | tar -x -C /tmp/v01
    python benchmarks/bench.py --v01 /tmp/v01/src
"""
import argparse
import json
import os
import resource
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from synth import make, scores  # noqa: E402


def run(which, shape, v01_src=None):
    img, truth, lab = make(shape=shape, seed=0)
    imgB, truthB, _ = make(shape=shape, seed=1)
    imgB = imgB.copy()
    imgB.flat[np.random.default_rng(5).choice(imgB.size, 30, replace=False)] = 40.0  # zingers
    t0 = time.perf_counter()
    if which == "v01":
        sys.path.insert(0, v01_src)
        os.environ["SEGFOREST_FORCE_CPU"] = "1"
        from napari_segforest.classifier.base import PixelClassifier
        from napari_segforest.features.engine import FeatureSpec
        clf = PixelClassifier(kind="rf", feature_spec=FeatureSpec())
        clf.fit(img, lab)
        t1 = time.perf_counter()
        pred = clf.predict(img)
        t2 = time.perf_counter()
        clf._image_id = None
        predB = clf.predict(imgB)
    else:
        from xwiclassifier import FeatureSpec, PixelClassifier
        clf = PixelClassifier(FeatureSpec(spacing=(2.0, 1.0, 1.0))).fit(img, lab)
        t1 = time.perf_counter()
        pred = clf.predict(img)
        t2 = time.perf_counter()
        predB = clf.predict(imgB)
    return {"version": which, "train_s": round(t1 - t0, 2), "predict_s": round(t2 - t1, 2),
            "peak_RAM_GB": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2),
            "meme_image": {k: round(v, 3) for k, v in scores(pred, truth).items()},
            "nouvelle_image_avec_zingers": {k: round(v, 3) for k, v in scores(predB, truthB).items()}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--v01", default=None, help="dossier src du prototype v0.1")
    ap.add_argument("--shape", nargs=3, type=int, default=(48, 192, 192))
    ap.add_argument("--which", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.which:
        print(json.dumps(run(a.which, tuple(a.shape), a.v01)))
        sys.exit(0)
    for which in (["v01"] if a.v01 else []) + ["v02"]:
        cmd = [sys.executable, __file__, "--which", which, "--shape", *map(str, a.shape)]
        if a.v01:
            cmd += ["--v01", a.v01]
        out = subprocess.run(cmd, capture_output=True, text=True)
        print(out.stdout.strip().splitlines()[-1] if out.stdout.strip() else out.stderr[-500:])
