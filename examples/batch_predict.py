"""Segmentation par lots sans napari, avec un modele xwi (.joblib) ou ilastik (.ilp).

    python examples/batch_predict.py modele.joblib "donnees/*.tif" resultats/
"""
import glob
import os
import sys

import xwiclassifier as xwi


def main(model_path, pattern, outdir):
    model = xwi.load_model(model_path)
    print(model.describe())
    os.makedirs(outdir, exist_ok=True)
    for path in sorted(glob.glob(pattern)):
        vol = xwi.read_image(path)
        seg, unc = model.predict(vol, return_uncertainty=True, smooth=1.0)
        stem = os.path.splitext(os.path.basename(path))[0]
        xwi.write_image(os.path.join(outdir, stem + "_seg.tif"), seg)
        xwi.write_image(os.path.join(outdir, stem + "_incertitude.tif"), unc)
        print(f"{path} -> {stem}_seg.tif ({vol.shape})")


if __name__ == "__main__":
    main(*sys.argv[1:4])
