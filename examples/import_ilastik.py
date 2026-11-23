"""Recuperer un projet ilastik : classifieur tel quel, puis re-entrainement avec les features xwi.

    python examples/import_ilastik.py projet.ilp
"""
import sys

import numpy as np

import xwiclassifier as xwi


def main(ilp):
    exact = xwi.load_ilastik_model(ilp, mode="forest")
    print(exact.describe())
    ann = xwi.ilastik_annotations(ilp)
    seg = exact.predict(ann["image"], channel_axis=ann["channel_axis"])
    mask = ann["labels"] > 0
    print(f"Accord avec les annotations ilastik : {100 * (seg[mask] == ann['labels'][mask]).mean():.2f}%")
    exact.save("depuis_ilastik.joblib")

    retrained = xwi.load_ilastik_model(ilp, mode="retrain")  # features xwi, plus riches
    seg2 = retrained.predict(ann["image"], channel_axis=ann["channel_axis"])
    print(f"Accord entre forets ilastik et xwi : {100 * np.mean(seg == seg2):.2f}%")
    retrained.save("reentraine_xwi.joblib")


if __name__ == "__main__":
    main(sys.argv[1])
