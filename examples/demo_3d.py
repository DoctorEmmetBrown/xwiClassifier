"""Demo : volume 3D synthetique, annotation programmatique, classification.

Lancer :  python examples/demo_3d.py
(Ouvre napari si dispo, sinon fait juste la classif en mode headless.)
"""
import numpy as np
from skimage.data import binary_blobs

from napari_segforest.classifier.base import PixelClassifier
from napari_segforest.features.engine import FeatureSpec


def make_volume(shape=(40, 128, 128), seed=0):
    rng = np.random.default_rng(seed)
    blobs = binary_blobs(length=shape[1], n_dim=3, volume_fraction=0.3,
                         rng=seed).astype(np.float32)
    blobs = blobs[: shape[0]]
    img = blobs + 0.4 * rng.standard_normal(blobs.shape).astype(np.float32)
    return img, blobs.astype(np.uint8)


def main():
    img, truth = make_volume()
    # Annotation eparse : quelques pixels de chaque classe
    labels = np.zeros(img.shape, dtype=np.uint8)
    fg = np.argwhere(truth == 1)
    bg = np.argwhere(truth == 0)
    rng = np.random.default_rng(1)
    for z, y, x in fg[rng.choice(len(fg), 300, replace=False)]:
        labels[z, y, x] = 1
    for z, y, x in bg[rng.choice(len(bg), 300, replace=False)]:
        labels[z, y, x] = 2

    clf = PixelClassifier(kind="rf",
                          feature_spec=FeatureSpec(sigmas=(0, 1, 2, 4)))
    clf.fit(img, labels)
    pred = clf.predict(img)
    acc = (((pred == 1) == (truth == 1)).mean())
    print(f"Accuracy vs verite terrain : {acc:.3f}")
    print("Top features :", list(clf.feature_importances())[:5])

    try:
        import napari

        v = napari.Viewer()
        v.add_image(img, name="raw")
        v.add_labels(pred.astype(np.uint8), name="prediction")
        v.add_labels(truth, name="truth", visible=False)
        napari.run()
    except Exception as e:
        print(f"(napari non lance : {e})")


if __name__ == "__main__":
    main()
