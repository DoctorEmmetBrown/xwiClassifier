"""Demo : volume 3D synthetique, annotations eparses, entrainement et prediction.

    python examples/demo_3d.py            # ouvre napari s'il est installe
"""
import numpy as np
from skimage.data import binary_blobs

import xwiclassifier as xwi


def make_volume(shape=(40, 128, 128), seed=0):
    rng = np.random.default_rng(seed)
    blobs = binary_blobs(length=shape[1], n_dim=3, volume_fraction=0.3, rng=seed)[: shape[0]]
    img = blobs.astype(np.float32) + 0.4 * rng.standard_normal(blobs.shape).astype(np.float32)
    return img, blobs.astype(np.uint8) + 1  # 1 = fond, 2 = objets


def main():
    img, truth = make_volume()
    rng = np.random.default_rng(1)
    labels = np.zeros(img.shape, np.uint8)
    for c in (1, 2):
        idx = np.argwhere(truth == c)
        pick = idx[rng.choice(len(idx), 300, replace=False)]
        labels[tuple(pick.T)] = c

    model = xwi.PixelClassifier(class_names={1: "fond", 2: "objets"}).fit(img, labels)
    seg, unc = model.predict(img, return_uncertainty=True)
    print(f"Exactitude vs verite terrain : {(seg == truth).mean():.3f}")
    print("Contributions par famille :", {k: round(v, 3) for k, v in
                                          list(model.feature_importances(group='family').items())[:4]})
    try:
        import napari
        v = napari.Viewer()
        v.add_image(img, name="image")
        v.add_labels(seg, name="segmentation")
        v.add_image(unc, name="incertitude", colormap="magma", visible=False)
        napari.run()
    except Exception as exc:
        print(f"(napari non lance : {exc})")


if __name__ == "__main__":
    main()
