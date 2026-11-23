"""Volume synthetique : fond texture + blobs + fils fins, bruit, annotations eparses."""
import numpy as np
from scipy import ndimage as ndi


def make(shape=(48, 192, 192), seed=0):
    rng = np.random.default_rng(seed)
    truth = np.ones(shape, np.uint8)                       # 1 = fond
    zz, yy, xx = np.indices(shape)
    for _ in range(14):                                    # 2 = blobs
        c = rng.uniform([4, 10, 10], np.array(shape) - [4, 10, 10])
        r = rng.uniform(5, 11)
        truth[((zz - c[0]) * 2) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2 < r * r] = 2
    for _ in range(10):                                    # 3 = fils (tubes fins)
        p0 = rng.uniform([0, 0, 0], shape)
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        t = np.linspace(-300, 300, 4000)
        pts = np.round(p0[None] + t[:, None] * d[None] * [0.3, 1, 1]).astype(int)
        ok = np.all((pts >= 0) & (pts < shape), 1)
        m = np.zeros(shape, bool)
        m[tuple(pts[ok].T)] = True
        truth[ndi.binary_dilation(m, iterations=1) & (truth == 1)] = 3
    img = np.where(truth == 2, 0.7, 0.0) + np.where(truth == 3, 0.55, 0.0)
    img = img + 0.25 * ndi.gaussian_filter(rng.standard_normal(shape), 3) * 3   # fond texture
    img = ndi.gaussian_filter(img, 0.8) + 0.22 * rng.standard_normal(shape)     # flou + bruit
    lab = np.zeros(shape, np.uint8)
    for c, n in ((1, 600), (2, 400), (3, 250)):                                  # annotations eparses
        idx = np.argwhere(truth == c)
        pick = idx[rng.choice(len(idx), n, replace=False)]
        lab[tuple(pick.T)] = c
    return img.astype(np.float32), truth, lab


def scores(pred, truth):
    out = {"acc": float((pred == truth).mean())}
    for c, name in ((2, "blobs"), (3, "fils")):
        inter = np.sum((pred == c) & (truth == c))
        union = np.sum((pred == c) | (truth == c))
        out[f"IoU_{name}"] = float(inter / union)
    return out
