"""Outils de test : donnees synthetiques et fabrication de projets ilastik factices."""
import json

import numpy as np
import pytest


def toy_2d(seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((96, 112), np.float32)
    img[24:72, 28:84] = 1.0
    img += 0.25 * rng.standard_normal(img.shape).astype(np.float32)
    lab = np.zeros(img.shape, np.uint8)
    lab[38:58, 44:68] = 2
    lab[:10, :10] = 1
    lab[-10:, -10:] = 1
    return img, lab


def toy_3d(seed=0):
    rng = np.random.default_rng(seed)
    v = np.zeros((20, 48, 52), np.float32)
    v[5:15, 12:36, 14:40] = 1.0
    v += 0.25 * rng.standard_normal(v.shape).astype(np.float32)
    lab = np.zeros(v.shape, np.uint8)
    lab[9:11, 20:28, 22:32] = 2
    lab[:2, :6, :6] = 1
    lab[-2:, -6:, -6:] = 1
    return v, lab


@pytest.fixture
def data2d():
    return toy_2d()


@pytest.fixture
def data3d():
    return toy_3d()


# --------------------------------------------------------------------------
# Ecriture de forets au format vigra a partir d'arbres scikit-learn
# --------------------------------------------------------------------------
def _le_to_strict(t):
    """Seuil vigra t_v tel que (x < t_v) <=> (x <= t) pour x float32."""
    t32 = np.float32(t)
    if float(t32) > t:
        t32 = np.nextafter(t32, np.float32(-np.inf))
    return float(np.nextafter(t32, np.float32(np.inf)))


def sklearn_tree_to_vigra(tree, n_features, n_classes):
    t = tree.tree_
    order, stack = [], [0]
    while stack:
        k = stack.pop()
        order.append(k)
        if t.children_left[k] >= 0:
            stack.append(t.children_right[k])
            stack.append(t.children_left[k])
    pos, cur = {}, 2
    for k in order:
        pos[k] = cur
        cur += 5 if t.children_left[k] >= 0 else 2
    topo = np.zeros(cur, np.int32)
    topo[0], topo[1] = n_features, n_classes
    params = []
    for k in order:
        i = pos[k]
        if t.children_left[k] >= 0:
            topo[i:i + 5] = [0, len(params), pos[t.children_left[k]], pos[t.children_right[k]], t.feature[k]]
            params += [1.0, _le_to_strict(t.threshold[k])]
        else:
            v = t.value[k][0]
            topo[i:i + 2] = [0x40000000, len(params)]
            params += [float(t.n_node_samples[k])] + list(v / v.sum())
    return topo, np.asarray(params, np.float64)


def write_vigra_forest(group, trees, n_features, n_classes, labels):
    ext = group.create_group("_ext_param")
    ext["labels"] = np.asarray(labels, np.uint32)
    ext["column_count_"] = np.array([float(n_features)])
    ext["class_count_"] = np.array([float(n_classes)])
    opt = group.create_group("_options")
    opt["predict_weighted_"] = np.array([0.0])
    opt["tree_count_"] = np.array([float(len(trees))])
    for i, tr in enumerate(trees):
        topo, par = sklearn_tree_to_vigra(tr, n_features, n_classes)
        g = group.create_group(f"Tree_{i:02d}")
        g["topology"] = topo
        g["parameters"] = par


def _tags(axes):
    return json.dumps({"axes": [{"key": a, "typeFlags": 2, "resolution": 0, "description": ""}
                                for a in axes]})


def make_fake_ilp(path, image, labels, train_forest=True, internal=True, raw_file=None):
    """Projet ilastik minimal (structure de lazyflow/ilastik 1.4) pour une image 2D ou 3D."""
    import h5py
    from sklearn.ensemble import RandomForestClassifier

    from xwiclassifier.classifier.sampler import TrainingSampler
    from xwiclassifier.features.ilastik import IlastikFeatures

    nd = image.ndim
    axes = "yxc" if nd == 2 else "zyxc"
    scales = [0.3, 0.7, 1.0, 1.6, 3.5]
    feats = IlastikFeatures.all_features(scales, compute_in_2d=[nd == 2] * len(scales))
    with h5py.File(path, "w") as f:
        f["ilastikVersion"] = b"1.4.0"
        f["workflowName"] = b"Pixel Classification"
        fs = f.create_group("FeatureSelections")
        fs["FeatureIds"] = [s.encode() for s in feats.feature_ids]
        fs["Scales"] = np.asarray(scales)
        fs["SelectionMatrix"] = feats.matrix
        fs["ComputeIn2d"] = np.asarray(feats.compute_in_2d)
        info = f.create_group("Input Data/infos/lane0000/Raw Data")
        info["axistags"] = _tags(axes).encode()
        info["shape"] = np.asarray(image.shape + (1,))
        info["nickname"] = b"img"
        if internal:
            ds = f.create_dataset("Input Data/local_data/abc", data=image[..., None])
            ds.attrs["axistags"] = _tags(axes)
            info["location"] = b"ProjectInternal"
            info["filePath"] = b"/Input Data/local_data/abc"
        else:
            info["location"] = b"FileSystem"
            info["filePath"] = raw_file.encode()
        pc = f.create_group("PixelClassification")
        pc["LabelNames"] = [b"fond", b"objet"]
        pc["LabelColors"] = np.array([[255, 0, 0], [0, 255, 0]])
        ls = pc.create_group("LabelSets/labels000")
        # deux blocs disjoints (comme ilastik : bornes a la boite englobante)
        half = labels.shape[0] // 2
        for b, (lo, hi) in enumerate(((0, half), (half, labels.shape[0]))):
            sub = labels[lo:hi]
            if not sub.any():
                continue
            idx = np.argwhere(sub > 0)
            mn, mx = idx.min(0), idx.max(0) + 1
            sl = tuple(slice(a, c) for a, c in zip(mn, mx))
            blk = sub[sl][..., None]
            d = ls.create_dataset(f"block{b:04d}", data=blk)
            starts = [lo + mn[0]] + list(mn[1:])
            stops = [lo + mx[0]] + list(mx[1:])
            d.attrs["blockSlice"] = "[" + ",".join(f"{a}:{c}" for a, c in zip(starts, stops)) + ",0:1]"
            d.attrs["axistags"] = _tags(axes)
        if train_forest:
            X, y = TrainingSampler(feats, cache_bytes=0).samples(image, labels)
            rf = RandomForestClassifier(n_estimators=12, random_state=0).fit(X, y)
            cf = pc.create_group("ClassifierForests")
            for k in range(2):
                write_vigra_forest(cf.create_group(f"Forest{k:04d}"), rf.estimators_[6 * k: 6 * (k + 1)],
                                   X.shape[1], 2, [1, 2])
            cf["known_labels"] = np.array([1, 2], np.uint32)
            cf["feature_names"] = [n.encode() for n in feats.channel_names(nd)]
            return rf, feats
    return None, feats


# --------------------------------------------------------------------------
# Dossiers de modele IPSDK Explorer factices (format de Settings.mho 4.0)
# --------------------------------------------------------------------------
def _ascii(text):
    return ".".join(str(ord(c)) for c in text)


def write_ipsdk_settings(folder, sizes, flags, dimension="2D", multires=(1,), classes=(), images=()):
    import os
    import xml.etree.ElementTree as ET

    root = ET.Element("IPSDK")
    ae = ET.SubElement(root, "AllElements")
    for nm in images:
        im = ET.SubElement(ae, "Image")
        ET.SubElement(im, "Name").text = nm
    lc = ET.SubElement(root, "LabelClasses")
    ET.SubElement(lc, "NumberLabels").text = str(len(classes))
    for i, (name, color, value) in enumerate(classes):
        el = ET.SubElement(lc, f"Label_{i}")
        ET.SubElement(el, "Color").text = ",".join(map(str, color))
        n = ET.SubElement(el, "Name")
        n.text = _ascii(name)
        n.set("ASCII", "True")
        ET.SubElement(el, "Value").text = str(value)
    fe = ET.SubElement(root, "Features")
    ET.SubElement(fe, "All_Sizes").text = ",".join(map(str, sizes))
    for name in ("Gaussian", "Mean", "Laplacian Difference", "High Pass", "Variance",
                 "Erosion", "Dilation", "Opening", "Closing"):
        ET.SubElement(fe, name.replace(" ", "_")).text = ",".join(str(v) for v in flags.get(name, [False] * len(sizes)))
    ET.SubElement(fe, "Dimension").text = dimension
    ET.SubElement(fe, "MultiRes").text = ",".join(map(str, multires))
    os.makedirs(folder, exist_ok=True)
    ET.ElementTree(root).write(os.path.join(folder, "Settings.mho"))


class _OldTree:
    """Arbre picke au format scikit-learn < 1.3 (sans 'missing_go_to_left')."""

    def __init__(self, tree):
        self.t = tree

    def __reduce__(self):
        from sklearn.tree._tree import Tree

        st = self.t.__getstate__()
        nodes = st["nodes"]
        names = [n for n in nodes.dtype.names if n != "missing_go_to_left"]
        old = np.zeros(len(nodes), dtype=[(n, nodes.dtype[n]) for n in names])
        for n in names:
            old[n] = nodes[n]
        state = {"max_depth": st["max_depth"], "node_count": st["node_count"], "nodes": old,
                 "values": st["values"]}
        return (Tree, (self.t.n_features, np.asarray(self.t.n_classes), self.t.n_outputs), state)


def make_fake_ipsdk_model(folder, images, labels, sizes=(1, 2, 3), flags=None, dimension="2D",
                          multires=(1,), old_sklearn=False, gaussian_normalize=True, zoom_mode="corner",
                          names=("Fond", "Objet"), n_trees=20):
    """Dossier IPSDK Explorer complet, entraine comme le fait l'Explorer (classes 0..n-1)."""
    import copy
    import os

    import joblib
    from sklearn.ensemble import RandomForestClassifier

    from xwiclassifier.features.ipsdk import IPSDKFeatures
    from xwiclassifier.io import write_image

    flags = flags or {"Gaussian": [True, True, True], "Mean": [False, True, False],
                      "Laplacian Difference": [True, False, True], "High Pass": [False, True, False],
                      "Variance": [True, False, False], "Erosion": [False, True, False],
                      "Dilation": [True, False, False], "Opening": [False, False, True],
                      "Closing": [False, True, False]}
    nm_imgs = [f"img{i}" for i in range(len(images))]
    write_ipsdk_settings(folder, sizes, flags, dimension, multires,
                         [(n, (255, 40 * i, 0), i) for i, n in enumerate(names)], nm_imgs)
    feats = IPSDKFeatures.from_settings(list(sizes), flags, multires, dimension,
                                        gaussian_normalize=gaussian_normalize, zoom_mode=zoom_mode)
    Xs, ys = [], []
    for nm, img, lab in zip(nm_imgs, images, labels):
        write_image(os.path.join(folder, nm, "Image.tif"), img)
        write_image(os.path.join(folder, nm, "ImageDraw.tif"), lab.astype(np.uint16))
        f = feats.compute(img)
        Xs.append(f[lab > 0])
        ys.append(lab[lab > 0].astype(int) - 1)  # IPSDK : numLabel = valeur dessinee - 1
    rf = RandomForestClassifier(n_estimators=n_trees, random_state=0).fit(np.concatenate(Xs), np.concatenate(ys))
    to_dump = rf
    if old_sklearn:
        to_dump = copy.copy(rf)
        ests = []
        for e in rf.estimators_:
            e2 = copy.copy(e)
            e2.__dict__.pop("monotonic_cst", None)
            e2.__dict__["tree_"] = _OldTree(e.tree_)
            e2.__dict__["n_features_"] = e.n_features_in_
            ests.append(e2)
        to_dump.__dict__["estimators_"] = ests
        to_dump.__dict__["base_estimator_"] = to_dump.__dict__.pop("estimator_")
        to_dump.__dict__["n_features_"] = rf.n_features_in_
        to_dump.__dict__["_sklearn_version"] = "0.24.1"
    joblib.dump(to_dump, os.path.join(folder, "Model"))
    open(os.path.join(folder, "ModelIPSDK.bin"), "wb").write(b"\x00IPSDK-binaire-factice")
    return rf, feats
