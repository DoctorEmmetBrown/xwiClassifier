"""Lecture / ecriture d'images 2D/3D (TIFF, HDF5, NumPy, Zarr, PNG/JPEG)."""
from __future__ import annotations

import os

import numpy as np

_H5 = (".h5", ".hdf5", ".hdf", ".he5")


def _split_inner(path: str):
    """Separe 'fichier.h5/groupe/dataset' en (fichier, chemin interne)."""
    low = path.lower()
    for ext in _H5 + (".zarr", ".n5"):
        i = low.find(ext + "/")
        if i >= 0:
            return path[: i + len(ext)], path[i + len(ext) + 1:]
    return path, None


def read_image(path: str, key: str | None = None) -> np.ndarray:
    """Lit une image. Pour HDF5/Zarr, ``key`` (ou 'fichier.h5/dataset') choisit le jeu."""
    path = os.fspath(path)
    path, inner = _split_inner(path)
    key = key or inner
    low = path.lower()
    if low.endswith((".tif", ".tiff")):
        import tifffile
        return tifffile.imread(path)
    if low.endswith(".npy"):
        return np.load(path, allow_pickle=False)
    if low.endswith(".npz"):
        with np.load(path, allow_pickle=False) as z:
            return z[key or list(z.keys())[0]]
    if low.endswith(_H5):
        import h5py
        with h5py.File(path, "r") as f:
            if key is None:
                found = []
                f.visititems(lambda n, o: found.append(n) if isinstance(o, h5py.Dataset) else None)
                if not found:
                    raise ValueError(f"Aucun dataset dans {path}")
                key = max(found, key=lambda n: f[n].size)
            return f[key][()]
    if low.endswith((".zarr", ".n5")):
        import zarr
        z = zarr.open(path, mode="r")
        return np.asarray(z[key] if key else z)
    from skimage import io as skio
    return np.asarray(skio.imread(path))


def write_image(path: str, data: np.ndarray, key: str = "data", compress: bool = True) -> None:
    """Ecrit une image ; le format est deduit de l'extension."""
    path = os.fspath(path)
    path, inner = _split_inner(path)
    key = inner or key
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    low = path.lower()
    data = np.asarray(data)
    if low.endswith((".tif", ".tiff")):
        import tifffile
        tifffile.imwrite(path, data, bigtiff=data.nbytes > 2 ** 31,
                         compression="zlib" if compress else None)
    elif low.endswith(".npy"):
        np.save(path, data)
    elif low.endswith(_H5):
        import h5py
        with h5py.File(path, "a") as f:
            if key in f:
                del f[key]
            f.create_dataset(key, data=data, compression="gzip" if compress else None,
                             chunks=True if data.ndim else None)
    elif low.endswith(".zarr"):
        import zarr
        zarr.save_array(path, data)
    else:
        from skimage import io as skio
        skio.imsave(path, data, check_contrast=False)
