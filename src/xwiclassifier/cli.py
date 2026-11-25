"""Ligne de commande ``xwi`` : entrainer, predire, convertir et inspecter des modeles."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np


def _progress(label: str):
    t0 = time.perf_counter()

    def cb(i, n):
        sys.stderr.write(f"\r  {label} : bloc {i}/{n} ({time.perf_counter() - t0:.1f}s)")
        if i == n:
            sys.stderr.write("\n")
        sys.stderr.flush()
    return cb


def _expand(paths):
    out = []
    for p in paths:
        hits = sorted(glob.glob(p))
        out.extend(hits if hits else [p])
    return out


def _parse_names(s: str | None) -> dict:
    if not s:
        return {}
    res = {}
    for part in s.split(","):
        k, v = part.split("=", 1)
        res[int(k)] = v.strip()
    return res


def _load(args):
    from .io import load_model

    kw = {}
    if args.model.lower().endswith(".ilp"):
        kw = {"mode": getattr(args, "ilastik_mode", "forest")}
        if getattr(args, "raw", None):
            kw["raw_paths"] = args.raw
    from .io.models import _is_nnunet
    if _is_nnunet(args.model) and getattr(args, "no_tta", False):
        kw["use_mirroring"] = False
    model = load_model(args.model, **kw)
    _report_ipsdk(model)
    return model


def _report_ipsdk(model):
    src = model.source or {}
    if src.get("type") == "ipsdk-explorer" and src.get("annotation_agreement") is not None:
        print(f"Modele IPSDK Explorer : accord avec ses annotations {100 * src['annotation_agreement']:.2f}% "
              f"(confiance moyenne {100 * src.get('annotation_confidence', 0):.2f}%)")


# --------------------------------------------------------------------------
def cmd_predict(args) -> int:
    from .classifier.model import labels_from_proba, uncertainty
    from .io import read_image, write_image

    model = _load(args)
    inputs = _expand(args.inputs)
    multi = len(inputs) > 1 or os.path.isdir(args.output) or not os.path.splitext(args.output)[1]
    if multi:
        os.makedirs(args.output, exist_ok=True)
    block = tuple([args.block] * 3) if args.block else None
    for path in inputs:
        img = read_image(path, key=args.key)
        print(f"{path} : {img.shape} {img.dtype}")
        nd = img.ndim - (1 if args.channel_axis is not None else 0)
        kw = {"spacing": tuple(args.spacing)} if args.spacing and getattr(model, "kind", "") == "nnunet" else {}
        proba = model.predict_proba(img, channel_axis=args.channel_axis,
                                    block_shape=block[:nd] if block else None,
                                    smooth=args.smooth, progress=_progress("prediction"), **kw)
        seg = labels_from_proba(proba, model.classes_)
        if args.ipsdk_values:
            from .io.ipsdk_explorer import ipsdk_output_values
            seg = ipsdk_output_values(model, seg)
        stem = os.path.splitext(os.path.basename(path))[0]
        ext = "." + args.format
        seg_path = os.path.join(args.output, stem + "_seg" + ext) if multi else args.output
        write_image(seg_path, seg)
        lab = labels_from_proba(proba, model.classes_) if args.ipsdk_values else seg
        print(f"  -> {seg_path}  " + ", ".join(
            f"{model.class_label(c)}: {100 * (lab == c).mean():.1f}%" for c in model.classes_))
        base = os.path.splitext(seg_path)[0].removesuffix("_seg")
        if args.proba:
            write_image(base + "_proba" + ext, np.moveaxis(proba, -1, 0))
        if args.uncertainty:
            write_image(base + "_uncertainty" + ext, uncertainty(proba))
    return 0


def cmd_train(args) -> int:
    from .classifier import PixelClassifier
    from .features import DEFAULT_FAMILIES, DEFAULT_RADII, DEFAULT_SCALES, FeatureSpec
    from .io import read_image

    imgs, labs = _expand(args.image), _expand(args.labels)
    if len(imgs) != len(labs):
        print("Il faut autant de --image que de --labels.", file=sys.stderr)
        return 2
    spec = FeatureSpec.from_grid(
        families=args.families or DEFAULT_FAMILIES, scales=args.scales or DEFAULT_SCALES,
        radii=args.radii or DEFAULT_RADII, spacing=tuple(args.spacing) if args.spacing else None,
        compute_in_2d=args.in2d, normalize=args.normalize)
    params = {"n_estimators": args.n_estimators} if args.n_estimators else {}
    model = PixelClassifier(spec, kind=args.classifier, params=params,
                            class_names=_parse_names(args.class_names))
    images = [read_image(p) for p in imgs]
    labels = [read_image(p).astype(np.int64) for p in labs]
    t0 = time.perf_counter()
    model.fit(images, labels, channel_axis=args.channel_axis, progress=_progress("features"))
    print(f"Entraine en {time.perf_counter() - t0:.1f}s : {model.training_info}")
    model.save(args.output)
    print(f"Modele sauve : {args.output}")
    _print_importances(model, 8)
    return 0


def cmd_convert(args) -> int:
    from .io import load_model
    from .features import FeatureSpec

    kw = {}
    is_dir = os.path.isdir(args.model) or args.model.lower().endswith(".mho")
    if args.model.lower().endswith(".ilp") or is_dir:
        kw["mode"] = args.mode
        if args.mode == "retrain":
            kw["features"] = args.features if args.features in ("ilastik", "ipsdk", "equivalent") else FeatureSpec()
        if args.raw and not is_dir:
            kw["raw_paths"] = args.raw
    model = load_model(args.model, **kw)
    _report_ipsdk(model)
    model.save(args.output)
    print(model.describe())
    print(f"Modele converti : {args.output}")
    return 0


def cmd_info(args) -> int:
    model = _load(args)
    print(model.describe())
    _print_importances(model, args.top)
    return 0


def _print_importances(model, top: int) -> None:
    imp = model.feature_importances(group="family")
    if imp:
        print("Contribution par famille :")
        for k, v in list(imp.items())[:top]:
            print(f"  {100 * v:5.1f}%  {k}")


def cmd_inspect(args) -> int:
    low = args.path.lower()
    from .io.ipsdk_explorer import is_ipsdk_model_folder
    if (os.path.isdir(args.path) or low.endswith(".mho")) and is_ipsdk_model_folder(args.path):
        from .io.ipsdk_explorer import _resolve_folder, ipsdk_annotations, read_settings
        folder = _resolve_folder(args.path)
        st = read_settings(folder)
        anns = ipsdk_annotations(folder)
        info = {"type": "modele IPSDK Explorer", "dossier": folder, "dimension": st["dimension"],
                "tailles": st["sizes"], "resolutions": st["resolutions"],
                "filtres": {n: [s for s, f in zip(st["sizes"], st["flags"][n]) if f] for n in st["order"]},
                "classes": st["classes"],
                "foret_sklearn": os.path.exists(os.path.join(folder, "Model")),
                "images": [{"nom": a["name"], "forme": list(a["image"].shape),
                            "pixels_annotes": 0 if a["labels"] is None else int((a["labels"] > 0).sum())}
                           for a in anns]}
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return 0
    if low.endswith(".ilp"):
        from .io.ilastik_project import IlastikProject

        with IlastikProject(args.path) as p:
            fo = p.forests()
            info = {"ilastik": p.ilastik_version, "workflow": p.workflow,
                    "labels": p.label_names(), "features": p.feature_selection() is not None,
                    "lanes": [vars(li) for li in p.lanes()],
                    "foret": None if fo is None else {"arbres": len(fo["trees"]),
                                                      "n_features": fo["n_features"]}}
        print(json.dumps(info, indent=2, ensure_ascii=False, default=str))
        return 0
    if low.endswith((".joblib", ".pkl")):
        args.model = args.path
        return cmd_info(argparse.Namespace(model=args.path, top=10))
    from .io.ipsdk import inspect_binary

    print(json.dumps(inspect_binary(args.path), indent=2, ensure_ascii=False))
    return 0


def cmd_ilastik_export(args) -> int:
    from .io import write_image
    from .io.ilastik_project import IlastikProject, _squeeze_for_model

    os.makedirs(args.output, exist_ok=True)
    with IlastikProject(args.project) as p:
        for k, li in enumerate(p.lanes()):
            raw, lab, axes = p.annotations(k, args.raw[k] if args.raw else None)
            raw, lab, _ = _squeeze_for_model(raw, axes, lab)
            stem = li.nickname or f"lane{k}"
            write_image(os.path.join(args.output, f"{stem}_image.tif"), raw)
            if lab is not None:
                write_image(os.path.join(args.output, f"{stem}_labels.tif"), lab)
            print(f"lane {k} ({stem}) : image {raw.shape}, annotations "
                  f"{'aucune' if lab is None else int((lab > 0).sum())} pixels")
        names = p.label_names()
    print("Classes :", ", ".join(f"{i + 1}={n}" for i, n in enumerate(names)))
    return 0


def cmd_ipsdk_export(args) -> int:
    from .io import write_image
    from .io.ipsdk_explorer import ipsdk_annotations, read_settings, _resolve_folder

    os.makedirs(args.output, exist_ok=True)
    for a in ipsdk_annotations(args.model):
        write_image(os.path.join(args.output, f"{a['name']}_image.tif"), a["image"])
        if a["labels"] is not None:
            write_image(os.path.join(args.output, f"{a['name']}_labels.tif"), a["labels"])
        print(f"{a['name']} : image {a['image'].shape}, annotations "
              f"{'aucune' if a['labels'] is None else int((a['labels'] > 0).sum())} pixels")
    st = read_settings(_resolve_folder(args.model))
    print("Classes :", ", ".join(f"{c['index'] + 1}={c['name']}" for c in st["classes"]))
    return 0


def cmd_ipsdk_distill(args) -> int:
    from .io import read_image
    from .io.ipsdk import distill_ipsdk_model

    images = [read_image(p) for p in _expand(args.image)]
    model = distill_ipsdk_model(args.model, images, kind=args.classifier,
                                samples_per_class=args.samples, channel_axis=args.channel_axis)
    model.save(args.output)
    print(model.describe())
    print(f"Accord avec IPSDK (pixels mis de cote) : {100 * model.source['holdout_agreement']:.2f}%")
    return 0


def cmd_features(args) -> int:
    from .features import FAMILIES

    for name, f in FAMILIES.items():
        unit = "rayon" if f.kind == "morpho" else ("-" if f.kind == "raw" else "sigma")
        print(f"  {name:16s} [{f.kind:6s}, {unit:5s}, {f.channels} canal/canaux]  {f.description}")
    return 0


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="xwi", description=__doc__)
    ap.add_argument("--device", choices=("auto", "cpu", "gpu"), default=None,
                    help="peripherique de calcul des filtres (defaut : auto)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("predict", help="segmenter une ou plusieurs images avec un modele")
    p.add_argument("model", help="modele .joblib (xwi), projet ilastik .ilp, dossier IPSDK Explorer ou modele nnU-Net")
    p.add_argument("inputs", nargs="+", help="images (TIFF, HDF5, npy... ; motifs glob acceptes)")
    p.add_argument("-o", "--output", required=True, help="fichier (une image) ou dossier de sortie")
    p.add_argument("--proba", action="store_true", help="ecrire aussi les probabilites")
    p.add_argument("--uncertainty", action="store_true", help="ecrire aussi la carte d'incertitude")
    p.add_argument("--smooth", type=float, default=0.0, help="lissage des probabilites (sigma, px)")
    p.add_argument("--block", type=int, default=None, help="taille de bloc (px), defaut automatique")
    p.add_argument("--channel-axis", type=int, default=None, help="axe des canaux si multicanal")
    p.add_argument("--key", default=None, help="dataset a lire dans un HDF5/Zarr")
    p.add_argument("--format", default="tif", help="format des sorties (tif, npy, h5)")
    p.add_argument("--ilastik-mode", choices=("forest", "retrain"), default="forest")
    p.add_argument("--raw", nargs="*", help="chemins des donnees ilastik si deplacees")
    p.add_argument("--spacing", nargs="+", type=float, default=None,
                   help="taille de voxel (z y x) pour un modele nnU-Net (defaut : celle de l'entrainement)")
    p.add_argument("--no-tta", action="store_true",
                   help="modele nnU-Net : sans augmentation par miroirs (8x plus rapide, un peu moins precis)")
    p.add_argument("--ipsdk-values", action="store_true",
                   help="modele IPSDK Explorer : ecrire les valeurs de sortie d'IPSDK (champ Value)")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("train", help="entrainer un modele depuis images + images de labels")
    p.add_argument("--image", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", required=True, help="0 = non annote, 1..N = classes")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--classifier", default="rf", choices=("rf", "extratrees", "lightgbm", "xgboost"))
    p.add_argument("--n-estimators", type=int, default=None)
    p.add_argument("--families", nargs="+", default=None)
    p.add_argument("--scales", nargs="+", type=float, default=None)
    p.add_argument("--radii", nargs="+", type=int, default=None)
    p.add_argument("--spacing", nargs="+", type=float, default=None, help="taille de voxel (z y x)")
    p.add_argument("--in2d", action="store_true", help="filtres 2D par coupe pour des volumes 3D")
    p.add_argument("--normalize", default="none", choices=("none", "minmax", "percentile", "zscore"))
    p.add_argument("--class-names", default=None, help='ex. "1=fond,2=fils"')
    p.add_argument("--channel-axis", type=int, default=None)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("convert", help="convertir ilastik / IPSDK Explorer / ancien modele en modele xwi")
    p.add_argument("model", help="projet .ilp, dossier de modele IPSDK Explorer ou ancien .joblib")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--mode", choices=("forest", "retrain"), default="forest")
    p.add_argument("--features", choices=("xwi", "ilastik", "ipsdk", "equivalent"), default="xwi",
                   help="jeu de features pour --mode retrain (ipsdk/equivalent : modeles IPSDK)")
    p.add_argument("--raw", nargs="*", help="chemins des donnees ilastik si deplacees")
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("info", help="decrire un modele")
    p.add_argument("model")
    p.add_argument("--top", type=int, default=10)
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("inspect", help="analyser un fichier (.ilp, .joblib, .bin IPSDK...)")
    p.add_argument("path")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("ilastik-export", help="exporter images + annotations d'un projet ilastik")
    p.add_argument("project")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--raw", nargs="*")
    p.set_defaults(func=cmd_ilastik_export)

    p = sub.add_parser("ipsdk-export", help="exporter images + annotations d'un modele IPSDK Explorer")
    p.add_argument("model", help="dossier du modele (contenant Settings.mho)")
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(func=cmd_ipsdk_export)

    p = sub.add_parser("ipsdk-distill", help="reproduire un modele IPSDK (.bin) avec PyIPSDK")
    p.add_argument("model")
    p.add_argument("--image", nargs="+", required=True, help="images representatives")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--samples", type=int, default=50_000, help="pixels par classe")
    p.add_argument("--classifier", default="rf")
    p.add_argument("--channel-axis", type=int, default=None)
    p.set_defaults(func=cmd_ipsdk_distill)

    from .nnunet.cli import add_nnunet_parser
    add_nnunet_parser(sub)

    p = sub.add_parser("features", help="lister les familles de features disponibles")
    p.set_defaults(func=cmd_features)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.device:
        from .backend import set_device
        set_device(args.device)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
