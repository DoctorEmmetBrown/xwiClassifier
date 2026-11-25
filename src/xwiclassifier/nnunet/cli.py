"""Sous-commandes ``xwi nnunet ...`` : jeux de donnees, preparation, entrainement, suivi."""
from __future__ import annotations

import json
import sys
import time

import numpy as np


def _ws(args):
    from .workspace import NnUNetWorkspace
    return NnUNetWorkspace(args.root)


def _parse_classes(s: str) -> dict:
    out = {}
    for part in s.split(","):
        k, v = part.split("=", 1)
        out[int(k)] = v.strip()
    return out


def _parse_roi(s: str | None):
    if not s:
        return None
    return tuple(slice(*[int(x) if x else None for x in part.split(":")]) for part in s.split(","))


def cmd_init(args):
    ws = _ws(args)
    print(f"Espace de travail : {ws.root}")
    print("Pour utiliser les commandes nnU-Net directement :")
    for k in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results", "nnUNet_extTrainer"):
        print(f'  export {k}="{ws.env()[k]}"')
    from .training import detect_gpus, nnunet_available
    print(f"nnU-Net installe : {nnunet_available()} ; GPU detectes : {detect_gpus() or 'aucun'}")
    return 0


def cmd_create(args):
    ws = _ws(args)
    ds = ws.create_dataset(args.name, _parse_classes(args.classes), n_channels=args.channels,
                           normalization=args.normalization, dataset_id=args.id)
    print(f"Cree : {ds.folder}")
    return 0


def cmd_add(args):
    from ..io import read_image

    ds = _ws(args).get_dataset(args.dataset)
    pairs = []
    if args.from_ilastik:
        from ..io.ilastik_project import ilastik_annotations
        a = ilastik_annotations(args.from_ilastik)
        pairs.append((a["image"], a["labels"], a["channel_axis"], args.from_ilastik))
    elif args.from_ipsdk:
        from ..io.ipsdk_explorer import ipsdk_annotations
        for a in ipsdk_annotations(args.from_ipsdk):
            if a["labels"] is not None:
                pairs.append((a["image"], a["labels"], a["channel_axis"], f"{args.from_ipsdk}:{a['name']}"))
    else:
        if not (args.image and args.labels):
            print("--image et --labels requis (ou --from-ilastik / --from-ipsdk).", file=sys.stderr)
            return 2
        pairs.append((read_image(args.image), read_image(args.labels), args.channel_axis, args.image))
    for img, lab, ch, src in pairs:
        case = ds.add_case(img, np.asarray(lab), spacing=args.spacing, channel_axis=ch, roi=_parse_roi(args.roi),
                           margin=args.margin, mode=args.mode, source=str(src))
        print(f"{case} ajoute ({args.mode}) depuis {src}")
    print(ds.summary())
    return 0


def cmd_plan(args):
    from .training import plan_and_preprocess
    ds = _ws(args).get_dataset(args.dataset)
    plan_and_preprocess(ds, planner=args.planner, num_processes=args.np)
    print(f"Pret a entrainer : {ds.name} (planificateur {args.planner})")
    return 0


def cmd_train(args):
    from .training import train
    ds = _ws(args).get_dataset(args.dataset)
    folds = ["all"] if args.folds == ["all"] else [int(f) for f in args.folds]
    job = train(ds, configuration=args.configuration, folds=folds, epochs=args.epochs, gpus=args.gpus,
                device=args.device, planner=args.planner, ddp=args.ddp, continue_training=args.cont)
    print(f"Entrainement lance ({len(job.processes)} processus) -> {job.model_folder}")
    for lg in job.logs:
        print(f"  journal : {lg}")
    if args.detach:
        return 0
    try:
        while job.running():
            time.sleep(args.poll)
            _print_status(job.status())
    except KeyboardInterrupt:
        print("\nInterruption : arret de l'entrainement (reprise possible avec --continue).")
        job.terminate()
        return 130
    code = job.wait()
    _print_status(job.status())
    return code


def _print_status(st):
    parts = []
    for f, i in st["folds"].items():
        dice = f"{i['best_ema_dice']:.3f}" if i["best_ema_dice"] is not None else "-"
        parts.append(f"fold {f}: epoque {i['epoch']} Dice EMA {dice}{' (termine)' if i['finished'] else ''}")
    print("  " + " | ".join(parts) if parts else "  (en attente des premiers journaux)", flush=True)


def cmd_status(args):
    from .training import training_status
    ws = _ws(args)
    ds = ws.get_dataset(args.dataset)
    for m in ws.trained_models(ds) or []:
        print(m.name)
        _print_status(training_status(m))
    if not ws.trained_models(ds):
        print("Aucun modele pour ce jeu de donnees.")
    return 0


def cmd_list(args):
    ws = _ws(args)
    print(f"Espace : {ws.root}")
    for d in ws.datasets():
        models = [m.name for m in ws.trained_models(ws.get_dataset(d.id))]
        print(f"  {d.name} : {d.n_cases} cas, classes {d.classes}, modeles : {models or 'aucun'}")
    return 0


def add_nnunet_parser(sub):
    p = sub.add_parser("nnunet", help="segmentation par apprentissage profond (nnU-Net)")
    p.add_argument("--root", default=None, help="espace de travail (defaut : $XWI_NNUNET_ROOT ou ~/xwi_nnunet)")
    s = p.add_subparsers(dest="nn_cmd", required=True)

    q = s.add_parser("init", help="afficher l'espace de travail et les GPU")
    q.set_defaults(func=cmd_init)

    q = s.add_parser("create", help="creer un jeu de donnees (un par organe / structure)")
    q.add_argument("name")
    q.add_argument("--classes", required=True, help='ex. "1=fond,2=vaisseaux,3=glomerules"')
    q.add_argument("--channels", type=int, default=1)
    q.add_argument("--normalization", choices=("volume", "ct", "none"), default="volume")
    q.add_argument("--id", type=int, default=None)
    q.set_defaults(func=cmd_create)

    q = s.add_parser("add", help="ajouter des cas (annotations partielles ou region validee)")
    q.add_argument("dataset")
    q.add_argument("--image")
    q.add_argument("--labels", help="0 = non annote, 1 = fond, 2.. = classes")
    q.add_argument("--from-ilastik", default=None, help="importer les annotations d'un projet .ilp")
    q.add_argument("--from-ipsdk", default=None, help="importer les annotations d'un modele IPSDK Explorer")
    q.add_argument("--spacing", nargs="+", type=float, default=None)
    q.add_argument("--mode", choices=("sparse", "dense"), default="sparse")
    q.add_argument("--roi", default=None, help='region validee, ex. "100:228,0:512,0:512"')
    q.add_argument("--margin", type=int, default=16)
    q.add_argument("--channel-axis", type=int, default=None)
    q.set_defaults(func=cmd_add)

    q = s.add_parser("plan", help="analyser et pretraiter (nnUNetv2_plan_and_preprocess)")
    q.add_argument("dataset")
    q.add_argument("--planner", choices=("default", "resenc_m", "resenc_l"), default="default")
    q.add_argument("--np", type=int, default=4, help="processus de pretraitement")
    q.set_defaults(func=cmd_plan)

    q = s.add_parser("train", help="entrainer (folds repartis sur les GPU)")
    q.add_argument("dataset")
    q.add_argument("--folds", nargs="+", default=["all"], help='"all" ou 0 1 2 3 4')
    q.add_argument("--epochs", type=int, default=250)
    q.add_argument("--configuration", default=None, help="2d, 3d_fullres... (defaut selon les donnees)")
    q.add_argument("--gpus", nargs="+", type=int, default=None, help="indices des GPU (defaut : tous)")
    q.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    q.add_argument("--planner", choices=("default", "resenc_m", "resenc_l"), default="default")
    q.add_argument("--ddp", action="store_true", help="un seul fold sur tous les GPU")
    q.add_argument("--continue", dest="cont", action="store_true", help="reprendre un entrainement interrompu")
    q.add_argument("--detach", action="store_true", help="ne pas attendre la fin")
    q.add_argument("--poll", type=float, default=60.0, help="intervalle d'affichage (s)")
    q.set_defaults(func=cmd_train)

    q = s.add_parser("status", help="etat des entrainements d'un jeu de donnees")
    q.add_argument("dataset")
    q.set_defaults(func=cmd_status)

    q = s.add_parser("list", help="lister jeux de donnees et modeles")
    q.set_defaults(func=cmd_list)
    return p


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps({"usage": "xwi nnunet ..."}))
