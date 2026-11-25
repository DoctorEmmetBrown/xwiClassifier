"""Preparation et entrainement nnU-Net (sous-processus, journaux, suivi).

Avec plusieurs GPU, les folds sont repartis : chaque GPU entraine ses folds l'un
apres l'autre (methode recommandee par nnU-Net). ``ddp=True`` utilise au contraire
tous les GPU pour un seul fold (data parallel).
"""
from __future__ import annotations

import datetime as _dt
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .workspace import NnUNetDataset

TRAINER = "nnUNetTrainerXwi"
PLANNERS = {
    "default": ("ExperimentPlanner", "nnUNetPlans"),
    "resenc_m": ("nnUNetPlannerResEncM", "nnUNetResEncUNetMPlans"),  # ~9-11 Go de VRAM
    "resenc_l": ("nnUNetPlannerResEncL", "nnUNetResEncUNetLPlans"),  # ~24 Go de VRAM
}


def nnunet_available() -> bool:
    try:
        import nnunetv2  # noqa: F401
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def detect_gpus() -> list[int]:
    """Indices des GPU NVIDIA visibles (nvidia-smi, sinon torch)."""
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20).stdout
            n = sum(1 for line in out.splitlines() if line.startswith("GPU "))
            if n:
                return list(range(n))
        except Exception:
            pass
    try:
        import torch
        return list(range(torch.cuda.device_count()))
    except Exception:
        return []


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    if detect_gpus():
        return "cuda"
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _python_call(module_call: str) -> list:
    return [sys.executable, "-c", module_call]


def plan_and_preprocess(dataset: NnUNetDataset, configurations=None, planner: str = "default",
                        num_processes: int = 4, verify: bool = True, wait: bool = True):
    """Analyse le jeu de donnees, choisit l'architecture et pretraite (nnUNetv2_plan_and_preprocess)."""
    pl, plans = PLANNERS[planner]
    cfg = list(configurations or [dataset.default_configuration])
    args = ["-d", str(dataset.id), "-pl", pl, "-c", *cfg, "-np", str(num_processes), "-npfp", str(num_processes)]
    if verify:
        args.append("--verify_dataset_integrity")
    log = dataset.ws.logs / f"{dataset.name}_plan_{_stamp()}.log"
    cmd = _python_call("import sys; from nnunetv2.experiment_planning.plan_and_preprocess_entrypoints "
                       "import plan_and_preprocess_entry as e; sys.argv=['nnUNetv2_plan_and_preprocess']"
                       f"+{args!r}; e()")
    proc = _spawn(cmd, dataset.ws.env(), log)
    job = TrainingJob(dataset, [proc], [log], kind="plan", plans=plans, configuration=cfg[0])
    if wait:
        job.wait(check=True)
    return job


def train(dataset: NnUNetDataset, configuration: str | None = None, folds=("all",), epochs: int = 250,
          iterations_per_epoch: int | None = None, val_iterations: int | None = None, gpus=None,
          device: str = "auto", planner: str = "default", ddp: bool = False, continue_training: bool = False,
          npz: bool = False, wait: bool = False):
    """Lance l'entrainement (non bloquant par defaut) et renvoie un TrainingJob.

    ``folds`` : ("all",) = un modele sur toutes les donnees ; (0, 1, 2, 3, 4) = validation croisee.
    ``epochs`` : nombre d'epoques (1000 = defaut nnU-Net ; 100-250 suffit souvent pour demarrer).
    """
    configuration = configuration or dataset.default_configuration
    plans = PLANNERS[planner][1]
    device = resolve_device(device)
    gpus = list(gpus) if gpus is not None else (detect_gpus() if device == "cuda" else [])
    extra = {"XWI_NNUNET_EPOCHS": int(epochs)}
    if iterations_per_epoch:
        extra["XWI_NNUNET_ITERS"] = int(iterations_per_epoch)
    if val_iterations:
        extra["XWI_NNUNET_VAL_ITERS"] = int(val_iterations)
    folds = [str(f) for f in folds]
    flags = ["1" if continue_training else "0", "1" if npz else "0"]
    procs, logs = [], []
    stamp = _stamp()
    if device == "cuda" and gpus and (len(gpus) == 1 or ddp or len(folds) == 1):
        groups = [(gpus if ddp else gpus[:1], folds)]
    elif device == "cuda" and gpus:
        groups = [([g], folds[i::len(gpus)]) for i, g in enumerate(gpus) if folds[i::len(gpus)]]
    else:
        groups = [([], folds)]
    for gpu_ids, fl in groups:
        env_extra = dict(extra)
        if gpu_ids:
            env_extra["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
        n_gpu = str(len(gpu_ids)) if (ddp and len(gpu_ids) > 1) else "1"
        cmd = [sys.executable, "-m", "xwiclassifier.nnunet._launch", str(dataset.id), configuration, plans,
               device, n_gpu, *flags, *fl]
        log = dataset.ws.logs / f"{dataset.name}_train_{stamp}_{'gpu' + '-'.join(map(str, gpu_ids)) if gpu_ids else device}.log"
        procs.append(_spawn(cmd, dataset.ws.env(env_extra), log))
        logs.append(log)
    job = TrainingJob(dataset, procs, logs, kind="train", plans=plans, configuration=configuration,
                      folds=folds, epochs=int(epochs))
    if wait:
        job.wait(check=True)
    return job


def model_folder(dataset: NnUNetDataset, configuration: str | None = None, planner: str = "default") -> Path:
    plans = PLANNERS[planner][1]
    return dataset.ws.results / dataset.name / f"{TRAINER}__{plans}__{configuration or dataset.default_configuration}"


# --------------------------------------------------------------------------
def _stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _spawn(cmd: list, env: dict, log: Path):
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log, "w", buffering=1)
    fh.write("$ " + " ".join(cmd) + "\n")
    return subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(log.parent))


@dataclass
class TrainingJob:
    dataset: NnUNetDataset
    processes: list
    logs: list
    kind: str = "train"
    plans: str = "nnUNetPlans"
    configuration: str = "3d_fullres"
    folds: list = field(default_factory=list)
    epochs: int = 0

    def running(self) -> bool:
        return any(p.poll() is None for p in self.processes)

    def wait(self, check: bool = False) -> int:
        codes = [p.wait() for p in self.processes]
        if check and any(codes):
            tail = "\n".join(Path(lg).read_text(errors="replace").splitlines()[-25:] for lg in self.logs)
            raise RuntimeError(f"nnU-Net ({self.kind}) a echoue :\n{tail}")
        return max(codes) if codes else 0

    def terminate(self) -> None:
        for p in self.processes:
            if p.poll() is None:
                p.terminate()

    @property
    def model_folder(self) -> Path:
        return self.dataset.ws.results / self.dataset.name / f"{TRAINER}__{self.plans}__{self.configuration}"

    def status(self) -> dict:
        st = training_status(self.model_folder, self.folds)
        st["running"] = self.running()
        st["logs"] = [str(lg) for lg in self.logs]
        return st


_EPOCH = re.compile(r"^Epoch (\d+)")
_DICE = re.compile(r"Pseudo dice \[(.*?)\]")
_EMA = re.compile(r"New best EMA pseudo Dice: ([0-9.]+)")
_TIME = re.compile(r"Epoch time: ([0-9.]+) s")
_LOSS = re.compile(r"^(train_loss|val_loss) (-?[0-9.]+(?:e-?\d+)?)")
_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:e-?\d+)?")


def _numbers(text: str) -> list:
    """Nombres d'une liste nnU-Net, ex. '[np.float32(0.77), 0.81]' -> [0.77, 0.81]."""
    inside = re.findall(r"\((-?\d+(?:\.\d+)?(?:e-?\d+)?)\)", text)
    return [float(x) for x in (inside or _NUM.findall(text))]


def training_status(model_dir, folds=None) -> dict:
    """Etat de l'entrainement d'apres les journaux nnU-Net de chaque fold."""
    model_dir = Path(model_dir)
    out = {"folds": {}}
    fold_dirs = [model_dir / f"fold_{f}" for f in folds] if folds else sorted(model_dir.glob("fold_*"))
    for fd in fold_dirs:
        info = {"epoch": None, "pseudo_dice": None, "best_ema_dice": None, "epoch_time_s": None,
                "train_loss": None, "val_loss": None, "finished": (fd / "checkpoint_final.pth").exists()}
        logs = sorted(fd.glob("training_log_*.txt"))
        if logs:
            for line in logs[-1].read_text(errors="replace").splitlines():
                text = (line.split(": ", 1)[-1] if ": " in line else line).strip()
                if m := _EPOCH.match(text):
                    info["epoch"] = int(m.group(1))
                elif m := _DICE.search(text):
                    info["pseudo_dice"] = _numbers(m.group(1))
                elif m := _LOSS.match(text):
                    info[m.group(1)] = float(m.group(2))
                elif m := _EMA.search(text):
                    info["best_ema_dice"] = float(m.group(1))
                elif m := _TIME.search(text):
                    info["epoch_time_s"] = float(m.group(1))
        out["folds"][fd.name[5:]] = info
    return out
