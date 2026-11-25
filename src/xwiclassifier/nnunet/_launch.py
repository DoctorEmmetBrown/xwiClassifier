"""Lanceur interne (sous-processus) : entraine une liste de folds sur un peripherique.

    python -m xwiclassifier.nnunet._launch DATASET CONFIG PLANS DEVICE NUM_GPUS CONTINUE NPZ FOLD [FOLD...]
"""
import sys


def main(argv):
    dataset, config, plans, device, num_gpus, cont, npz, *folds = argv
    import torch
    from nnunetv2.run.run_training import run_training

    dev = torch.device(device)
    if device == "cpu":
        torch.set_num_threads(max(1, torch.get_num_threads()))
    for fold in folds:
        fold = fold if fold == "all" else int(fold)
        print(f"[xwi] entrainement : jeu {dataset}, configuration {config}, fold {fold}, peripherique {device}",
              flush=True)
        run_training(dataset, config, fold, trainer_class_name="nnUNetTrainerXwi", plans_identifier=plans,
                     num_gpus=int(num_gpus), export_validation_probabilities=npz == "1",
                     continue_training=cont == "1", device=dev)
        print(f"[xwi] fold {fold} termine", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
