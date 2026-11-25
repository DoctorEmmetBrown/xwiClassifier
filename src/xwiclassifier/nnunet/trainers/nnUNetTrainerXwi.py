"""Entraineur nnU-Net utilise par xwiClassifier (dossier declare via nnUNet_extTrainer).

Identique a nnUNetTrainer (meme reseau, memes augmentations) ; seule la duree est
reglable par variables d'environnement, positionnees par xwi :
  XWI_NNUNET_EPOCHS, XWI_NNUNET_ITERS (iterations par epoque),
  XWI_NNUNET_VAL_ITERS (iterations de validation par epoque).
"""
import os

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerXwi(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = int(os.environ.get("XWI_NNUNET_EPOCHS", self.num_epochs))
        self.num_iterations_per_epoch = int(os.environ.get("XWI_NNUNET_ITERS", self.num_iterations_per_epoch))
        self.num_val_iterations_per_epoch = int(os.environ.get("XWI_NNUNET_VAL_ITERS",
                                                               self.num_val_iterations_per_epoch))
