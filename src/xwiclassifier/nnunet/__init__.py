"""Extension nnU-Net de xwiClassifier (optionnelle : ``pip install xwiclassifier[nnunet]``).

Flux typique ::

    from xwiclassifier.nnunet import NnUNetWorkspace, plan_and_preprocess, train, NnUNetSegmenter

    ws = NnUNetWorkspace("~/xwi_nnunet")
    ds = ws.create_dataset("Rein vaisseaux", {1: "fond", 2: "vaisseaux"})
    ds.add_case(volume, annotations, spacing=(2.0, 2.0, 2.0))             # annotations partielles
    ds.add_case(volume, segmentation_corrigee, roi=region, mode="dense")  # region validee
    plan_and_preprocess(ds)
    job = train(ds, folds=("all",), epochs=250)                            # GPU detectes automatiquement
    job.wait()
    seg = NnUNetSegmenter(job.model_folder).predict(autre_volume)
"""
from .inference import NnUNetSegmenter, find_model_folder, is_nnunet_model_folder  # noqa: F401
from .training import (  # noqa: F401
    PLANNERS, TrainingJob, detect_gpus, model_folder, nnunet_available, plan_and_preprocess,
    resolve_device, train, training_status,
)
from .workspace import NnUNetDataset, NnUNetWorkspace, default_root  # noqa: F401
