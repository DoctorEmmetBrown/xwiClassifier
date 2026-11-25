# Journal des modifications

## 0.3.0

- Import des modèles IPSDK Explorer (dossiers Smart Segmentation) sans IPSDK : lecture de
  `Settings.mho`, de la forêt scikit-learn `Model` quelle que soit sa version (pickles 0.24
  compris), des images et annotations ; vérification et calibration sur les annotations.
- Extracteur `IPSDKFeatures` reproduisant la grille de features IPSDK (Gaussian, Mean,
  Laplacian Difference, High Pass, Variance, Erosion, Dilation, Opening, Closing,
  dimension 2D/3D, multi-résolution).
- Nouvelles familles xwi transcrites d'IPSDK : `mean`, `high_pass`, `box_variance` ;
  morphologie `morpho_shape="exact"` (disque/boule identiques à IPSDK).
- CLI : `predict`/`convert`/`inspect`/`info` acceptent un dossier IPSDK, `ipsdk-export`,
  option `--ipsdk-values` ; napari : chargement d'un dossier de modèle IPSDK.

## 0.2.0

- Paquet renommé `xwiclassifier` ; le cœur ne dépend plus de napari.
- Moteur de features réécrit : dérivées gaussiennes exactes avec noyaux corrigés
  (façon vigra), valeurs propres analytiques, cache par échelle, multi-thread,
  anisotropie, filtres 2D par coupe, normalisation globale, familles `variance` et `ridge`,
  morphologie mutualisée et accélérée.
- Traitement par blocs avec halo exact, budget mémoire adapté à la machine,
  plan d'entraînement adaptatif, réutilisation des features entre fit et predict.
- `PixelClassifier` : multi-images, plafond par classe, probabilités, incertitude,
  lissage, importances groupées ; format de modèle v2 (et relecture des modèles v0.1).
- Import des projets ilastik (classifieur reproduit à l'identique, ou ré-entraînement).
- Récupération des modèles IPSDK par distillation (PyIPSDK) et inspecteur de fichiers.
- Ligne de commande `xwi` et API Python.
- Widget napari réécrit (threads, prédiction de la vue, mode live, import ilastik/IPSDK).
- Correction : la normalisation min/max par image de la v0.1 faisait échouer un
  modèle sur une nouvelle image contenant quelques pixels aberrants.

## 0.1.0

- Prototype napari : features lisses et morphologiques, RF/ExtraTrees/LightGBM/XGBoost,
  annotation multi-classes.
