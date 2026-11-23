# Journal des modifications

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
