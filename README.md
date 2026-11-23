# napari-segforest

Classification interactive de pixels/voxels pour images **2D et 3D**, sous
[napari]. Alternative libre à ilastik et à IPSDK Explorer (Reactiv'IP), conçue
pour aller **au-delà d'ilastik** : catalogue de features ouvert et entièrement
paramétrable, et classifieur pluggable (Random Forest par défaut, gradient
boosting en option).

## Pourquoi

- **Plus de features qu'ilastik** : familles et échelles (sigmas) librement
  combinables, plus possibilité d'ajouter des features custom — pour donner au
  Random Forest une décision plus facile et plus robuste.
- **Rapide** : backend hybride. Filtres exécutés sur **GPU** via
  `pyclesperanto` si une carte est disponible, sinon repli automatique sur
  **CPU** (`scikit-image`/`scipy`). Aucun changement de code côté utilisateur.
- **Interactif** : annotation au pinceau dans napari, entraînement, prédiction,
  et mode *live* (réentraînement à chaque coup de pinceau).
- **Pluggable** : `rf`, `extratrees`, `lightgbm`, `xgboost`.

## Architecture

```
src/napari_segforest/
├── napari.yaml              # manifest du plugin (widgets exposés)
├── backend.py              # abstraction CPU/GPU + primitives de filtres
├── features/
│   └── engine.py           # FeatureEngine + FeatureSpec (catalogue ouvert)
├── classifier/
│   └── base.py             # PixelClassifier (RF/boosting), fit/predict/save
└── widgets/
    ├── pixel_classifier.py # widget principal type ilastik
    └── feature_explorer.py # visualisation des canaux de features
examples/demo_3d.py         # démo headless + napari sur volume 3D
tests/test_pipeline.py      # tests CPU sans Qt
```

Le flux : `image → FeatureEngine.compute() → pile (H,W[,D],N_features) →
PixelClassifier.fit(pixels peints) → predict(volume) → layer Labels`.

### Le moteur de features (cœur du projet)

`FeatureSpec(families=..., sigmas=..., extra=[...])` décrit de façon
reproductible le jeu de features. Familles disponibles :

- **Lisses** (paramètre = sigma, en px) : `intensity`, `gaussian`, `gradient`,
  `log` (Laplacian of Gaussian), `dog` (Difference of Gaussians),
  `hessian_eigen`, `structure_eigen`.
- **Morphologiques** (paramètre = rayon, en px) : `erosion`, `dilation`,
  `opening`, `closing`, `morph_gradient` (dilatation − érosion),
  `white_tophat` (objets clairs), `black_tophat` (objets sombres), `median`.

La morphologie est calculée en n-D (2D **et** 3D) via `scipy.ndimage`, avec
accélération GPU `pyclesperanto` si disponible. OpenCV n'est pas utilisé car il
est limité à la 2D. Chaque famille × chaque taille produit un ou plusieurs
canaux. Pour ajouter une feature maison :

```python
spec = FeatureSpec(
    families=("intensity", "gaussian", "log", "hessian_eigen"),
    sigmas=(0, 1, 2, 4, 8, 16),
    extra=[("ma_feature", lambda img: mon_filtre(img))],
)
```

## Installation

```bash
pip install -e .              # CPU
pip install -e ".[gpu]"       # + pyclesperanto (installer cupy selon CUDA)
pip install -e ".[boosting]"  # + lightgbm / xgboost
```

Forcer le CPU (debug) : `export SEGFOREST_FORCE_CPU=1`.

## Utilisation

```bash
napari                        # puis Plugins → SegForest → Pixel classifier
python examples/demo_3d.py    # démo
```

Dans le widget : sélectionner l'image, régler le nombre de classes (1 à 5, en
plus du **Background** créé par défaut), **renommer** chaque classe via les
champs texte (ex. « noyau », « membrane »), cocher les familles de features et
les sigmas, créer le layer d'annotation, cliquer une classe (sa couleur devient
celle du pinceau), peindre, puis *Entraîner* et *Prédire*. Cocher *Mode live*
pour un retour immédiat. Les noms de classes sont sauvegardés et rechargés avec
le modèle (`.joblib`).

## Feuille de route

- Prédiction par blocs (dask) pour gros volumes ne tenant pas en RAM.
- Object classification (post-segmentation) façon ilastik.
- Export des features GPU sans aller-retour mémoire.
- Apprentissage actif (suggérer les zones à annoter).

[napari]: https://napari.org
```
