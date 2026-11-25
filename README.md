# xwiClassifier

Classification de pixels / voxels **2D et 3D** par apprentissage (features + forêts
d'arbres), conçue comme alternative libre à **ilastik** et à **IPSDK Explorer
(Reactiv'IP)**. Trois façons de s'en servir :

- **dans napari** : annotation au pinceau, entraînement et prédiction interactifs ;
- **en Python** : `import xwiclassifier as xwi`, pour scripter des traitements par lots ;
- **en ligne de commande** : `xwi predict modele.joblib volume.tif -o seg.tif`.

Un modèle entraîné dans napari s'utilise tel quel en script ou en ligne de
commande, et les modèles **ilastik** (`.ilp`) se chargent directement.

## Points forts

- **Catalogue de features ouvert** : lissage, gradient, laplacien, DoG, valeurs
  propres de la Hessienne et du tenseur de structure, écart-type local,
  tubularité (fils/fibres), morphologie (érosion, dilatation, ouverture,
  fermeture, gradient morphologique, top-hats, médian) et filtres d'IPSDK (moyenne,
  passe-haut, variance sur boîte). Chaque famille se combine
  librement avec des sigmas / rayons au choix, bien au-delà d'ilastik.
- **Rapide et sobre en mémoire** : dérivées gaussiennes exactes, valeurs propres
  analytiques, cache partagé par échelle, multi-thread, traitement **par blocs
  avec halo** (résultat identique au calcul global, mémoire bornée), GPU optionnel.
- **Robuste sur de nouvelles acquisitions** : normalisation calculée sur l'image
  entière (pas par bloc ni par image min/max), prise en compte de l'**anisotropie**
  (taille de voxel), filtres 2D par coupe en option.
- **Classifieurs** : RandomForest (défaut), ExtraTrees, LightGBM, XGBoost.
- **Import ilastik sans ilastik ni vigra** : features reproduites à l'identique et
  forêts converties en scikit-learn ; ou ré-entraînement sur les annotations du projet.
- **Import des modèles IPSDK Explorer** (dossiers Smart Segmentation) sans IPSDK.

## Installation

```bash
git clone https://github.com/DoctorEmmetBrown/xwiClassifier.git
cd xwiClassifier
python -m venv xwi-env              # le nom de l'environnement est libre
source xwi-env/bin/activate         # Windows : xwi-env\Scripts\activate
pip install -e ".[napari]"          # cœur + napari
```

Options : `.[boosting]` (LightGBM, XGBoost ; sur macOS XGBoost demande
`brew install libomp`), `.[gpu]` (pyclesperanto), `.[nnunet]` (apprentissage profond,
voir docs/nnunet.md), `.[zarr]`, `.[all]`, `.[dev]` (tests).
Sans napari, `pip install -e .` suffit pour le Python et la ligne de commande.

## Dans napari

```bash
napari          # puis Plugins → xwiClassifier → Classification de pixels
```

1. choisir l'image (la taille de voxel est reprise de l'échelle du layer, ou saisie) ;
2. régler le nombre de classes (background + jusqu'à 9), les renommer ;
3. cliquer une classe : sa couleur devient celle du pinceau ; régler la taille du pinceau ;
4. cocher les features (matrice famille × sigma, et famille × rayon pour la morphologie) ;
5. **Entraîner** (prédit la coupe affichée), **Prédire tout** pour le volume entier ;
6. *Mode live* : ré-entraînement automatique après chaque trait ; *incertitude* :
   indique où annoter en priorité ;
7. **Charger** un modèle `.joblib` ou un projet ilastik `.ilp` (utiliser son
   classifieur, ou importer image + annotations pour continuer dans napari).

## En Python (sans napari)

```python
import xwiclassifier as xwi

# appliquer un modèle existant (xwi .joblib ou projet ilastik .ilp)
model = xwi.load_model("modele.joblib")
vol = xwi.read_image("volume.tif")
seg, proba = model.predict(vol, return_proba=True)        # calcul par blocs
xwi.write_image("segmentation.tif", seg)

# entraîner
spec = xwi.FeatureSpec.from_grid(
    families=("gaussian", "gradient", "hessian_eigen", "structure_eigen", "variance", "ridge",
              "white_tophat"),
    scales=(0.7, 1.0, 1.6, 3.5, 5.0), radii=(1, 2, 4),
    spacing=(2.0, 1.0, 1.0))                               # voxels deux fois plus épais en z
model = xwi.PixelClassifier(spec, kind="rf", class_names={1: "fond", 2: "fils"})
model.fit([vol1, vol2], [annot1, annot2])                  # 0 = non annoté
print(model.feature_importances(group="family"))
model.save("modele.joblib")
```

## En ligne de commande

```bash
xwi features                                              # familles disponibles
xwi train --image vol.tif --labels annot.tif -o modele.joblib \
          --spacing 2 1 1 --class-names "1=fond,2=fils"
xwi predict modele.joblib "data/*.tif" -o resultats/ --proba --uncertainty --smooth 1
xwi info modele.joblib                                    # description + contributions
xwi convert projet.ilp -o modele.joblib                   # ilastik -> modèle xwi
xwi convert DossierIPSDK -o modele.joblib                # IPSDK Explorer -> modèle xwi
xwi ilastik-export projet.ilp -o annotations/             # images + annotations en TIFF
xwi inspect fichier.(ilp|joblib|bin)                      # analyser un fichier
```

`python -m xwiclassifier ...` est équivalent à `xwi ...`. Formats lus : TIFF,
HDF5 (`fichier.h5/dataset`), NumPy, Zarr, PNG/JPEG.

## Récupérer des projets ilastik

```python
m = xwi.load_ilastik_model("projet.ilp")                         # classifieur ilastik tel quel
m = xwi.load_ilastik_model("projet.ilp", mode="retrain")         # ré-entraîné, features xwi
m = xwi.load_ilastik_model("projet.ilp", mode="retrain", features="ilastik")
```

- `mode="forest"` reproduit les features d'ilastik (pré-lissage, fenêtres 2σ et
  3,5σ, noyaux de dérivée corrigés comme vigra, ordre des canaux) et convertit
  ses forêts vigra en forêt scikit-learn aux prédictions identiques. Validé sur un
  projet réel (ilastik 1.4, 2D) : 100 % d'accord avec les 1132 pixels annotés.
- Les données référencées hors du projet sont recherchées à leur chemin d'origine ;
  sinon, indiquez-les avec `raw_paths=[...]` (ou `--raw`).
- Limite connue : la reproduction des projets **3D** suit le même code mais n'a été
  validée que sur des projets synthétiques ; testez-la sur un de vos projets.

## Récupérer des modèles IPSDK Explorer

Un modèle « Smart Segmentation » est un dossier
(`%APPDATA%\ReactivIP\Explorer\Machine_Learning\Pixel_Classification\<modèle>`) qui contient
une forêt **scikit-learn** (`Model`), sa grille de features (`Settings.mho`) et les images
annotées. xwi le charge directement, sans IPSDK ni licence :

```bash
xwi predict "…/Pixel_Classification/MonModele" vol.tif -o seg.tif [--ipsdk-values]
xwi convert "…/Pixel_Classification/MonModele" -o mon_modele.joblib
```

Les filtres IPSDK (Gaussian, Mean, Laplacian Difference, High Pass, Variance, morphologie,
multi-résolution) sont reproduits à l'identique, et transcrits en familles xwi (`mean`,
`high_pass`, `box_variance`, morphologie `exact`). L'import vérifie le modèle sur ses propres
annotations. Détails, correspondances et limites : [docs/ipsdk.md](docs/ipsdk.md).

Pour un fichier `.bin` isolé, sans son dossier : `xwi ipsdk-distill` (PyIPSDK requis).

## Apprentissage profond (nnU-Net)

Pour les structures difficiles (arbre vasculaire, glomérules…) et un modèle par
organe : extension nnU-Net (`pip install -e ".[nnunet]"`, GPU NVIDIA conseillé).
Les annotations xwi, ilastik ou IPSDK (partielles ou régions validées) alimentent
un jeu de données par organe ; l'entraînement répartit les folds sur vos GPU ; la
prédiction par blocs s'applique à des volumes et séries entiers.

```bash
xwi nnunet create "Rein vaisseaux" --classes "1=fond,2=vaisseaux"
xwi nnunet add 501 --image vol.tif --labels annotations.tif --spacing 2 2 2
xwi nnunet plan 501 && xwi nnunet train 501 --epochs 250
xwi predict ~/xwi_nnunet/nnUNet_results/Dataset501_ReinVaisseaux "serie/*.tif" -o seg/
```

Détails : [docs/nnunet.md](docs/nnunet.md). Widget napari : *nnU-Net*.

## Réglages de performance

| Réglage | Effet |
|---|---|
| `XWI_MEMORY_BUDGET_GB` | budget mémoire par bloc (défaut ≈ 30 % de la RAM) ; plus grand = moins de halo recalculé |
| `XWI_DEVICE=cpu\|gpu\|auto` / `--device` | filtres sur GPU (pyclesperanto) si disponible |
| `spacing` | adapte les filtres à l'anisotropie (indispensable si z ≠ x,y) |
| `compute_in_2d` | filtres coupe par coupe (volumes très anisotropes, coupes indépendantes) |
| `normalize="percentile"` | robuste aux variations d'intensité entre acquisitions |
| `max_samples_per_class` | plafonne les pixels d'entraînement par classe (défaut 200 000) |
| `smooth` | lissage des probabilités avant décision (moins de bruit poivre et sel) |

Résultats de benchmark et limites : voir [docs/benchmarks.md](docs/benchmarks.md).

## Architecture

```
src/xwiclassifier/
├── backend.py            filtres n-D CPU/GPU (noyaux gaussiens façon vigra, morphologie)
├── blocks.py             découpage en blocs avec halo, budget mémoire
├── features/
│   ├── engine.py         FeatureSpec + FeatureEngine (catalogue ouvert, cache, threads)
│   ├── eigen.py          valeurs propres analytiques 2x2 / 3x3
│   ├── ilastik.py        reproduction des features ilastik
│   └── ipsdk.py          reproduction des features IPSDK Explorer
├── classifier/
│   ├── model.py          PixelClassifier (fit, predict par blocs, importances)
│   ├── sampler.py        échantillons d'entraînement par bloc (+ cache)
│   ├── estimators.py     RF / ExtraTrees / LightGBM / XGBoost
│   └── vigra_forest.py   lecture des forêts vigra et conversion scikit-learn
├── io/                   images, modèles (.joblib v2 et v0.1), ilastik (.ilp), IPSDK Explorer
├── nnunet/               extension nnU-Net (jeux de données, entraînement, prédiction par blocs)
├── cli.py                commande xwi
└── napari/               widgets et manifeste du plugin
```

## Tests

```bash
pip install -e ".[dev]"
pytest                                                     # cœur, ilastik, CLI
XWI_ILASTIK_TEST_PROJECT=mon_projet.ilp pytest            # + validation sur un vrai projet
```

Les tests du widget napari nécessitent `napari` et `pytest-qt`.

## Sécurité

Les modèles `.joblib` reposent sur pickle : ne chargez que des modèles de confiance.

## Feuille de route

- auto-context (deuxième étage entraîné sur les probabilités du premier) ;
- classification d'objets après segmentation ;
- dérivées gaussiennes sur GPU, export ONNX des forêts ;
- validation sur des projets ilastik 3D réels et sur des modèles IPSDK Explorer réels.

Licence BSD-3-Clause.
