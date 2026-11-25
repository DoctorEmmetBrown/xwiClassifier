# Récupérer les modèles IPSDK Explorer

Analyse faite sur le code d'IPSDK Explorer 4.0.1 (module `moduleRandomForest`) et la
documentation IPSDK fournie avec l'installation.

## Ce qu'est un modèle « Smart Segmentation »

Un modèle est un **dossier**, sous Windows :

```
C:\Users\<vous>\AppData\Roaming\ReactivIP\Explorer\Machine_Learning\Pixel_Classification\<modèle>\
├── Settings.mho          XML : grille de features, classes, liste des images
├── Model                 forêt scikit-learn (RandomForestClassifier) picklée par joblib
├── ModelIPSDK.bin        la même forêt au format IPSDK (non utilisé par xwi)
└── <image>\Image.tif     image d'entraînement
    <image>\ImageDraw.tif annotations (0 = rien, k+1 = classe k)
```

IPSDK Explorer entraîne en réalité une `RandomForestClassifier` de **scikit-learn 0.24.1**
puis la recopie nœud par nœud dans le format IPSDK. xwi relit directement la forêt
scikit-learn (le `.bin` est inutile) sans dépendre de la version installée : un pickle
0.24 ne se charge plus tel quel dans scikit-learn ≥ 1.3 (format des nœuds modifié),
xwi capture l'état brut des arbres et les reconstruit.

## Utilisation

```bash
xwi inspect  "…/Pixel_Classification/MonModele"            # filtres, classes, images
xwi predict  "…/Pixel_Classification/MonModele" vol.tif -o seg.tif
xwi predict  "…/MonModele" vol.tif -o seg.tif --ipsdk-values   # mêmes valeurs de sortie qu'IPSDK
xwi convert  "…/MonModele" -o mon_modele.joblib            # modèle xwi autonome
xwi convert  "…/MonModele" -o reentraine.joblib --mode retrain   # ré-entraîné, features xwi
xwi ipsdk-export "…/MonModele" -o annotations/             # images + annotations en TIFF
```

En Python : `xwi.load_model("…/MonModele")`. Dans napari : *Dossier de modèle IPSDK
Explorer…* (utiliser la forêt, ou importer images + annotations pour continuer à annoter).

Copiez simplement le dossier du modèle depuis le PC Windows : seuls `Settings.mho`,
`Model` et les sous-dossiers d'images sont nécessaires.

## Les features IPSDK et leur équivalent xwi

Le vecteur de features IPSDK est : image originale, puis pour chaque résolution, chaque
filtre et chaque taille cochée, la réponse du filtre. La *taille* `s` est une demi-taille
de noyau (affichée « (2s+1)x(2s+1) »).

| Filtre IPSDK | Définition (doc IPSDK) | Famille xwi équivalente |
|---|---|---|
| Gaussian `s` | lissage gaussien σ = s, noyau couvrant 99,7 % (demi-taille ≥ 2) | `gaussian` σ = s |
| Mean `s` | moyenne sur une boîte (2s+1)^d | `mean` r = s |
| Laplacian Difference `s` | G(s) − G(1,6 s) | `dog` σ = s |
| High Pass `s` | noyau (2s+1)^d de −1, centre (2s+1)^d, bords miroir | `high_pass` r = s |
| Variance `s` | variance de population sur une boîte (2s+1)^d | `box_variance` r = s |
| Erosion / Dilation `s` | disque / boule x²+y²(+z²) ≤ s² | `erosion` / `dilation` r = s, `morpho_shape="exact"` |
| Opening / Closing `s` | dilatation∘érosion / érosion∘dilatation | `opening` / `closing` r = s, `morpho_shape="exact"` |
| Dimension 2D sur un volume | filtres coupe par coupe | `compute_in_2d=True` |
| MultiRes r | réduction linéaire ×1/r, filtre, agrandissement | (uniquement dans l'extracteur IPSDK) |

`mean`, `high_pass` et `box_variance` sont de nouvelles familles du moteur xwi ; avec
`morpho_shape="exact"`, la morphologie xwi est identique à celle d'IPSDK. Pour les modèles
importés, xwi utilise un extracteur dédié (`IPSDKFeatures`) qui reproduit la grille IPSDK
à l'identique (ordre des canaux, noms, bords, multi-résolution).

## Vérification et calibration

Deux détails ne sont pas documentés : la normalisation du noyau gaussien tronqué
(effet ≤ 0,3 %) et la convention de grille du zoom multi-résolution. À l'import, xwi
évalue chaque variante sur les pixels annotés stockés dans le dossier (la forêt doit les
reconnaître presque parfaitement si les features sont bien reproduites) et garde la
meilleure. `xwi info` affiche l'accord obtenu : un accord nettement inférieur à 100 %
signale un écart de reproduction à me remonter.

Limites : validé sur des dossiers de modèle construits au format exact d'IPSDK Explorer
4.0, pas encore sur un modèle produit par IPSDK lui-même. Les images couleur (3 canaux)
sont gérées mais moins testées que les images en niveaux de gris.
