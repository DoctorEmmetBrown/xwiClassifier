# Segmentation par apprentissage profond : extension nnU-Net

nnU-Net configure automatiquement son réseau (taille de fenêtre, profondeur,
espacement, augmentations) pour **chaque jeu de données** : un modèle par organe
ou structure (vaisseaux du rein, glomérules, alvéoles…), entraîné à la résolution
réelle de vos images.

## Installation (machines Linux avec GPU NVIDIA)

```bash
# 1. PyTorch adapté à votre CUDA (voir https://pytorch.org, ex. CUDA 12.1) :
pip install torch --index-url https://download.pytorch.org/whl/cu121
# 2. xwiClassifier avec l'extension nnU-Net :
pip install -e ".[nnunet,napari]"
xwi nnunet init            # affiche l'espace de travail et les GPU détectés
```

Sur Mac (Apple Silicon), l'entraînement est possible mais lent ; la prédiction
fonctionne sur le GPU Apple (`device="mps"`). Le mieux : entraîner sur les machines
à 2 GPU, puis copier le dossier du modèle pour prédire ailleurs.

Espace de travail : `~/xwi_nnunet` par défaut (`--root` ou `XWI_NNUNET_ROOT`),
avec les dossiers `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`.

## Flux de travail

1. **Un jeu de données par organe / structure**
   `xwi nnunet create "Rein vaisseaux" --classes "1=fond,2=vaisseaux"`
2. **Ajouter des données d'entraînement**, de deux façons combinables :
   - *annotations partielles* (traits, quelques coupes) : les voxels non annotés
     sont marqués `ignore`, nnU-Net ne les compte ni comme fond ni comme objet ;
   - *régions validées* : une segmentation (par ex. xwi Random Forest puis
     corrigée dans napari) entièrement vérifiée sur une région ; une marge de
     contexte autour est marquée `ignore`.
   ```bash
   xwi nnunet add 501 --image vol.tif --labels traits.tif --spacing 2 2 2
   xwi nnunet add 501 --image vol.tif --labels seg_corrigee.tif --mode dense --roi "100:228,0:1024,0:1024"
   xwi nnunet add 501 --from-ilastik projet.ilp        # ou --from-ipsdk DossierModele
   ```
3. **Préparer** : `xwi nnunet plan 501` (`--planner resenc_m` ou `resenc_l` pour
   les architectures ResEnc si vos GPU ont ≥ 10 ou ≥ 24 Go).
4. **Entraîner** : `xwi nnunet train 501 --epochs 250`
   - `--folds all` (défaut) : un modèle sur toutes les données ;
     `--folds 0 1 2 3 4` : validation croisée, les folds sont **répartis sur les
     2 GPU** (chacun enchaîne ses folds) ;
   - `--ddp` : un seul fold sur les 2 GPU ;
   - `--continue` reprend un entraînement interrompu ; `xwi nnunet status 501`
     affiche l'époque et le Dice ; journaux dans `<espace>/logs`.
5. **Prédire** (volume ou série, sans napari) :
   ```bash
   xwi predict ~/xwi_nnunet/nnUNet_results/Dataset501_ReinVaisseaux "serie/*.tif" -o seg/ --spacing 2 2 2
   ```
   En Python : `xwi.load_model(dossier_resultats).predict(volume)`.

Tout cela existe aussi dans napari : *Plugins → xwiClassifier → nnU-Net*
(ajout de la région affichée, entraînement en arrière-plan avec suivi, prédiction
de la coupe ou du volume).

## Choix techniques

- **Normalisation** calculée une fois sur le volume entier (z-score robuste,
  percentiles 0,5–99,5), à l'export comme à la prédiction. nnU-Net normalise
  sinon chaque image qu'on lui donne : en prédiction par blocs, chaque bloc serait
  normalisé différemment. Option `ct` : statistiques globales des données
  d'entraînement (intensités quantitatives).
- **Prédiction par blocs** avec une marge d'une demi-fenêtre nnU-Net : volumes de
  taille quelconque. Sur un test, labels identiques au calcul global sur 99,7 % des
  voxels (les écarts restants : voxels incertains, la fenêtre glissante ne tombant
  pas aux mêmes positions).
- **Entraîneur `nnUNetTrainerXwi`** (via `nnUNet_extTrainer`) : identique à
  nnU-Net, durée réglable (`--epochs`). 1000 époques = défaut nnU-Net ; 150–300
  suffisent souvent pour démarrer.
- **Labels** : xwi 1 = fond → nnU-Net 0, classes suivantes décalées de 1 ; 0 xwi
  (non annoté) → `ignore`.

## Conseils pour l'histologie virtuelle

- La fenêtre nnU-Net est en voxels : vérifiez dans `nnUNetPlans.json` qu'elle couvre
  assez de contexte (un glomérule entier, plusieurs calibres de vaisseaux). Sinon,
  sous-échantillonnez les volumes à l'export, ou utilisez `--planner resenc_l`.
- Mélangez des cas de plusieurs échantillons et acquisitions pour généraliser.
- Pour l'arbre vasculaire, privilégiez les régions validées denses (la connectivité
  s'apprend mieux que sur des traits isolés).

## Limites actuelles

Testé de bout en bout (création, préparation, entraînement, prédiction) sur CPU
avec de petits volumes synthétiques ; pas encore sur vos GPU ni sur vos données.
Les premiers entraînements réels permettront d'ajuster les réglages par défaut.
