# Benchmarks v0.1 → v0.2

Mesures réalisées le 23/11/2026 dans un environnement Linux aarch64 à **4 cœurs et
3 Go de RAM** (petite machine : les temps absolus seront meilleurs sur un poste
de travail). Script : `benchmarks/bench.py` ; données synthétiques
(`benchmarks/synth.py`) : fond texturé, blobs ellipsoïdaux, fils fins, flou et
bruit, 1250 voxels annotés éparpillés (600 fond, 400 blobs, 250 fils).

## Moteur de features seul (même jeu de 43 features, volume 48×192×192)

| | temps | pic RAM |
|---|---|---|
| v0.1 | 10,7 s | 0,77 Go |
| v0.2, 1 thread | 3,8 s (×2,9) | 0,81 Go |
| v0.2, 4 threads | 1,95 s (×5,5) | 1,67 Go |

## Chaîne complète (volume 48×192×192)

| | features | entraînement | prédiction | IoU blobs | IoU fils |
|---|---|---|---|---|---|
| v0.1 (défaut) | 43 | 11,6 s | 1,5 s | 0,69 | 0,38 |
| v0.2 (défaut, spacing 2,1,1) | 67 | 2,9 s | 0,9 s | 0,68 | 0,40 |

Sur une seule image, la qualité est équivalente (les écarts de ±0,02 d'IoU sont de
l'ordre de la variabilité de la forêt) ; le gain est surtout en vitesse.

## Application à une nouvelle acquisition (30 pixels aberrants « zingers »)

| | IoU blobs | IoU fils |
|---|---|---|
| v0.1 | **0,00** | **0,00** |
| v0.2 | 0,67 | 0,37 |

La normalisation min/max par image de la v0.1 écrase toutes les intensités dès
qu'un pixel aberrant apparaît : le modèle ne reconnaît plus rien. La v0.2 travaille
sur les intensités brutes (ou une normalisation par percentiles calculée sur
l'image entière).

## Gros volume dans peu de RAM (64×320×320, 6,5 M voxels, 3 Go de RAM)

| | temps total | pic RAM | IoU blobs | IoU fils |
|---|---|---|---|---|
| v0.1 | 45 s | 3,4 Go (limite) | 0,62 | 0,28 |
| v0.2 | 74 s | 3,1 Go | 0,64 | 0,33 |

Avec si peu de mémoire, la v0.2 découpe en blocs et recalcule des halos (σ=10 →
46 px) : elle est plus lente mais sa mémoire est bornée, alors que la v0.1 doit
tenir toutes les features en RAM (×2 avec la copie `np.stack`) et échoue au-delà.
Sur une machine de 16 Go ou plus, ce volume tient en un seul bloc et l'on
retrouve le gain de vitesse du tableau précédent. `XWI_MEMORY_BUDGET_GB`
permet d'ajuster le compromis.

## Reproduction d'un projet ilastik réel

Projet `pc.ilp` (ilastik 1.4.0b11, image 1024×1344, 49 features, 4 forêts de 25 arbres) :
100 % d'accord avec les 1132 pixels annotés, prédiction de l'image en 0,9 s.
Ré-entraînement sur les mêmes annotations : 99,2 % (features xwi) et 99,4 %
(features ilastik) d'accord avec la forêt ilastik sur toute l'image.
