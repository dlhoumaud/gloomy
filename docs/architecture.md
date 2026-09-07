# Couches et neurones

## Couche dense

Une `DenseLayer` relie chaque entrée à chaque neurone de sortie. Pour `I` entrées et `O` sorties, elle possède `I * O` poids et `O` biais. Chaque neurone reçoit toutes les valeurs de la couche précédente.

Le réseau construit :

```text
entrée (taille de la séquence)
  -> L couches cachées de N neurones
  -> sortie de 1 neurone
```

Avec `-l 0`, il n'y a pas de couche cachée : la séquence est directement reliée à la sortie. Avec `-l 2 -n 8`, il y a deux couches cachées de 8 neurones, puis la sortie.

## Neurones

`-n N` fixe la largeur de chaque couche cachée. Augmenter `N` donne plus de capacité pour représenter des relations complexes, mais augmente le nombre de paramètres et le risque de surajustement. Comme il n'y a pas d'entraînement ni de validation dans ce projet, augmenter `N` ne rend pas actuellement les prédictions plus justes.

Le nombre de paramètres des couches cachées dépend de la taille de la séquence et de `N`. Une séquence longue avec beaucoup de neurones peut rapidement produire un modèle inutilement grand.

## Limites actuelles

Les poids sont initialisés aléatoirement à chaque construction. La commande ne charge pas de poids appris, ne reçoit pas de cible et ne calcule pas de perte. Le programme est donc un démonstrateur de propagation avant, pas encore un modèle prédictif entraînable.

De plus, lors de plusieurs prédictions, la séquence grandit mais le réseau est recréé avec de nouveaux poids. Ce comportement est compatible avec la démonstration CLI, mais il empêche une extrapolation stable.
