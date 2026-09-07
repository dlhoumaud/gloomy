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

`-n N` fixe la largeur de chaque couche cachée. Augmenter `N` donne plus de capacité pour représenter des relations complexes, mais augmente le nombre de paramètres et le risque de surajustement. Le projet possède maintenant un entraînement MSE + SGD via `LearningEngine`, mais aucune validation automatique ni régularisation.

Le nombre de paramètres des couches cachées dépend de la taille de la séquence et de `N`. Une séquence longue avec beaucoup de neurones peut rapidement produire un modèle inutilement grand.

## Limites actuelles

Les classes C++ d'entraînement savent recevoir des cibles, calculer une perte, propager les gradients et mettre à jour les paramètres avec SGD. Le CLI ne propose toutefois pas encore de commande pour charger un dataset et entraîner un modèle.

Une mémoire FIFO bornée existe également comme composant indépendant, mais elle n'est pas encore orchestrée automatiquement par le `LearningEngine`.

## Normalisation streaming

`StreamingNormalizer` conserve séparément les statistiques des entrées : nombre d'observations, moyenne, variance, minimum et maximum. La moyenne et la variance sont mises à jour avec l'algorithme de Welford, sans conserver le dataset complet.

```cpp
StreamingNormalizer normalizer(input_dimensions);
normalizer.update(training_input);
const std::vector<double> normalized = normalizer.normalize(input);
```

Les statistiques doivent être apprises sur le flux d'entraînement uniquement. Elles ne doivent pas être recalculées avec les données de validation ou de test, afin d'éviter une fuite d'information. Les dimensions sont fixes et les valeurs non finies sont refusées.

De plus, lors de plusieurs prédictions, la séquence grandit mais le réseau est recréé avec de nouveaux poids. Ce comportement est compatible avec la démonstration CLI, mais il empêche une extrapolation stable.
