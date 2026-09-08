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

## Initialisation des poids et reproductibilité

Les poids de chaque `DenseLayer` sont initialisés aléatoirement dans `[-0.5, 0.5]` via un générateur `std::mt19937` partagé par toutes les couches (et non plus `rand()`/`RAND_MAX`). `DenseLayer::seedWeightInitialization(seed)` fixe ce générateur avant de construire des couches, pour une initialisation reproductible.

Le CLI (`bin/gloomy`) l'appelle automatiquement avec la clé `seed` de `GloomyConfig` (par défaut `5489`, le seed par défaut de `std::mt19937` lui-même) une fois la configuration résolue (défauts, puis fichier `-f`/`--config`) : à seed égal, deux exécutions identiques produisent des poids, prédictions et courbes d'apprentissage strictement identiques. `make benchmark` fixe de son côté un seed `1234` indépendant, pour des résultats reproductibles d'un run à l'autre (hors mesures de temps, qui restent dépendantes de la machine). Sans appel explicite, le générateur reste initialisé une fois via `std::random_device` (comportement non déterministe, comme avant ce changement).

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

## Robustesse : gradient checking, stabilité et valeurs non finies

Le gradient checking (comparaison entre le gradient analytique et une différence centrée numérique) couvre maintenant l'ensemble du réseau — plusieurs couches, chaque activation entraînable (`none`, `sigmoid`, `relu`, `leaky_relu`, `tanh`) — et le cas multi-sortie avec `softmax`, y compris le terme croisé de son gradient (jamais exercé par un réseau à une seule sortie). Voir `checkNetworkGradient` dans `tests/loss_tests.cpp`.

Sigmoid, tanh, ReLU et Leaky ReLU restent finis pour des entrées très grandes (`±1e8` testé) : sigmoid et tanh saturent proprement vers leurs bornes sans produire de `NaN`, et ReLU/Leaky ReLU croissent linéairement sans dépassement pour des profondeurs raisonnables. `softmax` soustrayait déjà le maximum avant l'exponentielle pour éviter les débordements.

`LossFunction::compute`/`gradient` et `DenseLayer::forward`/`backward` rejettent maintenant explicitement toute valeur non finie (`NaN`/infini) en entrée, avec un message clair, plutôt que de laisser une valeur invalide se propager silencieusement dans le calcul. Cette validation reste focalisée sur les frontières d'API les plus exposées (perte, couche) ; les gradients accumulés par lot dans `Optimizer` et les échantillons stockés en mémoire d'apprentissage ne sont pas encore vérifiés de la même façon (voir [feuille de route](roadmap.md), « Priorité moyenne : robustesse mathématique »).
