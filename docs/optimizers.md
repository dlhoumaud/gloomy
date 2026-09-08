# Optimiseurs

## SGD

`SGDOptimizer` applique :

```text
parameter = parameter - learning_rate * gradient
```

Il ne conserve pas d'état supplémentaire par paramètre. C'est le choix de référence pour l'embarqué lorsque la RAM est prioritaire.

## SGD avec momentum

`MomentumOptimizer` conserve une vitesse par poids et par biais :

```text
velocity = momentum * velocity + gradient
parameter = parameter - learning_rate * velocity
```

Le paramètre `momentum` doit être dans `[0, 1)`, avec `0.9` comme valeur de départ classique. L'état du momentum consomme un `double` supplémentaire par paramètre, donc environ la même mémoire que les paramètres float64 eux-mêmes.

```cpp
MomentumOptimizer optimizer(0.01, 0.9);
LearningEngine engine(network, loss, optimizer);
```

Momentum peut accélérer la convergence et lisser les directions bruyantes, mais il n'est pas automatiquement supérieur à SGD sous une contrainte RAM stricte. La méthode `stateBytes()` permet de mesurer son coût.

## Comparaison expérimentale

Comparer au minimum :

- perte finale ;
- nombre d'updates ;
- temps d'entraînement ;
- stabilité en online learning ;
- mémoire des paramètres ;
- mémoire d'état de l'optimiseur.

## Adam

`AdamOptimizer` combine un premier moment, un second moment et une correction de biais :

```text
m = beta1 * m + (1 - beta1) * gradient
v = beta2 * v + (1 - beta2) * gradient^2
parameter -= learning_rate * corrected_m / (sqrt(corrected_v) + epsilon)
```

Il conserve deux `double` par paramètre, soit environ deux fois le coût d'état de Momentum. Adam peut être plus rapide sur certains problèmes, mais sa mémoire et ses opérations supplémentaires le rendent moins évident pour un microcontrôleur. `stateBytes()` permet de comparer ce coût directement.

Adam doit être comparé à SGD et Momentum sur les mêmes données, seeds, nombre d'updates et budgets mémoire. Il ne doit pas être considéré comme supérieur par défaut.

## Persistance de l'état

`OptimizerSerialization` sauvegarde et restaure n'importe quel `Optimizer` concret (SGD, Momentum ou Adam) dans un fichier binaire versionné, protégé par un checksum FNV-1a :

```cpp
OptimizerSerialization::save("optimizer.bin", optimizer);
std::unique_ptr<Optimizer> restored =
    OptimizerSerialization::load("optimizer.bin", network.layers());
```

Le fichier contient le type d'optimiseur, ses hyperparamètres (`learning_rate`, `momentum`, `beta1`/`beta2`/`epsilon`), le compte d'updates pour Adam, ainsi que les vitesses de Momentum ou les premiers et seconds moments d'Adam. `load()` reconstruit l'optimiseur concret et vérifie que la forme de chaque couche sauvegardée correspond exactement au réseau fourni (`network.layers()`) avant de restaurer le buffer : un fichier taillé pour une autre architecture, tronqué, corrompu ou de version incompatible est rejeté avec une exception plutôt que de restaurer un état invalide.

Cette persistance reste utile comme export séparé du réseau et de la mémoire d'apprentissage, mais elle est aussi intégrée dans le format unifié `GLOOMY_MODEL` via `ModelSerialization` (voir [feuille de route](roadmap.md)). Le runtime online peut désormais sauvegarder l'état complet d'un apprentissage dans un seul fichier lorsqu'un `model_path` est fourni.
