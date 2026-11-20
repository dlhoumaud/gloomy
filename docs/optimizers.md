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

Le runtime configuré sélectionne `sgd`, `momentum` ou `adam` via `optimizer`. Les coûts et la qualité dépendent du jeu de données et du nombre d'updates; aucun optimiseur n'est le meilleur par défaut.

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

## Sauvegarde et restauration

`OptimizerSerialization` sauvegarde le type, les hyperparamètres et l'état courant de SGD, Momentum ou Adam dans un checkpoint binaire versionné avec checksum.

```cpp
OptimizerSerialization::save("optimizer.bin", optimizer);
OptimizerSerialization::load("optimizer.bin", optimizer, network.layers());
```

Le chargement restaure aussi les hyperparamètres. Le type du checkpoint doit correspondre à l'instance fournie, et les dimensions des buffers doivent correspondre aux couches du réseau. Les fichiers tronqués, corrompus ou contenant des valeurs non finies sont refusés.

Un checkpoint pris avant le premier update contient des buffers vides; au premier update, l'optimiseur initialise son état. `OptimizerSerialization` permet de sauvegarder séparément l'optimiseur, tandis que `GLOOMY_MODEL` regroupe le réseau, le normaliseur et l'état de l'optimiseur (ainsi qu'une mémoire optionnelle). Pour charger le format séparé, restaurer d'abord le réseau et utiliser une instance du même type d'optimiseur.
