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
