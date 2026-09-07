# Fonctions de perte

Toutes les pertes implémentent l'abstraction `LossFunction` avec `compute(prediction, target)` et `gradient(prediction, target)`.

## MSE

```text
MSE = moyenne((prediction - target)^2)
dMSE/dprediction = 2 * (prediction - target) / N
```

MSE pénalise fortement les grandes erreurs. Elle est adaptée comme baseline de régression, mais une valeur aberrante peut dominer le batch.

## MAE

```text
MAE = moyenne(abs(prediction - target))
dMAE/dprediction = sign(prediction - target) / N
```

MAE est plus robuste aux valeurs aberrantes. Au point exact `prediction = target`, cette implémentation utilise un gradient nul, car la dérivée n'est pas unique.

## Huber

Avec un seuil `delta` :

```text
0.5 * erreur^2                         si abs(erreur) <= delta
 delta * (abs(erreur) - 0.5 * delta)  sinon
```

Huber est quadratique près de la cible et linéaire pour les grandes erreurs. Elle fournit un compromis entre la précision locale de MSE et la robustesse de MAE.

```cpp
HuberLoss loss(1.0);
LearningEngine engine(network, loss, optimizer);
```

Les trois pertes sont indépendantes du réseau, de l'optimiseur et de la mémoire. Il est donc possible de comparer leurs performances dans le benchmark sans modifier l'architecture.
