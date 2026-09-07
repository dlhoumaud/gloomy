# Métriques

## Régression

`Metrics::regression()` retourne deux métriques brutes :

```text
MAE  = moyenne(abs(prediction - cible))
RMSE = sqrt(moyenne((prediction - cible)^2))
```

La MAE est plus robuste aux valeurs aberrantes. La RMSE pénalise davantage les grandes erreurs. Les deux vecteurs doivent être non vides et avoir la même dimension.

```cpp
const RegressionMetrics metrics =
    Metrics::regression(prediction, target);
```

## Forgetting

`Metrics::forgetting(performance_before, performance_after)` calcule :

```text
forgetting = performance_before - performance_after
```

Cette convention suppose que la performance est une valeur où plus grand est meilleur, par exemple une exactitude. Pour une perte où plus petit est meilleur, il faut conserver les pertes brutes et définir explicitement la convention d'analyse afin de ne pas inverser l'interprétation.

Une expérience de catastrophic forgetting doit mesurer séparément :

1. la performance sur le régime A avant l'apprentissage de B ;
2. la performance sur A après l'apprentissage de B ;
3. la performance sur A après replay depuis la mémoire.

Les métriques ne connaissent ni le réseau ni la stratégie de mémoire. Elles peuvent donc être utilisées par un futur benchmark pour comparer FIFO, Reservoir, Prioritized, Novelty, Hybrid et les budgets mémoire quantifiés.
