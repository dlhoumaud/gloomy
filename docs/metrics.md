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

`Metrics::forgetting` applique la formule générique ci-dessus. Le benchmark régression utilise une perte, où plus petit est meilleur; il conserve la convention `avant - après`, donc une valeur négative signifie que la perte a augmenté (oubli accru). Dans le protocole A → B → A, la récupération est définie comme `MAE_A_après_B - MAE_A_après_retour_A`; une valeur positive indique une amélioration après le retour à A.

Une expérience de catastrophic forgetting mesure séparément :

1. la performance sur le régime A avant l'apprentissage de B ;
2. la performance sur A après l'apprentissage de B ;
3. la performance sur A après replay depuis la mémoire.

Les métriques ne connaissent ni le réseau ni la stratégie de mémoire. Le runner actuel les utilise pour comparer les stratégies de mémoire sur ses scénarios synthétiques. Ces résultats ne remplacent pas une validation sur des données représentatives.
