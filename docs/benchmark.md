# Benchmark

`BenchmarkResult` rassemble les mesures brutes nécessaires aux comparaisons expérimentales :

- expérience, stratégie mémoire, capacité et seed ;
- précision de stockage ;
- optimiseur ;
- fonction de perte ;
- perte d'entraînement et de validation ;
- MAE et RMSE ;
- moyenne, écart-type et intervalle de confiance à 95 % sur les seeds ;
- ratio MAE mémoire limitée / MAE dataset complet ;
- temps d'entraînement et latence d'inférence ;
- échantillons/sec et updates/sec ;
- mémoire utilisée en octets ;
- nombre d'échantillons conservés ;
- nombre de mises à jour ;
- forgetting.

`BenchmarkCsv::write()` exporte une collection de résultats dans un CSV. Les trois premières colonnes conservent l'ordre historique pour les outils qui lisent déjà ces fichiers :

```cpp
BenchmarkCsv::write("results.csv", results);
```

Les champs textuels sont échappés selon les règles CSV. Le runner implémenté dans `src/BenchmarkRunner.cpp` produit les scénarios décrits ci-dessous.

## Runner actuel

La commande suivante exécute une campagne à trois seeds sur une régression synthétique `y = 2x + 1` :

```bash
make benchmark
```

La matrice compare SGD, Momentum et Adam, les pertes MSE, MAE et Huber, et les capacités `32`, `64`, `128` et `256`. Elle couvre FIFO, Reservoir, Prioritized, Novelty, Hybrid et FIFO quantifié int16/int8. Le dataset contient `400` exemples d'entraînement et `80` de validation ; les scénarios utilisent des updates par batch de `8` exemples. Les lignes brutes sont suivies de lignes `mean` avec moyenne, écart-type et intervalle de confiance à 95 % sur les trois seeds. Chaque seed initialise le réseau et les générateurs des stratégies Reservoir, Prioritized et Hybrid; un même seed reproduit donc l'initialisation et les tirages de mémoire.

Une baseline `last_value_baseline` prédit la dernière valeur d'entrée. Pour chaque scénario à mémoire bornée, `mae_ratio_vs_full` donne `MAE du scénario / MAE full_dataset` sur le même seed, optimiseur et perte ; une valeur inférieure à `1` indique une MAE meilleure que la référence full dataset. `inference_time_us` est le temps moyen par exemple de la boucle d'évaluation, qui comprend prédiction, perte et métriques; ce n'est pas une mesure isolée du seul appel `forward()`. Les colonnes de débit rapportent les exemples et updates par seconde dans la boucle chronométrée. Les octets mémoire sont une estimation des échantillons stockés, des paramètres du réseau et de l'état de l'optimiseur; ils n'incluent pas le coût complet des conteneurs/allocations et ne représentent pas le pic RSS du processus.

Quatre fichiers sont produits à la racine : `benchmark_results.csv` rassemble la campagne, `benchmark_memory.csv` contient baseline et comparaisons mémoire, `benchmark_forgetting.csv` contient les expériences d'oubli, et `benchmark_drift.csv` contient le protocole A → B → A.

Le runner conserve deux lignes de comparaison A → B : `forgetting_no_replay` entraîne sur A puis B sans replay, tandis que `forgetting_fifo_replay` réentraîne périodiquement depuis la mémoire FIFO de A. Leur colonne `forgetting` suit la convention `perte_A_avant - perte_A_après`; une valeur négative signifie que la perte A a augmenté après l'apprentissage de B.

Le protocole `concept_drift_a_b_a` entraîne sur A, bascule sur B, puis revient sur A. Il compare `no_replay`, FIFO, Reservoir, Prioritized, Novelty et Hybrid sur trois seeds. Chaque phase reçoit dix passages et le même nombre d'updates (50) pour chaque stratégie. La mémoire a une capacité de 64. Le CSV donne la MAE sur A après la première phase A, après B et après le retour à A, ainsi que la MAE sur B après B et après le retour à A.

Pour ce protocole, `forgetting = MAE_A_après_A - MAE_A_après_B` : une valeur négative indique un oubli. `recovery = MAE_A_après_B - MAE_A_après_retour_A` : une valeur positive indique une récupération. Les lignes `mean` contiennent les moyennes, écarts-types et IC 95 % du forgetting et de la récupération sur les seeds. Ces résultats restent indicatifs : le scénario est synthétique et la dérive est une bascule nette entre deux fonctions.

Cette campagne reste synthétique et sert à détecter des tendances, pas à établir un classement universel. Les poids du réseau restent en float64; les scénarios int16/int8 quantifient uniquement les vecteurs de la mémoire FIFO. Les temps et débits dépendent de la machine et ne doivent être comparés qu'à environnement constant. La campagne ne mesure ni pic RSS, ni énergie, ni percentiles de latence, ni coût CPU théorique ou performances sur matériel embarqué.

## Baseline obligatoire

Chaque campagne devrait inclure une référence `full_dataset`, puis comparer les budgets bornés :

Le ratio MAE limité/full est calculé directement pour chaque stratégie, seed, optimiseur et perte.

Les résultats bruts restent la référence pour éviter de masquer les compromis. Les prochains benchmarks utiles pour la priorité latence/CPU devront ajouter des workloads représentatifs et des mesures sur cible, en conservant séparément les métriques de qualité, mémoire, temps et débit. Les précisions `float32` pour les poids et `int8` pour les poids/calcul ne sont pas encore disponibles.
