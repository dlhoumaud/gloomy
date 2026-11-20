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

`BenchmarkCsv::write()` exporte une collection de résultats dans un CSV. Les trois premières colonnes conservent l'ordre historique pour les outils qui lisent déjà ce fichier :

```cpp
BenchmarkCsv::write("results.csv", results);
```

Les champs textuels sont échappés selon les règles CSV. Le composant n'exécute pas encore automatiquement les configurations : il fournit le format de sortie commun que le futur runner pourra remplir.

## Runner actuel

La commande suivante exécute une campagne à trois seeds sur une régression synthétique `y = 2x + 1` :

```bash
make benchmark
```

La matrice compare SGD, Momentum et Adam, les pertes MSE, MAE et Huber, et les capacités `32`, `64`, `128` et `256`. Elle couvre FIFO, Reservoir, Prioritized, Novelty, Hybrid et FIFO quantifié int16/int8. Le dataset contient `400` exemples d'entraînement et `80` de validation ; les scénarios utilisent des updates par batch de `8` exemples. Les lignes brutes sont suivies de lignes `mean` avec moyenne, écart-type et intervalle de confiance à 95 % sur les trois seeds.

Une baseline `last_value_baseline` prédit la dernière valeur d'entrée. Pour chaque scénario à mémoire bornée, `mae_ratio_vs_full` donne `MAE du scénario / MAE full_dataset` sur le même seed, optimiseur et perte ; une valeur inférieure à `1` indique une MAE meilleure que la référence full dataset. Les colonnes de débit rapportent les exemples et updates par seconde. Les octets mémoire incluent une estimation des échantillons, des paramètres du réseau et de l'état de l'optimiseur ; ils ne représentent pas le pic RSS du processus.

Trois fichiers sont produits à la racine : `benchmark_results.csv` rassemble la campagne, `benchmark_memory.csv` contient baseline et comparaisons mémoire, et `benchmark_forgetting.csv` contient l'expérience A → B avec/sans replay.

Le runner ajoute aussi deux lignes d'expérience de catastrophic forgetting : `forgetting_no_replay` entraîne sur A puis B sans replay, tandis que `forgetting_fifo_replay` réentraîne périodiquement depuis la mémoire FIFO de A. Leur colonne `forgetting` suit la convention `performance_before - performance_after`; ici il s'agit d'une perte, donc une valeur négative signifie que la perte a augmenté après l'apprentissage de B.

Cette campagne reste synthétique et sert à détecter des tendances, pas à établir un classement universel. Les poids du réseau restent en float64 ; les scénarios int16/int8 quantifient uniquement la mémoire d'apprentissage. Les temps et débits dépendent de la machine et ne doivent être comparés qu'à environnement constant.

Pour contrôler les valeurs numériques sans faux positif sur l'en-tête `inference_time_us`, vérifier les colonnes de données plutôt que rechercher `inf` dans tout le fichier :

```bash
awk -F, 'NR > 1 { for (i = 6; i <= 26; ++i) if (i != 7 && ($i == "" || $i != $i + 0)) exit 1 }' benchmark_results.csv
```

## Baseline obligatoire

Chaque campagne devrait inclure une référence `full_dataset`, puis comparer les budgets bornés :

Le ratio MAE limité/full est calculé directement pour chaque stratégie, seed, optimiseur et perte.

Les résultats doivent aussi distinguer `float32`, `int16` et `int8` lorsque ces formats seront disponibles. La métrique composite performance/mémoire/CPU pourra être étudiée plus tard, mais les colonnes brutes doivent rester la référence pour éviter de masquer les compromis.
