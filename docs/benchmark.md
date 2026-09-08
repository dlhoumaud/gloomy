# Benchmark

`BenchmarkResult` rassemble les mesures brutes nécessaires aux comparaisons expérimentales :

- stratégie mémoire ;
- précision de stockage ;
- optimiseur ;
- perte d'entraînement et de validation ;
- MAE et RMSE ;
- temps d'entraînement et latence d'inférence ;
- mémoire utilisée en octets ;
- nombre d'échantillons conservés ;
- nombre de mises à jour ;
- forgetting.

`BenchmarkCsv::write()` exporte une collection de résultats dans un CSV avec en-tête stable :

```cpp
BenchmarkCsv::write("results.csv", results);
```

Les champs textuels sont échappés selon les règles CSV. Le composant n'exécute pas encore automatiquement les configurations : il fournit le format de sortie commun que le futur runner pourra remplir.

## Runner actuel

La commande suivante exécute un benchmark déterministe sur une régression synthétique `y = 2x + 1` :

```bash
make benchmark
```

Elle compare SGD, Momentum et Adam sur le même réseau et les mêmes données, d'abord avec le dataset complet, puis avec FIFO, Reservoir, Prioritized, Novelty, Hybrid, FIFO int16 et FIFO int8 à capacité `16`. Les scénarios bornés utilisent le mode online et un replay de taille `8`. Le fichier `benchmark_results.csv` est produit à la racine et contient les pertes, MAE, RMSE, temps d'entraînement, latence moyenne d'inférence, mémoire des paramètres et état optimiseur, nombre d'échantillons et nombre d'updates.

Le runner ajoute aussi deux lignes d'expérience de catastrophic forgetting : `forgetting_no_replay` entraîne sur A puis B sans replay, tandis que `forgetting_fifo_replay` réentraîne périodiquement depuis la mémoire FIFO de A. Leur colonne `forgetting` suit la convention `performance_before - performance_after`; ici il s'agit d'une perte, donc une valeur négative signifie que la perte a augmenté après l'apprentissage de B.

Ce runner est une première baseline contrôlée. Plusieurs capacités mémoire, plus de configurations de quantification et des campagnes plus larges restent encore à ajouter à la campagne. Les poids du réseau restent en float64 ; les scénarios quantifiés mesurent uniquement la mémoire d'apprentissage. Les temps restent dépendants de la machine et ne doivent être comparés qu'à environnement constant.

Pour contrôler les valeurs numériques sans faux positif sur l'en-tête `inference_time_us`, vérifier les colonnes de données plutôt que rechercher `inf` dans tout le fichier :

```bash
awk -F, 'NR > 1 { for (i = 4; i <= 13; ++i) if ($i == "" || $i != $i + 0) exit 1 }' benchmark_results.csv
```

## Baseline obligatoire

Chaque campagne devrait inclure une référence `full_dataset`, puis comparer les budgets bornés :

```text
full_dataset
256 samples
128 samples
64 samples
32 samples
```

Les résultats doivent aussi distinguer `float32`, `int16` et `int8` lorsque ces formats seront disponibles. La métrique composite performance/mémoire/CPU pourra être étudiée plus tard, mais les colonnes brutes doivent rester la référence pour éviter de masquer les compromis.
