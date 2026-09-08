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
- forgetting ;
- capacité de la mémoire d'apprentissage utilisée (`memory_capacity`, `0` quand la notion ne s'applique pas) ;
- ratio `mae / mae_du_scénario_full_dataset` pour le même optimiseur (`mae_ratio_to_full_dataset`, `0` quand aucune référence n'est disponible) ;
- seed utilisée pour l'initialisation des poids et le générateur de la mémoire (`seed`, `0` quand la notion ne s'applique pas) ;
- moyenne et écart-type du MAE sur l'ensemble des seeds d'un même scénario (`mae_mean`, `mae_stddev`, identiques sur chaque ligne du groupe, `0` quand un seul run existe).

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

Elle compare SGD, Momentum et Adam sur le même réseau et les mêmes données, d'abord avec le dataset complet (100 epochs, seed unique), puis avec FIFO, Reservoir, Prioritized, Novelty et Hybrid à quatre capacités (`32`, `64`, `128`, `256`), chacune répétée sur **3 seeds** (`1234`, `2345`, `3456`, pilotant à la fois l'initialisation des poids et le générateur de la mémoire d'apprentissage), et enfin FIFO int16 et FIFO int8 à capacité `16` (seed unique). Les scénarios bornés utilisent le mode online (un seul passage sur les données) et un replay de taille `8`. Le fichier `benchmark_results.csv` est produit à la racine et contient les pertes, MAE, RMSE, temps d'entraînement, latence moyenne d'inférence, mémoire des paramètres et état optimiseur, nombre d'échantillons, nombre d'updates, la capacité mémoire utilisée, le ratio au dataset complet, la seed et la moyenne/écart-type du MAE sur les 3 seeds.

Le runner ajoute aussi deux lignes d'expérience de catastrophic forgetting : `forgetting_no_replay` entraîne sur A puis B sans replay, tandis que `forgetting_fifo_replay` réentraîne périodiquement depuis la mémoire FIFO de A. Leur colonne `forgetting` suit la convention `performance_before - performance_after`; ici il s'agit d'une perte, donc une valeur négative signifie que la perte a augmenté après l'apprentissage de B.

**À propos de `mae_ratio_to_full_dataset`** : ce ratio compare le MAE de chaque scénario borné au MAE du scénario `full_dataset` du même optimiseur. Attention à son interprétation actuelle : `full_dataset` entraîne 100 epochs en batch sur le jeu complet (convergence quasi parfaite sur cette régression synthétique, MAE proche de zéro), alors que les scénarios bornés font un seul passage en apprentissage online. Le ratio observé (parfois plusieurs millions) mélange donc **deux effets distincts** : le nombre de passages sur les données et la capacité de mémoire. Isoler l'effet de la seule capacité (à nombre de passages égal) reste à faire — voir [feuille de route](roadmap.md).

Ce runner est une première baseline contrôlée. Plus de configurations de quantification (capacités multiples pour int16/int8, comme pour float64), plusieurs seeds pour le dataset complet et les scénarios quantifiés (déjà fait pour les 5 stratégies float64 bornées), des intervalles de confiance, une baseline « dernière valeur connue », et des campagnes plus larges restent encore à ajouter. Les poids du réseau restent en float64 ; les scénarios quantifiés mesurent uniquement la mémoire d'apprentissage. Les temps restent dépendants de la machine et ne doivent être comparés qu'à environnement constant.

Pour contrôler les valeurs numériques sans faux positif sur l'en-tête `inference_time_us`, vérifier les colonnes de données par motif plutôt que rechercher `inf` dans tout le fichier ou comparer une conversion arithmétique (`mawk`, l'implémentation par défaut d'`awk` sur beaucoup de systèmes Debian/Ubuntu, tronque silencieusement certains flottants en notation scientifique à forte précision, ex. `8.13...e-17` → `8` : une comparaison `$i != $i + 0` y déclenche alors un faux positif) :

```bash
awk -F, 'NR > 1 { for (i = 4; i <= 18; ++i) if ($i !~ /^-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$/) exit 1 }' benchmark_results.csv
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
