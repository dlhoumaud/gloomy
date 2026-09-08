# Benchmark

`BenchmarkResult` rassemble les mesures brutes nécessaires aux comparaisons expérimentales :

- stratégie mémoire, précision de stockage, optimiseur, fonction de perte (`loss_function`, `"mse"`/`"mae"`/`"huber"`, vide quand la notion ne s'applique pas) ;
- perte d'entraînement et de validation ;
- MAE et RMSE ;
- temps d'entraînement et latence d'inférence ;
- mémoire utilisée en octets ;
- nombre d'échantillons conservés ;
- nombre de mises à jour ;
- forgetting ;
- capacité de la mémoire d'apprentissage utilisée (`memory_capacity`, `0` quand la notion ne s'applique pas) ;
- ratio `mae / mae_du_scénario_full_dataset` pour le même optimiseur et la même perte (`mae_ratio_to_full_dataset`, `0` quand aucune référence n'est disponible) ;
- seed utilisée pour l'initialisation des poids et le générateur de la mémoire (`seed`, `0` quand la notion ne s'applique pas) ;
- moyenne et écart-type (population) du MAE sur l'ensemble des seeds d'un même scénario (`mae_mean`, `mae_stddev`, identiques sur chaque ligne du groupe, `0` quand un seul run existe) ;
- demi-largeur de l'intervalle de confiance à 95% sur `mae_mean` (`mae_ci95_margin`, loi de Student, `0` quand un seul run existe ou que le nombre de seeds n'est pas géré) ;
- nombre approximatif d'opérations multiplication-accumulation pour la procédure d'entraînement (`approximate_macs`, grossier par construction) ;
- débit mesuré (`samples_per_second`, `updates_per_second`, `0` quand `training_time_ms` vaut `0`).

`BenchmarkCsv::write()` exporte une collection de résultats dans un CSV avec en-tête stable :

```cpp
BenchmarkCsv::write("results.csv", results);
```

Les champs textuels sont échappés selon les règles CSV.

## Runner actuel

La commande suivante exécute un benchmark déterministe sur une régression synthétique `y = 2x + 1` :

```bash
make benchmark
```

Elle calcule d'abord une **baseline naïve** `baseline_last_value` — prédire pour toute la validation la cible du dernier échantillon d'entraînement connu, sans aucun apprentissage, évaluée avec les 3 pertes — puis compare SGD, Momentum et Adam, chacun avec MSE, MAE et Huber, sur le même réseau et les mêmes données :

- **dataset complet** (100 epochs, seed unique) : 3 optimiseurs × 3 pertes = 9 scénarios ;
- **FIFO, Reservoir, Prioritized, Novelty, Hybrid** à quatre capacités (`32`, `64`, `128`, `256`), chacune répétée sur **3 seeds** (`1234`, `2345`, `3456`, pilotant à la fois l'initialisation des poids et le générateur de la mémoire d'apprentissage) : 4 × 5 × 3 optimiseurs × 3 pertes × 3 seeds = 540 scénarios ;
- **FIFO int16 et FIFO int8** à capacité `16` (seed unique) : 2 × 3 optimiseurs × 3 pertes = 18 scénarios ;
- deux scénarios de **catastrophic forgetting** (voir plus bas), seed unique, perte MSE uniquement.

Les scénarios bornés utilisent le mode online (un seul passage sur les données) et un replay de taille `8`. L'ensemble tourne en moins d'une seconde sur une machine de développement courante.

### Fichiers produits

Un fichier CSV séparé par expérience, plus un fichier combiné pour la compatibilité avec les usages existants :

| Fichier | Contenu |
| --- | --- |
| `benchmark_baseline.csv` | `baseline_last_value`, 3 lignes (une par perte) |
| `benchmark_full_dataset.csv` | dataset complet, 9 lignes |
| `benchmark_memory_capacity.csv` | balayage capacités × stratégies × optimiseurs × pertes × seeds, 540 lignes |
| `benchmark_quantization.csv` | FIFO int16/int8, 18 lignes |
| `benchmark_forgetting.csv` | les deux scénarios de forgetting |
| `benchmark_results.csv` | tout ce qui précède, concaténé, dans cet ordre |

**`baseline_last_value`** sert de plancher de comparaison : tout modèle entraîné doit au moins faire mieux. Sur cette régression synthétique, elle atteint un MAE d'environ `2.1` avec MSE/Huber (les 3 pertes évaluent le même prédicteur constant, donc `mae`/`rmse` sont identiques d'une ligne à l'autre ; seule `validation_loss`, calculée avec la perte correspondante, diffère). `training_loss`, `training_time_ms`, `memory_capacity`, etc. valent `0` — cette ligne n'entraîne rien. Certains scénarios bornés à faible capacité peuvent s'avérer plus proches de cette baseline que du dataset complet, ce qui est en soi une information utile sur le coût réel d'une mémoire trop petite.

Le runner ajoute aussi deux lignes d'expérience de catastrophic forgetting : `forgetting_no_replay` entraîne sur A puis B sans replay, tandis que `forgetting_fifo_replay` réentraîne périodiquement depuis la mémoire FIFO de A. Leur colonne `forgetting` suit la convention `performance_before - performance_after`; ici il s'agit d'une perte, donc une valeur négative signifie que la perte a augmenté après l'apprentissage de B.

**À propos de `mae_ratio_to_full_dataset`** : ce ratio compare le MAE de chaque scénario borné au MAE du scénario `full_dataset` du même optimiseur et de la même perte. Attention à son interprétation actuelle : `full_dataset` entraîne 100 epochs en batch sur le jeu complet (convergence quasi parfaite sur cette régression synthétique, MAE proche de zéro), alors que les scénarios bornés font un seul passage en apprentissage online. Le ratio observé (parfois plusieurs millions) mélange donc **deux effets distincts** : le nombre de passages sur les données et la capacité de mémoire. Isoler l'effet de la seule capacité (à nombre de passages égal) reste à faire — voir [feuille de route](roadmap.md).

**À propos de `mae_ci95_margin`** : calculée via la loi de Student pour `df = 2` (3 seeds), avec l'écart-type d'échantillon (diviseur `N-1`), comme l'exige la formule — pas l'écart-type population exposé par `mae_stddev`. Reste à `0` si le nombre de seeds change un jour sans mettre à jour la valeur critique codée en dur dans `BenchmarkRunner.cpp` (`t_critical_95_df2`), plutôt que d'utiliser silencieusement une valeur fausse. Avec seulement 3 points, cet intervalle reste une approximation grossière (l'hypothèse de normalité sous-jacente à Student est difficile à justifier sur un échantillon aussi petit) ; à lire comme un ordre de grandeur, pas comme une garantie statistique forte.

**À propos de `approximate_macs`** : nombre de MAC pour un forward pass (`entrées × sorties` par couche dense), multiplié par `3` (règle empirique forward + backward-par-rapport-aux-entrées + backward-par-rapport-aux-poids) et par le nombre d'échantillons traités. Ignore le coût de l'optimiseur (différent entre SGD, Momentum et Adam) et celui des fonctions d'activation. À utiliser pour comparer des ordres de grandeur entre scénarios, pas comme un compte cycle-exact.

Ce runner est une première baseline contrôlée. Un même seed unique pour le dataset complet et les scénarios quantifiés (au lieu des 3 seeds du balayage de capacités), une baseline `float32`, et des campagnes plus larges (jeux de données réels, réseaux plus profonds) restent à ajouter. Les poids du réseau restent en float64 ; les scénarios quantifiés mesurent uniquement la mémoire d'apprentissage. Les temps restent dépendants de la machine et ne doivent être comparés qu'à environnement constant.

Pour contrôler les valeurs numériques sans faux positif sur l'en-tête `inference_time_us`, vérifier les colonnes de données par motif plutôt que rechercher `inf` dans tout le fichier ou comparer une conversion arithmétique (`mawk`, l'implémentation par défaut d'`awk` sur beaucoup de systèmes Debian/Ubuntu, tronque silencieusement certains flottants en notation scientifique à forte précision, ex. `8.13...e-17` → `8` : une comparaison `$i != $i + 0` y déclenche alors un faux positif). La colonne 4 (`loss_function`) est textuelle, la vérification numérique démarre donc à la colonne 5 :

```bash
awk -F, 'NR > 1 { for (i = 5; i <= 23; ++i) if ($i !~ /^-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$/) exit 1 }' benchmark_results.csv
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
