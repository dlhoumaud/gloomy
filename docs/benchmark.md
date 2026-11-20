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

Une baseline `last_value_baseline` prédit la dernière valeur d'entrée. Pour chaque scénario à mémoire bornée, `mae_ratio_vs_full` donne `MAE du scénario / MAE full_dataset` sur le même seed, optimiseur et perte ; une valeur inférieure à `1` indique une MAE meilleure que la référence full dataset. `inference_time_us` est le temps moyen par exemple de la boucle d'évaluation, qui comprend prédiction, perte et métriques; ce n'est pas une mesure isolée du seul appel `forward()`. Les colonnes de débit rapportent les exemples et updates par seconde dans la boucle chronométrée. `memory_used_bytes` estime les allocations conservées par la stratégie mémoire, le réseau, l'optimiseur et les buffers du `LearningEngine`. Cette estimation exclut le dataset complet, les métadonnées de l'allocateur et les autres allocations du processus. `memoryUsedBytes()` des mémoires quantifiées conserve son sens historique de taille logique des échantillons encodés.

Les CSV de la campagne sont produits à la racine : `benchmark_results.csv` rassemble les résultats, `benchmark_memory.csv` contient baseline et comparaisons mémoire, `benchmark_forgetting.csv` contient les expériences d'oubli, et `benchmark_drift.csv` contient le protocole A → B → A. `benchmark_profile.csv` rapporte la latence moyenne, p50, p95, les allocations C++ par opération et `process_peak_rss_bytes` pour le processus de benchmark, sur les opérations `forward()`, l'inférence sans cache, le gradient de perte, un update batch, un replay mémoire et une copie de mini-batch du réseau Dense `1 → 8 → 1`. Le RSS est le maximum observé par le système d'exploitation pour le processus; il vaut zéro si la plateforme ne fournit pas cette mesure. Le décompte d'allocations mesure les appels ordinaires aux opérateurs globaux `new`/`new[]`; il ne couvre pas les allocations alignées ou internes à des bibliothèques C. Les percentiles décrivent ce workload synthétique et cette machine.

`benchmark_artifacts.csv` compare la taille fichier et le temps moyen de chargement sur 100 itérations pour un checkpoint `GLOOMY_MODEL` (Adam, sans mémoire) et un artefact `GLOOMY_INFER`, sur le réseau Dense synthétique du benchmark. La mesure sert à suivre le coût du format, pas à prédire les performances d'un modèle applicatif.

Sur le workload actuel, les optimisations du chemin d'entraînement ont fait passer les allocations observées par opération de `72` à `0` pour l'update batch et de `129` à `0` pour le replay FIFO; l'inférence utilisant les buffers réutilisables et le calcul du gradient de perte rapportent également `0` allocation après échauffement. `forward()` par valeur conserve une allocation pour sa copie de sortie. Le dernier profil après contiguïsation a mesuré environ `0,137 µs` pour `forward()`, `0,112 µs` pour l'inférence sans cache, `0,142 µs` pour le gradient de perte, `1,48 µs` pour l'update batch et `2,53 µs` pour le replay. Les temps sont indicatifs et doivent être reconfirmés sur plusieurs campagnes et workloads applicatifs. Le benchmark ne mesure pas la qualité avant/après dans un protocole statistique dédié.

Le runner conserve deux lignes de comparaison A → B : `forgetting_no_replay` entraîne sur A puis B sans replay, tandis que `forgetting_fifo_replay` réentraîne périodiquement depuis la mémoire FIFO de A. Leur colonne `forgetting` suit la convention `perte_A_avant - perte_A_après`; une valeur négative signifie que la perte A a augmenté après l'apprentissage de B.

Le protocole `concept_drift_a_b_a` entraîne sur A, bascule sur B, puis revient sur A. Il compare `no_replay`, FIFO, Reservoir, Prioritized, Novelty et Hybrid sur trois seeds. Chaque phase reçoit dix passages et le même nombre d'updates (50) pour chaque stratégie. La mémoire a une capacité de 64. Le CSV donne la MAE sur A après la première phase A, après B et après le retour à A, ainsi que la MAE sur B après B et après le retour à A.

Le protocole `benchmark_progressive_drift.csv` ajoute trois régimes linéaires distincts A, B et C, séparés par des transitions dont les cibles interpolent progressivement entre les deux fonctions. L'ordre est A → transition A/B → B → transition B/C → C → retour à A. Il compare `no_replay`, FIFO, Reservoir, Prioritized, Novelty et Hybrid avec trois seeds, 40 observations par régime, dix passages et 50 updates par étape; chaque mémoire a une capacité de 64. Chaque ligne contient la MAE mesurée sur les trois régimes après l'étape, son temps d'entraînement et l'estimation mémoire. Hybrid utilise les signaux actuellement produits pendant l'apprentissage : priorité issue de l'erreur, nouveauté des entrées, récence et échantillons historiques. Les mesures sont brutes et ne démontrent pas une supériorité générale. Dans la campagne du 20 novembre 2026, après le retour à A, Hybrid obtient les MAE moyennes A/B/C `0,2734 / 1,1429 / 0,3418`; sans replay, elles sont `0,4460 / 0,9406 / 0,4532`. Le résultat montre un compromis propre à ce workload synthétique.

Pour ce protocole, `forgetting = MAE_A_après_A - MAE_A_après_B` : une valeur négative indique un oubli. `recovery = MAE_A_après_B - MAE_A_après_retour_A` : une valeur positive indique une récupération. Les lignes `mean` contiennent les moyennes, écarts-types et IC 95 % du forgetting et de la récupération sur les seeds. Ces résultats restent indicatifs : le scénario est synthétique et la dérive est une bascule nette entre deux fonctions.

Cette campagne reste synthétique et sert à détecter des tendances, pas à établir un classement universel. Les poids du réseau restent en float64; les scénarios int16/int8 quantifient uniquement les vecteurs de la mémoire FIFO. Les temps et débits dépendent de la machine et ne doivent être comparés qu'à environnement constant. La campagne ne mesure ni énergie, ni coût CPU théorique ou performances sur matériel embarqué.

## Baseline obligatoire

Chaque campagne devrait inclure une référence `full_dataset`, puis comparer les budgets bornés :

Le ratio MAE limité/full est calculé directement pour chaque stratégie, seed, optimiseur et perte.

Les résultats bruts restent la référence pour éviter de masquer les compromis. Les prochains benchmarks utiles pour la priorité latence/CPU devront ajouter des workloads représentatifs et des mesures sur cible, en conservant séparément les métriques de qualité, mémoire, temps et débit. Les précisions `float32` pour les poids et `int8` pour les poids/calcul ne sont pas encore disponibles.
