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

La baseline compare SGD, Momentum et Adam, les pertes MSE, MAE et Huber, et les capacités `32`, `64`, `128` et `256`. Elle couvre FIFO, Reservoir, Prioritized, Novelty, Hybrid et FIFO quantifié int16/int8. Le dataset contient `400` exemples d'entraînement et `80` de validation ; les scénarios utilisent des updates par batch/replay de `8` exemples. Les lignes brutes sont suivies de lignes `mean` avec moyenne, écart-type et intervalle de confiance à 95 % sur les trois seeds. Chaque seed initialise le réseau et les générateurs des stratégies Reservoir, Prioritized et Hybrid; un même seed reproduit donc l'initialisation et les tirages de mémoire.

Une baseline `last_value_baseline` prédit la dernière valeur d'entrée. Pour chaque scénario à mémoire bornée, `mae_ratio_vs_full` donne `MAE du scénario / MAE full_dataset` sur le même seed, optimiseur et perte ; une valeur inférieure à `1` indique une MAE meilleure que la référence full dataset. `inference_time_us` est le temps moyen par exemple de la boucle d'évaluation, qui comprend prédiction, perte et métriques; ce n'est pas une mesure isolée du seul appel `forward()`. Les colonnes de débit rapportent les exemples et updates par seconde dans la boucle chronométrée. `memory_used_bytes` estime les allocations conservées par la stratégie mémoire, le réseau, l'optimiseur et les buffers du `LearningEngine`. Cette estimation exclut le dataset complet, les métadonnées de l'allocateur et les autres allocations du processus. `memoryUsedBytes()` des mémoires quantifiées conserve son sens historique de taille logique des échantillons encodés.

Les CSV de la campagne sont produits à la racine : `benchmark_results.csv` rassemble les résultats, `benchmark_memory.csv` contient baseline et comparaisons mémoire, `benchmark_forgetting.csv` contient les expériences d'oubli, et `benchmark_drift.csv` contient le protocole A → B → A. `benchmark_profile.csv` rapporte la latence moyenne, p50, p95, les allocations C++ par opération et `process_peak_rss_bytes`. Il profile le réseau Dense `1 → 8 → 1` (régression scalaire) et un proxy multi-sortie `3 → 16 → 16 → 2` inspiré de l'exemple avancé; les opérations incluent `forward()`, l'inférence sans cache, le gradient de perte, un update batch, un replay mémoire et une copie de mini-batch. Le RSS est le maximum observé par le système d'exploitation pour le processus jusqu'à la fin du profil; il vaut zéro si la plateforme ne fournit pas cette mesure. Le décompte d'allocations mesure les appels ordinaires aux opérateurs globaux `new`/`new[]`; il ne couvre pas les allocations alignées ou internes à des bibliothèques C. Les percentiles décrivent ces workloads synthétiques et cette machine.

Les workloads de mesure actuellement définis sont des proxies synthétiques pour la régression scalaire (1 entrée → 1 sortie, réseau `1 → 8 → 1`), la régression multi-sortie (3 entrées → 2 sorties, réseau `3 → 16 → 16 → 2`) et le replay de rétention A → B → A. Ils ne remplacent pas un dataset applicatif réel. La cible de mesure des profils locaux est le poste Linux x86_64 équipé d'un AMD Ryzen 3 5300U (4 cœurs / 8 threads), compilé avec GCC 15.2.0 et les options Makefile (`-Wall -O2 -std=c++17`). Ce poste sert de baseline de développement, pas de cible embarquée. Aucun seuil de déploiement n'est fixé avant de connaître la cible et les données applicatives réelles.

`benchmark_artifacts.csv` compare la taille fichier et le temps moyen de chargement sur 100 itérations pour un checkpoint `GLOOMY_MODEL` (Adam, sans mémoire) et un artefact `GLOOMY_INFER`, sur le réseau Dense synthétique du benchmark. La mesure sert à suivre le coût du format, pas à prédire les performances d'un modèle applicatif.

La cible `make benchmark-comparison` compare le code antérieur aux optimisations (`34a3ef6`) au working tree courant avec un runner identique compilé contre chacun des deux arbres. Le protocole fixe la régression `y = 2x + 1`, le réseau `1 → 8 → 1`, SGD `0.01`, MSE, 400 exemples d'entraînement, 80 de validation, une epoch (50 updates) et trois seeds. `benchmark_optimization_comparison.csv` conserve les résultats par seed : durée d'entraînement, loss/MAE/RMSE de validation, allocations et octets demandés pendant l'entraînement, octets temporairement vivants au pic, octets créés pendant l'entraînement encore retenus à sa fin, RSS de processus et nombre d'updates. L'estimation d'allocations instrumente les appels C++ ordinaires `new`/`new[]`; les octets demandés n'incluent pas les frais de l'allocateur. La RSS comprend le processus complet et sert de contrôle grossier pour ce petit workload.

Snapshot du profil sur la machine de référence (20 novembre 2026) : le pic RSS du processus était de `5 144 576` octets. Sur le réseau scalaire, les temps moyens étaient `0,118 µs` pour l'inférence sans cache, `1,602 µs` pour un update batch et `2,875 µs` pour un replay; sur le réseau `3 → 16 → 16 → 2`, ils étaient `0,976 µs`, `21,316 µs` et `29,982 µs`. Ces trois opérations ont compté zéro allocation ordinaire `new`/`new[]` après échauffement; `forward()` par valeur en compte une et la copie de mini-batch 17. Les valeurs brutes, percentiles et RSS sont dans `benchmark_profile.csv`. Elles sont indicatives, dépendent de la fréquence CPU et ne constituent pas des seuils.

Sur la comparaison avant/après, le temps d'entraînement moyen sur les trois seeds passe de `0,1407 ms` à `0,0778 ms` (environ `44,7 %` plus court sur cette exécution). La loss/MAE/RMSE de validation restent exactement identiques pour chaque seed (`0,277953 / 0,514097 / 0,516006` en moyenne). Les allocations pendant l'entraînement passent de `4 456` à `11`, et les octets demandés de `147 416` à `368`; le pic des octets mesurés vivants passe de `1 392` à `368`. Les allocations retenues créées pendant l'entraînement passent toutefois de `216` à `368` octets, soit `152` octets de buffers réutilisables supplémentaires. La RSS moyenne observée reste autour de `4,25 MB` dans les deux builds, trop proche du coût de démarrage du processus pour conclure à un gain RSS sur ce petit modèle. Le gain mesuré conserve donc la qualité, réduit fortement les allocations temporaires et augmente légèrement la mémoire retenue. Il s'agit de l'effet groupé des optimisations depuis `34a3ef6`, pas d'une attribution individuelle à chaque changement; les résultats ne remplacent pas une comparaison sur données applicatives.

Le runner conserve deux lignes de comparaison A → B : `forgetting_no_replay` entraîne sur A puis B sans replay, tandis que `forgetting_fifo_replay` réentraîne périodiquement depuis la mémoire FIFO de A. Leur colonne `forgetting` suit la convention `perte_A_avant - perte_A_après`; une valeur négative signifie que la perte A a augmenté après l'apprentissage de B.

Le protocole `concept_drift_a_b_a` entraîne sur A, bascule sur B, puis revient sur A. Il compare `no_replay`, FIFO, Reservoir, Prioritized, Novelty et Hybrid sur trois seeds. Chaque phase reçoit dix passages et le même nombre d'updates (50) pour chaque stratégie. La mémoire a une capacité de 64. Le CSV donne la MAE sur A après la première phase A, après B et après le retour à A, ainsi que la MAE sur B après B et après le retour à A.

Le protocole `benchmark_progressive_drift.csv` ajoute trois régimes linéaires distincts A, B et C, séparés par des transitions dont les cibles interpolent progressivement entre les deux fonctions. L'ordre est A → transition A/B → B → transition B/C → C → retour à A. Il compare `no_replay`, FIFO, Reservoir, Prioritized, Novelty et Hybrid avec trois seeds, 40 observations par régime, dix passages et 50 updates par étape; chaque mémoire a une capacité de 64. Chaque ligne contient la MAE mesurée sur les trois régimes après l'étape, son temps d'entraînement et l'estimation mémoire. Hybrid utilise les signaux actuellement produits pendant l'apprentissage : priorité issue de l'erreur, nouveauté des entrées, récence et échantillons historiques. Les mesures sont brutes et ne démontrent pas une supériorité générale. Dans la campagne du 20 novembre 2026, après le retour à A, Hybrid obtient les MAE moyennes A/B/C `0,2734 / 1,1429 / 0,3418`; sans replay, elles sont `0,4460 / 0,9406 / 0,4532`. Le résultat montre un compromis propre à ce workload synthétique.

Pour ce protocole, `forgetting = MAE_A_après_A - MAE_A_après_B` : une valeur négative indique un oubli. `recovery = MAE_A_après_B - MAE_A_après_retour_A` : une valeur positive indique une récupération. Les lignes `mean` contiennent les moyennes, écarts-types et IC 95 % du forgetting et de la récupération sur les seeds. Ces résultats restent indicatifs : le scénario est synthétique et la dérive est une bascule nette entre deux fonctions.

Cette campagne reste synthétique et sert à détecter des tendances, pas à établir un classement universel. Les poids du réseau restent en float64; les scénarios int16/int8 quantifient uniquement les vecteurs de la mémoire FIFO. Les temps et débits dépendent de la machine et ne doivent être comparés qu'à environnement constant. La campagne ne mesure ni énergie, ni coût CPU théorique ou performances sur matériel embarqué.

## Baseline obligatoire

Chaque campagne devrait inclure une référence `full_dataset`, puis comparer les budgets bornés :

Le ratio MAE limité/full est calculé directement pour chaque stratégie, seed, optimiseur et perte.

Les résultats bruts restent la référence pour éviter de masquer les compromis. Les prochains benchmarks utiles pour la priorité latence/CPU devront ajouter des workloads représentatifs et des mesures sur cible, en conservant séparément les métriques de qualité, mémoire, temps et débit. Les précisions `float32` pour les poids et `int8` pour les poids/calcul ne sont pas encore disponibles.
