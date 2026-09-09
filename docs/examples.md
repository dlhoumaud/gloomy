# Qu'est-ce que je peux lui apprendre ?

Cette page répond concrètement à une question simple : **avec le Gloomy actuel, qu'est-ce qu'on peut réellement lui faire apprendre ?**

## Ce que Gloomy sait faire aujourd'hui

Gloomy n'est pas un système à qui l'on « enseigne » des connaissances générales (pas de texte, pas d'images, pas de conversation). C'est un petit moteur de réseau dense en C++ qui sait aujourd'hui faire deux types d'apprentissage réel sur des séries scalaires : `ONLINE_LEARNING_RUNTIME` et `TRAINING_RUNTIME` (voir [Configurations et limites](configurations.md)).

- `ONLINE_LEARNING_RUNTIME` apprend, en continu et sans réentraînement complet, à **prédire la valeur suivante d'une série scalaire** : par défaut, chaque valeur consécutive de la séquence d'entrée devient une observation (`x[i]`) et sa cible (`x[i+1]`) ; avec `window_size > 1`, l'observation devient une fenêtre des `window_size` dernières valeurs (`x[i..i+window_size-1]`) et la cible reste la valeur suivante (`x[i+window_size]`).
- `TRAINING_RUNTIME` entraîne le réseau sur l'ensemble complet de la séquence avec `LearningEngine::train()` puis peut sauvegarder l'état complet si `model_path` est renseigné.
- `concept_drift_detection=true` surveille en continu si l'erreur récente s'écarte significativement de son historique et, si c'est le cas, déclenche immédiatement un replay supplémentaire depuis la mémoire — voir [Mémoire d'apprentissage](memory.md), « Détection de concept drift ».

Concrètement, on peut lui apprendre à anticiper la suite d'un flux de mesures : un compteur, une température, une charge, un cours simplifié, un capteur — tant que c'est une seule valeur numérique par instant. Les exemples ci-dessous sont réels : chaque commande a été exécutée telle quelle avec le CLI actuel, les sorties sont copiées telles quelles.

Pour tester ces exemples vous-même :

```bash
make
./bin/gloomy "<sequence>" -f mon_exemple.config
```

Chaque ligne de sortie est `index observation cible prédiction perte derive` (avant la mise à jour des poids de cette étape ; `derive` vaut `1` si `concept_drift_detection` est actif et qu'une dérive est détectée à ce pas, `0` sinon — voir [Mémoire d'apprentissage](memory.md), « Détection de concept drift ») ; la dernière ligne résume `average_loss` et `memory_size`.

## Exemple 1 — Apprendre une tendance (compteur, mesure qui progresse)

Un relevé qui augmente régulièrement (ex. un compteur électrique heure par heure) :

```ini
# ex1.config
runtime=online_learning
activation=tanh
hidden_layers=1
neurons=8
loss=mse
optimizer=sgd
learning_rate=0.05
memory_strategy=fifo
memory_capacity=16
```

```bash
./bin/gloomy "10 12 14 16 18 20 22 24 26 28" -f ex1.config
```

```text
0	10	12	0	0	0
1	12	14	-0.130407	4.89972	0
2	14	16	0.447675	3.56908	0
3	16	18	0.770445	2.6024	0
4	18	20	0.861226	2.23156	0
5	20	22	0.901415	2.00461	0
6	22	24	0.923736	1.84206	0
7	24	26	0.937822	1.71707	0
8	26	28	0.947476	1.69676	0
average_loss=2.28481 memory_size=9
```

À regarder : la colonne perte descend de `4.9` à `1.7` au fil des observations — le réseau apprend réellement la tendance, en continu, sans qu'on lui repasse jamais les mêmes données depuis le début.

## Exemple 2 — Apprendre en présence d'une valeur aberrante (choix de la perte)

Un capteur globalement stable avec un pic ponctuel (défaut de mesure) : `20 20.5 21 45 21.5 22 22.5 23`. Deux exécutions, seule la clé `loss` change.

```ini
# ex2a.config — loss=mse (sensible aux grandes erreurs)
runtime=online_learning
activation=tanh
hidden_layers=1
neurons=8
loss=mse
optimizer=sgd
learning_rate=0.05
memory_strategy=fifo
memory_capacity=16
```

```text
0	20	20.5	0	0	0
1	20.5	21	-0.130407	4.89972	0
2	21	45	0.447675	1184.89	0
3	45	21.5	0.999996	872.418	0
4	21.5	22	0.982747	698.301	0
5	22	22.5	0.992079	582.187	0
6	22.5	23	0.99593	499.22	0
average_loss=548.845 memory_size=7
```

```ini
# ex2b.config — identique, sauf loss=huber, huber_delta=1.0
```

```text
0	20	20.5	0	0	0
1	20.5	21	-0.130407	1.3152	0
2	21	45	-0.0517869	20.7006	0
3	45	21.5	0.0807379	15.4986	0
4	21.5	22	0.148519	12.4097	0
5	22	22.5	0.154687	10.3496	0
6	22.5	23	0.154403	8.87508	0
average_loss=9.8784 memory_size=7
```

À regarder : le pic (`45`) fait exploser la perte MSE (`1184.89`) et **le réseau met plusieurs étapes à s'en remettre** (perte encore `~500` à la fin). Avec Huber, le même pic ne dépasse jamais `~21` de perte et le réseau retrouve une perte à un chiffre dès l'étape suivante. C'est exactement le compromis documenté dans [Fonctions de perte](losses.md) : Huber est quadratique près de la cible et linéaire pour les grandes erreurs, donc moins dominée par un aberrant.

## Exemple 3 — Ne pas tout oublier quand les données changent de régime

C'est le scénario le plus caractéristique de ce que la mémoire d'apprentissage apporte : un changement de régime (la relation entre l'entrée et la cible change en cours de route), suivi d'un retour à l'ancien régime. Sans rejouer d'anciens exemples, un réseau entraîné en continu **oublie catastrophiquement** ce qu'il savait avant le changement.

Ce scénario est déjà construit et mesuré par `make benchmark` (voir `src/BenchmarkRunner.cpp`) : un réseau apprend d'abord une relation linéaire (`régime A`), puis une relation inverse (`régime B`), et on mesure combien sa performance sur `régime A` s'est dégradée — avec et sans mémoire de replay FIFO pendant l'apprentissage du régime B.

```bash
make benchmark
cat benchmark_forgetting.csv
```

```text
forgetting_no_replay,float64,sgd,mse,...,-19.26244500298063,...
forgetting_fifo_replay,float64,sgd,mse,...,-4.8868810963178513,...
```

La colonne `forgetting` est `Metrics::forgetting` (perte sur `régime A` avant moins après ; plus proche de `0` est mieux). **Sans mémoire, l'oubli est environ 4 fois plus important** (`-19.26` contre `-4.89`) : rejouer même un sous-ensemble d'anciennes observations pendant l'apprentissage du nouveau régime préserve nettement mieux ce qui avait été appris. C'est le problème central que les stratégies de mémoire (FIFO, Reservoir, Prioritized, Novelty, Hybrid — voir [Mémoire d'apprentissage](memory.md)) essaient chacune d'atténuer différemment.

Ces valeurs précises dépendent de l'état du générateur de poids partagé (`DenseLayer::seedWeightInitialization`, voir [Couches et neurones](architecture.md)) au moment où ce scénario s'exécute dans `main()`. `BenchmarkRunner.cpp` réinitialise explicitement ce générateur à la seed `1234` juste avant cette expérience, après le balayage capacités × seeds des scénarios quantifiés qui la précèdent, ce qui stabilise ces valeurs tant que le code exécuté entre ce point de réinitialisation et ce scénario ne change pas, sans remettre en cause le rapport d'environ `4x` entre les deux. `make benchmark` produit aussi désormais un fichier séparé par expérience (`benchmark_baseline.csv`, `benchmark_full_dataset.csv`, `benchmark_memory_capacity.csv`, `benchmark_quantization.csv`, `benchmark_forgetting.csv`) en plus du `benchmark_results.csv` combiné — voir [Benchmark](benchmark.md).

## Ce que Gloomy ne sait pas (encore) apprendre

Pour ne pas se tromper d'attentes :

- **Une seule série à la fois** : le runtime online prédit toujours une seule valeur suivante, à partir d'une seule série scalaire. `window_size` (défaut `1`, configurable) permet de lui faire voir les `window_size` dernières valeurs à chaque prédiction — pas une vraie couche récurrente, mais une fenêtre glissante explicite — sans jamais mélanger plusieurs séries indépendantes ni prédire plusieurs valeurs futures d'un coup. Pas d'image, pas de texte.
- **Pas de classification** : `softmax` existe et son gradient est vérifié (voir [Fonctions d'activation](activations.md)), mais aucun runtime CLI ne l'exploite avec une sortie à plusieurs neurones et des cibles de classe.
- **`bin/gloomy` charge désormais un modèle sauvegardé au démarrage**, pour `runtime=online_learning` **et** `runtime=training` : si `model_path` pointe vers un fichier `GLOOMY_MODEL` existant, l'exécution reprend le réseau, la normalisation, l'optimiseur et la mémoire sauvegardés au lieu de repartir de zéro (rejeté proprement si `window_size` ne correspond pas au modèle repris) ; les deux runtimes sauvegardent toujours l'état à la fin de l'exécution si `model_path` est renseigné. Un binaire séparé, `bin/gloomy_infer`, charge lui aussi un réseau déjà entraîné (au format `NetworkSerialization` ou, en int8, `QuantizedNetworkSerialization`) mais uniquement pour produire une prédiction, sans réentraîner — voir [Quantification](quantization.md), « Runtime d'inférence minimal ».

## Pour aller plus loin

Les autres clés de configuration (`optimizer`, `memory_strategy` et leurs paramètres, `seed`, `batch_size`, `train_every`...) sont documentées avec leur effet dans [Configurations et limites](configurations.md), et le détail de chaque composant (pertes, mémoires, optimiseurs) dans les pages correspondantes du [sommaire](README.md).
