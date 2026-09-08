# Qu'est-ce que je peux lui apprendre ?

Cette page répond concrètement à une question simple : **avec le Gloomy actuel, qu'est-ce qu'on peut réellement lui faire apprendre ?**

## Ce que Gloomy sait faire aujourd'hui

Gloomy n'est pas un système à qui l'on « enseigne » des connaissances générales (pas de texte, pas d'images, pas de conversation). C'est un petit moteur de réseau dense en C++ qui sait aujourd'hui faire deux types d'apprentissage réel sur des séries scalaires : `ONLINE_LEARNING_RUNTIME` et `TRAINING_RUNTIME` (voir [Configurations et limites](configurations.md)).

- `ONLINE_LEARNING_RUNTIME` apprend, en continu et sans réentraînement complet, à **prédire la valeur suivante d'une série scalaire** : chaque valeur consécutive de la séquence d'entrée devient une observation (`x[i]`) et sa cible (`x[i+1]`).
- `TRAINING_RUNTIME` entraîne le réseau sur l'ensemble complet de la séquence avec `LearningEngine::train()` puis peut sauvegarder l'état complet si `model_path` est renseigné.

Concrètement, on peut lui apprendre à anticiper la suite d'un flux de mesures : un compteur, une température, une charge, un cours simplifié, un capteur — tant que c'est une seule valeur numérique par instant. Les exemples ci-dessous sont réels : chaque commande a été exécutée telle quelle avec le CLI actuel, les sorties sont copiées telles quelles.

Pour tester ces exemples vous-même :

```bash
make
./bin/gloomy "<sequence>" -f mon_exemple.config
```

Chaque ligne de sortie est `index observation cible prédiction perte` (avant la mise à jour des poids de cette étape) ; la dernière ligne résume `average_loss` et `memory_size`.

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
0	10	12	0	0
1	12	14	-0.130407	4.89972
2	14	16	0.447675	3.56908
3	16	18	0.770445	2.6024
4	18	20	0.861226	2.39249
5	20	22	0.901147	2.30411
6	22	24	0.923542	2.25507
7	24	26	0.937797	2.22378
8	26	28	0.947621	2.202
average_loss=2.4943 memory_size=9
```

À regarder : la colonne perte descend de `4.9` à `2.2` au fil des observations — le réseau apprend réellement la tendance, en continu, sans qu'on lui repasse jamais les mêmes données depuis le début.

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
0	20	20.5	0	0
1	20.5	21	-0.130407	4.89972
2	21	45	0.447675	1184.89
3	45	21.5	0.999996	872.418
4	21.5	22	0.982747	698.301
5	22	22.5	0.992079	582.187
6	22.5	23	0.99593	499.22
average_loss=548.845 memory_size=7
```

```ini
# ex2b.config — identique, sauf loss=huber, huber_delta=1.0
```

```text
0	20	20.5	0	0
1	20.5	21	-0.130407	1.3152
2	21	45	-0.0517869	20.7006
3	45	21.5	0.0807379	15.4986
4	21.5	22	0.148519	12.4097
5	22	22.5	0.154687	10.3496
6	22.5	23	0.154403	8.87508
average_loss=9.8784 memory_size=7
```

À regarder : le pic (`45`) fait exploser la perte MSE (`1184.89`) et **le réseau met plusieurs étapes à s'en remettre** (perte encore `~500` à la fin). Avec Huber, le même pic ne dépasse jamais `~21` de perte et le réseau retrouve une perte à un chiffre dès l'étape suivante. C'est exactement le compromis documenté dans [Fonctions de perte](losses.md) : Huber est quadratique près de la cible et linéaire pour les grandes erreurs, donc moins dominée par un aberrant.

## Exemple 3 — Ne pas tout oublier quand les données changent de régime

C'est le scénario le plus caractéristique de ce que la mémoire d'apprentissage apporte : un changement de régime (la relation entre l'entrée et la cible change en cours de route), suivi d'un retour à l'ancien régime. Sans rejouer d'anciens exemples, un réseau entraîné en continu **oublie catastrophiquement** ce qu'il savait avant le changement.

Ce scénario est déjà construit et mesuré par `make benchmark` (voir `src/BenchmarkRunner.cpp`) : un réseau apprend d'abord une relation linéaire (`régime A`), puis une relation inverse (`régime B`), et on mesure combien sa performance sur `régime A` s'est dégradée — avec et sans mémoire de replay FIFO pendant l'apprentissage du régime B.

```bash
make benchmark
grep forgetting benchmark_results.csv
```

```text
forgetting_no_replay,float64,sgd,...,-19.086601585036295
forgetting_fifo_replay,float64,sgd,...,-4.9040043026402236
```

La dernière colonne est `Metrics::forgetting` (perte sur `régime A` avant moins après ; plus proche de `0` est mieux). **Sans mémoire, l'oubli est environ 4 fois plus important** (`-19.09` contre `-4.90`) : rejouer même un sous-ensemble d'anciennes observations pendant l'apprentissage du nouveau régime préserve nettement mieux ce qui avait été appris. C'est le problème central que les stratégies de mémoire (FIFO, Reservoir, Prioritized, Novelty, Hybrid — voir [Mémoire d'apprentissage](memory.md)) essaient chacune d'atténuer différemment.

## Ce que Gloomy ne sait pas (encore) apprendre

Pour ne pas se tromper d'attentes :

- **Une seule valeur à la fois** : le runtime online est scalaire (une entrée, une sortie). Pas de séries multivariées, pas d'image, pas de texte.
- **Pas de vraie mémoire de contexte** : chaque prédiction ne voit que l'observation courante, pas une fenêtre des valeurs précédentes (pas de couche récurrente).
- **Pas de classification** : `softmax` existe et son gradient est vérifié (voir [Fonctions d'activation](activations.md)), mais aucun runtime CLI ne l'exploite avec une sortie à plusieurs neurones et des cibles de classe.
- **Le CLI ne charge pas encore** un modèle sauvegardé au démarrage ; le runtime `online_learning` sauvegarde désormais le réseau/l'optimiseur/la mémoire entraînés à la fin d'un run si `model_path` est renseigné, et le runtime `training` fait de même après un entraînement complet.

## Pour aller plus loin

Les autres clés de configuration (`optimizer`, `memory_strategy` et leurs paramètres, `seed`, `batch_size`, `train_every`...) sont documentées avec leur effet dans [Configurations et limites](configurations.md), et le détail de chaque composant (pertes, mémoires, optimiseurs) dans les pages correspondantes du [sommaire](README.md).
