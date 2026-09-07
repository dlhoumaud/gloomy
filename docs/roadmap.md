# Gloomy : état du projet et feuille de route

Ce document sert de fil conducteur du projet. Il décrit ce qui est réellement présent dans le code, ce qui a été validé et les étapes restantes pour atteindre un runtime de continual learning embarqué.

## 1. État actuel

Gloomy possède maintenant un moteur Dense C++ avec propagation avant, rétropropagation, entraînement par mini-batches, mémoires d'apprentissage interchangeables, quantification expérimentale, persistance partielle, métriques et benchmark.

Le projet compile sans dépendance externe avec C++17 et Make.

Commandes de validation actuelles :

```bash
make clean
make test
make
make benchmark
```

Le benchmark produit `benchmark_results.csv`, ignoré par Git.

## 2. Fonctionnalités implémentées

### Réseau neuronal

- couches entièrement connectées `DenseLayer` ;
- poids et biais accessibles pour l'entraînement ;
- propagation avant vectorielle ;
- rétropropagation couche par couche ;
- gradients des poids et biais ;
- propagation des gradients vers les couches précédentes ;
- accumulation et remise à zéro des gradients ;
- activations `none`, `sigmoid`, `relu`, `leaky_relu`, `tanh` ;
- softmax appliqué à la dernière couche ;
- validation des dimensions et des entrées.

### Fonctions de perte

- `MSELoss` ;
- `MAELoss` ;
- `HuberLoss` avec `delta` configurable ;
- gradients analytiques ;
- tests de valeurs et de gradients.

### Optimiseurs

- SGD ;
- SGD avec momentum ;
- Adam ;
- learning rate configurable ;
- validation des hyperparamètres ;
- mesure de la mémoire d'état avec `stateBytes()` pour Momentum et Adam.

### Moteur d'apprentissage

`LearningEngine` sépare le réseau, la perte et l'optimiseur.

Il supporte :

- entraînement sur un dataset de `TrainingSample` ;
- epochs ;
- mini-batches ;
- online learning avec batch de taille `1` ;
- entraînement depuis une mémoire bornée ;
- ajout d'une observation puis replay ;
- replay indexé ;
- recalcul de l'erreur et de la priorité après mise à jour des poids.

### Mémoires d'apprentissage

Toutes ces stratégies implémentent `LearningMemory` et sont remplaçables sans modifier le réseau :

- FIFO / sliding window ;
- Reservoir Sampling ;
- Prioritized Replay ;
- Novelty Memory ;
- Hybrid Memory avec partitions `recent`, `error`, `novelty`, `historical` ;
- métadonnées `priority`, `error`, `novelty`, `rarity`, `recency`, `diversity`, `age`, `usage_count` ;
- replay indexé par position ;
- âge et nombre d'utilisations mis à jour.

### Importance et scheduling

- `ImportanceScorer` avec composantes pondérées et normalisées ;
- `EverySampleScheduler` ;
- `EveryNScheduler` ;
- `OnHighErrorScheduler` ;
- entraînement différable sans perdre l'observation dans la mémoire.

### Normalisation

- statistiques streaming avec Welford ;
- moyenne, variance, minimum, maximum ;
- normalisation centrée-réduite ;
- restauration des statistiques ;
- règle documentée : ne pas calculer les statistiques avec validation/test.

### Quantification et mémoire compacte

- quantification affine int16 ;
- quantification affine int8 ;
- calibration `scale` / `zero_point` ;
- saturation et déquantification ;
- codec de `TrainingSample` int16 et int8 ;
- `QuantizedFIFOMemory` ;
- `QuantizedInt8FIFOMemory` ;
- `bytesPerSample()` et `memoryUsedBytes()`.

Les métadonnées sont conservées en précision native. Les poids du réseau restent actuellement en float64.

### Persistance

Persistance séparée déjà disponible pour :

- `TrainingSample` ;
- architecture et paramètres du réseau ;
- statistiques de normalisation.

Le fichier réseau est versionné et protégé par un checksum FNV-1a. Les tests vérifient le round-trip et le rejet d'une corruption.

### Métriques et benchmark

Métriques disponibles :

- MAE ;
- RMSE ;
- forgetting.

Le runner benchmark compare actuellement :

- dataset complet ;
- FIFO ;
- Reservoir ;
- Prioritized ;
- FIFO int16 ;
- FIFO int8 ;
- SGD, Momentum et Adam ;
- temps d'entraînement ;
- latence d'inférence ;
- mémoire des paramètres et de l'état optimiseur ;
- mémoire d'apprentissage ;
- MAE, RMSE et pertes.

Il contient aussi une expérience synthétique de catastrophic forgetting avec et sans replay FIFO.

## 3. Ce qui n'est pas encore fait

### Priorité haute : runtime d'apprentissage complet

1. **Format `GLOOMY_MODEL` unifié**

   Regrouper dans un seul fichier versionné :

   - header global ;
   - architecture ;
   - poids et biais ;
   - activation et post-activation ;
   - normalisation ;
   - paramètres de quantification ;
   - mémoire d'apprentissage ;
   - configuration de l'optimiseur ;
   - état de l'optimiseur ;
   - métadonnées ;
   - checksum global.

   Le format doit prévoir des sections, des tailles, une version et une compatibilité future. Le chargement doit être atomique : un fichier invalide ne doit pas laisser un modèle partiellement modifié.

2. **Persistance de l'état des optimiseurs**

   Ajouter à l'abstraction `Optimizer` un contrat de sauvegarde/restauration. Il faudra persister :

   - type d'optimiseur ;
   - learning rate ;
   - hyperparamètres ;
   - compte d'updates Adam ;
   - vitesses Momentum ;
   - premiers et seconds moments Adam.

   L'état doit être vérifié contre la forme des couches pour éviter de restaurer un buffer incompatible.

3. **Persistance des mémoires**

   Sauvegarder et restaurer :

   - stratégie utilisée ;
   - capacité ;
   - seed et état RNG si la reproductibilité est requise ;
   - échantillons ;
   - priorités ;
   - âge et usages ;
   - partitions Hybrid ;
   - paramètres de Novelty et Prioritized ;
   - représentation float64/int16/int8 ;
   - paramètres `scale` / `zero_point`.

4. **CLI d'entraînement et runtime online**

   Le CLI actuel sait uniquement construire un réseau d'inférence et générer des prédictions. Il faut ajouter des modes explicites :

   - `TRAINING_RUNTIME` ;
   - `INFERENCE_RUNTIME` ;
   - `ONLINE_LEARNING_RUNTIME`.

   Le runtime online devra suivre :

   ```text
   observation -> normalisation -> prédiction -> cible
   -> erreur -> mémoire -> scheduler -> replay -> mise à jour
   ```

### Configuration fichier

Aucune nouvelle option CLI n'a encore été ajoutée, et c'est cohérent pour l'instant : les nouvelles fonctionnalités sont des composants C++ testés séparément, tandis que le CLI historique reste focalisé sur l'inférence.

À terme, un fichier de configuration est préférable à une commande contenant des dizaines d'options. L'idée proposée est :

```bash
./bin/gloomy -f gloomy.config
```

Le fichier pourrait contenir :

```ini
runtime=online_learning
activation=tanh
post_activation=none
hidden_layers=2
neurons=16
loss=huber
huber_delta=1.0
optimizer=momentum
learning_rate=0.001
momentum=0.9
memory_strategy=hybrid
memory_capacity=256
recent_ratio=0.25
error_ratio=0.25
novelty_ratio=0.25
historical_ratio=0.25
novelty_threshold=1.0
precision=int8
train_every=10
batch_size=8
seed=1234
model_path=model.gloomy
metrics_path=benchmark.csv
```

Avant de l'implémenter, il faudra décider :

- format INI simple, JSON sans dépendance externe, ou format clé-valeur propriétaire ;
- priorité entre valeurs du fichier et options CLI ;
- validation et messages d'erreur ;
- valeurs par défaut centralisées ;
- support des secrets ou chemins externes ;
- compatibilité de version du fichier ;
- distinction entre configuration d'entraînement et configuration d'inférence.

Recommandation : commencer par un parseur clé-valeur INI minimal sans dépendance externe, avec priorité `CLI > fichier > défauts`. Ne pas appeler ce fichier `.env` au sens strict si ses valeurs ne sont pas destinées à être des variables d'environnement ; `gloomy.config` ou `gloomy.ini` serait plus explicite. Un alias `-f` peut néanmoins accepter n'importe quel chemin.

### Priorité moyenne : robustesse mathématique

- gradient checking généralisé à toutes les activations et plusieurs couches ;
- tests de gradient softmax multi-sortie ;
- tests de stabilité avec très grandes valeurs ;
- test de reproductibilité avec seed injectable ;
- remplacement de `rand()` par un générateur contrôlable ;
- validation systématique des valeurs non finies dans tous les composants.

### Priorité moyenne : mémoire et continual learning

- mise à jour de toutes les composantes du score d'importance ;
- vraie récence fondée sur l'âge plutôt que l'alternance actuelle de Hybrid ;
- mise à jour des priorités avec correction de biais d'échantillonnage ;
- exploration contrôlée des échantillons de faible priorité ;
- prototypes / coreset ;
- mémoire par régimes ;
- détection légère de concept drift ;
- règles adaptatives explicites avant tout mécanisme appris ;
- expériences A -> B -> A plus nombreuses et reproductibles.

### Priorité moyenne : benchmark scientifique

Étendre le runner à :

- capacités `32`, `64`, `128`, `256` ;
- Novelty et Hybrid ;
- float64, int16 et int8 sur les mêmes données ;
- pertes MSE, MAE et Huber ;
- plusieurs seeds ;
- moyenne, écart-type et intervalles de confiance ;
- baseline dernière valeur connue ;
- ratio `performance_memory_limited / performance_full_dataset` ;
- coût CPU et nombre d'opérations approximatif ;
- samples/sec et updates/sec ;
- fichiers CSV séparés par expérience.

### Priorité moyenne : compression et embarqué

- compression différentielle des séries temporelles ;
- stockage sans allocations pendant la boucle online ;
- arena allocator ou capacité statique ;
- buffers contigus ;
- quantification des poids ;
- runtime inference-only minimal ;
- génération d'un artefact modèle sans métadonnées inutiles ;
- compilation et tests sur une cible embarquée réelle ;
- mesure RAM, Flash, CPU et énergie.

### Priorité basse : extensions

- optimisations SIMD ;
- multithreading PC ;
- Adam quantifié ou optimiseur à état compressé ;
- classification multi-classe avec sortie `K` neurones ;
- pertes classification et cross-entropy ;
- détection de dérive plus avancée ;
- adaptation dynamique apprise.

## 4. Ordre recommandé pour la suite

1. Stabiliser la sérialisation unifiée du modèle.
2. Ajouter la persistance de l'état Optimizer.
3. Ajouter la persistance des mémoires et des paramètres de quantification.
4. Centraliser les défauts dans une configuration C++.
5. Ajouter `-f/--config` au CLI avec priorité CLI > fichier > défauts.
6. Exposer un mode online fonctionnel dans le CLI.
7. Étendre les benchmarks aux capacités et stratégies restantes.
8. Ajouter les tests de concept drift et catastrophic forgetting.
9. Optimiser les allocations et la représentation mémoire.
10. Préparer le runtime embarqué et la quantification des poids.

## 5. Limites connues à ne pas oublier

- le CLI recrée actuellement le réseau entre prédictions autorégressives, avec de nouveaux poids aléatoires ;
- le CLI ne charge pas encore de modèle sauvegardé ;
- le CLI ne lance pas encore l'entraînement ;
- `softmax` sur la sortie actuelle à un neurone vaut toujours `1` ;
- les couches utilisent encore des vecteurs imbriqués et des allocations dynamiques ;
- les mémoires natives stockent encore des `double` ;
- les poids restent en float64 ;
- les optimiseurs ne sont pas encore persistés ;
- les serializers séparés ne constituent pas encore un fichier modèle complet ;
- les statistiques de benchmark dépendent de la machine ;
- les données synthétiques ne remplacent pas une évaluation sur données réelles.

## 6. Règle de travail

Chaque prochaine étape doit rester petite et vérifiable :

1. annoncer le périmètre et les fichiers touchés ;
2. modifier une seule brique cohérente ;
3. ajouter ou adapter les tests ;
4. mettre à jour la documentation concernée ;
5. exécuter `make test` ;
6. exécuter `make` ;
7. exécuter `make benchmark` si le benchmark est concerné ;
8. mettre à jour ce document avec l'état réel.
