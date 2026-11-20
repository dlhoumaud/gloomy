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

Il contient aussi une expérience A → B de catastrophic forgetting et un protocole A → B → A sur trois seeds, avec et sans replay et plusieurs stratégies de mémoire.

## 3. Ce qui n'est pas encore fait

### Priorité haute : runtime d'apprentissage complet

1. **Format `GLOOMY_MODEL` unifié** — version 1 implémentée pour réseau, normalisation, optimiseur et mémoire optionnelle.

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

2. **Persistance de l'état des optimiseurs** — première version implémentée dans `OptimizerSerialization`.

   Ajouter à l'abstraction `Optimizer` un contrat de sauvegarde/restauration. Il faudra persister :

   - type d'optimiseur ;
   - learning rate ;
   - hyperparamètres ;
   - compte d'updates Adam ;
   - vitesses Momentum ;
   - premiers et seconds moments Adam.

   L'état est vérifié contre le type et la forme des couches pour éviter de restaurer un buffer incompatible. Il est inclus dans la section `OPTIMIZER` de `GLOOMY_MODEL`.

3. **Persistance des mémoires** — implémentée dans la section optionnelle `LEARNING_MEMORY` de `GLOOMY_MODEL` pour les stratégies intégrées.

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

4. **CLI d'entraînement et runtime online** — première version raccordée via `-f/--config`.

   Le CLI configuré charge et sauvegarde le format complet et propose les modes explicites :

   - `TRAINING_RUNTIME` ;
   - `INFERENCE_RUNTIME` ;
   - `ONLINE_LEARNING_RUNTIME`.

   Le runtime online devra suivre :

   ```text
   observation -> normalisation -> prédiction -> cible
   -> erreur -> mémoire -> scheduler -> replay -> mise à jour
   ```

### Configuration fichier

Le chemin CLI historique reste disponible pour les démonstrations d'inférence. Le chemin configuré utilise un fichier clé-valeur sans dépendance et propose déjà l'entraînement batch, le replay online et l'inférence depuis un `GLOOMY_MODEL`.

Le runtime configuré utilise un fichier de configuration plutôt qu'une commande contenant de nombreuses options :

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
dataset_path=training.txt
memory_strategy=hybrid
memory_capacity=256
recent_ratio=0.25
error_ratio=0.25
novelty_ratio=0.25
historical_ratio=0.25
novelty_threshold=1.0
precision=float64
train_every=10
batch_size=8
seed=1234
model_path=model.gloomy
```

Les éléments suivants restent à compléter :

- validation plus exhaustive des combinaisons de paramètres ;
- options CLI de surcharge pour tous les paramètres ;
- versionnement éventuel du fichier de configuration ;
- distinction plus stricte entre les paramètres d'entraînement et d'inférence.

Le parseur actuel accepte des lignes `clé=valeur`, des lignes vides et des commentaires commençant par `#` ou `;`. Les options implémentées suivent la priorité `CLI > fichier > défauts`.

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
- dérives progressives ou alternances entre plus de deux régimes ; le protocole A -> B -> A initial est maintenant dans le benchmark.

### Priorité moyenne : benchmark scientifique

La première campagne est implémentée : capacités `32/64/128/256`, Novelty et Hybrid, précisions float64/int16/int8 pour les mémoires, pertes MSE/MAE/Huber, trois seeds, moyennes/écarts-types/IC 95 %, baseline dernière valeur, ratio MAE limité/full, débits et CSV séparés. Il reste à estimer le coût CPU par nombre d'opérations et à élargir les datasets au-delà du scénario synthétique.

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

1. Rendre le chargement des composants transactionnel — implémenté pour le réseau, le normaliseur, les échantillons et l'optimiseur.
2. Stabiliser la sérialisation unifiée du modèle — version 1 implémentée avec sections réseau, normalisation, optimiseur et mémoire optionnelle ; quantification des poids et métadonnées de runtime restent à intégrer.
3. Ajouter la persistance des mémoires et des paramètres de quantification — implémenté dans la section mémoire optionnelle de `GLOOMY_MODEL` pour les stratégies intégrées et FIFO int16/int8.
4. Raccorder le chargement, l'entraînement et l'inférence au CLI — version initiale disponible avec `-f/--config`.
5. Étendre les benchmarks aux capacités et stratégies restantes — matrice initiale ajoutée pour capacités 32/64/128/256, stratégies intégrées, trois pertes, trois optimiseurs, trois seeds et baseline dernière valeur.
6. Étendre les expériences de concept drift — protocole A -> B -> A à trois seeds et six stratégies ajouté ; dérives progressives et plus de deux régimes restent à étudier.
7. Optimiser les allocations et la représentation mémoire — première réduction des copies temporaires dans les mini-batches de `LearningEngine` ; buffers contigus et allocations des couches restent à étudier.
8. Préparer le runtime embarqué et la quantification des poids.

## 5. Limites connues à ne pas oublier

- le chemin CLI historique recrée le réseau entre prédictions autorégressives avec de nouveaux poids ;
- `softmax` sur la sortie actuelle à un neurone vaut toujours `1` ;
- les couches utilisent encore des vecteurs imbriqués et des allocations dynamiques ;
- les mémoires natives stockent encore des `double` ;
- les poids restent en float64 ;
- la quantification des poids et certaines métadonnées de runtime ne sont pas encore persistées dans `GLOOMY_MODEL` ;
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
