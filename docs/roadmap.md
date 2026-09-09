# Gloomy : état du projet et feuille de route

Ce document sert de fil conducteur du projet. Il décrit ce qui est réellement présent dans le code, ce qui a été validé et les étapes restantes pour atteindre un runtime de continual learning embarqué.

## 1. État actuel

Gloomy possède maintenant un moteur Dense C++ avec propagation avant, rétropropagation, entraînement par mini-batches, mémoires d'apprentissage interchangeables, quantification expérimentale, persistance partielle, métriques et benchmark. Le CLI (`bin/gloomy`) sait générer des prédictions (`INFERENCE_RUNTIME`, comportement historique) et exécuter un flux d'apprentissage en continu configurable (`ONLINE_LEARNING_RUNTIME`, via un fichier `-f`/`--config`).

Le projet compile sans dépendance externe avec C++17 et Make.

Commandes de validation actuelles :

```bash
make clean
make test
make
make benchmark
```

Le benchmark produit `benchmark_results.csv`, ignoré par Git.

### Bug corrigé : dépendances de headers absentes du Makefile

Le `Makefile` compilait `bin/gloomy` de façon incrémentale (`$(OBJ_DIR)/%.o: src/%.cpp`), mais chaque `.o` ne dépendait que de son propre `.cpp`, pas des headers qu'il inclut. Après plusieurs changements de header sans `make clean` intermédiaire, `make` pouvait relier des `.o` compilés contre des versions différentes (incompatibles) d'un même header — une violation de l'ODR silencieuse, sans erreur de compilation. Symptôme observé : `bin/gloomy -f config.gloomy` avec `runtime=online_learning` corrompait le tas après quelques itérations d'apprentissage (`free(): invalid pointer`), de façon non déterministe selon l'historique de compilation, **uniquement pour la cible `bin/gloomy`** (`make test` et `make benchmark` compilent toujours toutes leurs sources en une seule invocation `g++`, donc n'étaient jamais affectés — ce qui explique pourquoi la suite de tests ne l'a pas détecté).

Corrigé en ajoutant `-MMD -MP` à `CXXFLAGS` et un `-include $(OBJ:.o=.d)` en bas du `Makefile` : chaque `.o` déclare désormais ses headers comme dépendances réelles, et `make` recompile correctement tout fichier affecté par un changement de header. Vérifié : modifier un header partagé par 10 fichiers déclenche maintenant la recompilation exacte de ces 10 fichiers (contre 0 avant le correctif) ; `bin/gloomy` a été testé sur des séquences de longueur 5 à 80 et les 5 stratégies de mémoire sans aucune récurrence après correction. Les fichiers `.d` générés sont ignorés par Git (`.gitignore`).

**Leçon retenue pour la suite** : après avoir modifié un header partagé, préférer `make clean && make && make test && make benchmark` à un `make` incrémental tant qu'un doute subsiste, même si le correctif ci-dessus élimine la cause en principe.

### Bug corrigé : persistance non sollicitée à des chemins par défaut non vides

`GloomyConfig::model_path`/`optimizer_path`/`memory_path`/`metrics_path` avaient des défauts non vides (`"model.gloomy"`, `"optimizer.gloomy"`, `"memory.gloomy"`, `"benchmark_results.csv"`), alors que `runOnline()`/`runTrainingRuntime()` (`src/main.cpp`) appellent inconditionnellement `saveOnlineArtifacts`/`saveTrainingArtifacts` en fin d'exécution, qui n'écrivent chaque artefact que si son chemin n'est pas vide. Résultat : **toute** exécution `runtime=online_learning` ou `runtime=training`, même avec une configuration minimale ne mentionnant aucun chemin, écrivait silencieusement 4 fichiers dans le répertoire courant — y compris en écrasant `benchmark_results.csv`, le fichier produit par `make benchmark` (un outil indépendant, qui n'utilise pas `GloomyConfig`). C'est exactement ce qui s'est produit en testant les exemples de [docs/examples.md](examples.md) : trois fichiers `model.gloomy`/`optimizer.gloomy`/`memory.gloomy` sont apparus à la racine du dépôt sans avoir été demandés.

Corrigé en vidant les quatre défauts : la persistance est maintenant réellement opt-in, comme la documentation ([Configurations et limites](configurations.md)) le décrivait déjà (« lorsqu'un `model_path` est renseigné ») sans que le code ne le respecte. `testGloomyConfigDefaults` caractérise désormais les quatre chemins comme vides par défaut. Vérifié manuellement : une exécution `online_learning` minimale n'écrit plus aucun fichier, et `model_path` explicite fonctionne toujours (le fichier attendu est créé). `*.gloomy` ajouté à `.gitignore`.

### Bug corrigé : `tmpnam()` dans `ModelSerialization`

`ModelSerialization::temporaryPath()` utilisait `std::tmpnam()` pour générer les chemins des fichiers temporaires servant à faire transiter chaque section (réseau, optimiseur, mémoire) par les sérialiseurs existants, basés sur des chemins de fichiers. `tmpnam()` ne fait que proposer un nom sans créer le fichier : une autre exécution peut s'y glisser entre la génération du nom et l'ouverture (TOCTOU) — glibc le signale explicitement comme dangereux à la liaison (`the use of 'tmpnam' is dangerous, better use 'mkstemp'`). Remplacé par `mkstemp()` (POSIX), qui crée et ouvre le fichier de façon atomique ; le chemin est construit sous `std::filesystem::temp_directory_path()`. Vérifié : plus aucun warning à la compilation ni à la liaison, et la sauvegarde/chargement du modèle unifié fonctionne toujours (`model_path` testé manuellement).

### Bug corrigé : calibration hors zéro dans `Int16Quantizer`/`Int8Quantizer`

`calibrate()` calculait `zero_point = qmin - min(values)/scale`, puis saturait silencieusement le résultat dans `[qmin, qmax]` s'il en sortait. Pour des valeurs qui ne contiennent pas `0` dans leur plage (un capteur toujours positif et loin de `0`, ex. `17.0 20.0 23.0 26.0`), ce calcul produisait systématiquement un `zero_point` hors plage, saturé — et la saturation qui suit dans `quantize()` écrasait alors **toutes** les valeurs vers le même code quantifié : perte totale d'information, sans qu'aucune exception ne soit levée. Mesuré avant correction : erreur de reconstruction ≈ `17.7` sur une plage de valeurs de seulement `9.0`. Ce bug touchait uniquement les données dont la plage ne contient pas `0` et n'est pas symétrique autour de `0` — c'est pourquoi il n'avait jamais été détecté : toutes les données de test existantes (poids de réseaux entraînés, à peu près centrés sur `0` ; échantillons de test qui contiennent déjà `0` ou sont symétriques) évitaient par coïncidence ce cas. Découvert en mesurant `DeltaQuantizer` (voir [Quantification](quantization.md), « Compression différentielle ») contre une quantification directe sur une série de type capteur avec un offset réaliste.

Corrigé en étendant systématiquement la plage de calibration pour qu'elle contienne toujours `0` avant de calculer `scale`/`zero_point`, ce qui garantit mathématiquement l'absence de saturation. Vérifié : tous les tests existants passent sans modification, et deux nouveaux cas de régression (int16 et int8, données loin de `0`) confirment que les valeurs restent distinguables après un aller-retour. Voir [Quantification](quantization.md), « Bug corrigé : calibration hors zéro », pour le détail mathématique.

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
- `CrossEntropyLoss` (classification, suppose une entrée post-softmax), vérifiée par différence finie sur un réseau complet — voir [Fonctions de perte](losses.md) ;
- gradients analytiques ;
- tests de valeurs et de gradients.

### Optimiseurs

- SGD ;
- SGD avec momentum ;
- Adam ;
- `CompressedAdamOptimizer` : mêmes mises à jour qu'Adam, moments conservés quantifiés en int16 entre deux pas (recalibrés à chaque update, ~1.75× moins d'octets d'état sur un petit réseau) — voir [Optimiseurs](optimizers.md), « Adam à état compressé » ;
- learning rate configurable ;
- validation des hyperparamètres ;
- mesure de la mémoire d'état avec `stateBytes()` pour Momentum, Adam et `CompressedAdamOptimizer`.

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
- les cinq composantes (`error`, `novelty`, `rarity`, `recency`, `diversity`) sont désormais réellement calculées par `LearningEngine` (auparavant, seule `error` l'était) — `recency`/`rarity` à l'ajout (`learn()`), les cinq au replay (`trainFromMemory()`, `novelty`/`diversity` par rapport aux autres échantillons du même batch) ; les poids par défaut (`error=1.0`, le reste à `0.0`) laissent la priorité inchangée, comportement historique préservé — voir [Mémoire d'apprentissage](memory.md), « Score d'importance » ;
- `EverySampleScheduler` ;
- `EveryNScheduler` ;
- `OnHighErrorScheduler` ;
- entraînement différable sans perdre l'observation dans la mémoire ;
- `ConceptDriftDetector` : détection active de dérive de concept (moyenne récente vs ligne de base historique, écart-type), branchée en option (`concept_drift_detection`) dans `runOnlineLearning()`, qui déclenche un replay supplémentaire dès qu'une dérive est signalée — voir [Mémoire d'apprentissage](memory.md), « Détection de concept drift » ;
- `PageHinkleyDetector` : seconde méthode de détection de dérive, un test séquentiel de détection de rupture classique de la littérature (Page-Hinkley), plus réactif sur le même changement de régime synthétique testé mais pas encore branché dans le runtime online — voir [Mémoire d'apprentissage](memory.md), « Détection de dérive plus avancée ».

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
- `bytesPerSample()` et `memoryUsedBytes()` ;
- `NetworkQuantization` : quantification int8 post-entraînement des poids/biais d'un `NeuralNetwork` (calibration séparée poids/biais par couche), avec `quantize()`/`dequantize()`/`quantizedBytes()` — voir [Quantification](quantization.md), « Quantification des poids d'un réseau entraîné » ;
- `DeltaQuantizer` : compression différentielle d'une série temporelle (quantifie les différences consécutives plutôt que les valeurs brutes), avec `encode()`/`decode()`/`quantizedBytes()` — un gain réel mais pas systématique, mesuré sur trois scénarios (voir [Quantification](quantization.md), « Compression différentielle »).

Les métadonnées sont conservées en précision native. Le réseau continue de s'entraîner en float64 ; seule une copie post-entraînement peut être quantifiée pour le déploiement.

### Persistance

Persistance séparée déjà disponible pour :

- `TrainingSample` ;
- architecture et paramètres du réseau ;
- statistiques de normalisation ;
- état de l'optimiseur (SGD, Momentum, Adam) via `OptimizerSerialization` ;
- mémoire d'apprentissage native (FIFO, Reservoir, Prioritized, Novelty, Hybrid) via `LearningMemorySerialization`.

Un format unifié `GLOOMY_MODEL` est désormais également disponible via `ModelSerialization` (`src/headers/ModelSerialization.h`, `src/ModelSerialization.cpp`) : il regroupe dans un fichier versionné et protégé par checksum le réseau, la normalisation, l'optimiseur et la mémoire d'apprentissage. Le fichier réseau, le fichier optimiseur, le fichier mémoire et le fichier unifié sont tous vérifiés par un checksum FNV-1a. Les tests vérifient le round-trip et le rejet d'une corruption. Pour l'optimiseur, `load()` reconstruit le type concret à partir du fichier et refuse de restaurer un état dont la forme (nombre de couches, dimensions par couche) ne correspond pas exactement au réseau fourni. Pour la mémoire, `load()` restaure aussi l'état complet du générateur `std::mt19937` (Reservoir, Prioritized, Hybrid) et les partitions (Hybrid), afin que le replay reste reproductible après un redémarrage. Les mémoires quantifiées (`QuantizedFIFOMemory`, `QuantizedInt8FIFOMemory`) sont désormais couvertes elles aussi (format version 3), avec leurs paramètres de calibration `scale`/`zero_point` — voir [Mémoire d'apprentissage](memory.md), « Mémoires quantifiées ».

Les runtimes `online_learning` **et** `training` relisent désormais `model_path` au démarrage pour reprendre un état sauvegardé (réseau, normalisation, optimiseur, mémoire) plutôt que d'en construire un neuf ; un `window_size` de configuration incompatible avec le modèle repris est rejeté explicitement plutôt que mélangé silencieusement. `optimizer_path`, `memory_path` et `metrics_path` sont également déjà consommés à l'écriture par les deux runtimes (`saveOnlineArtifacts`/`saveTrainingArtifacts`), en plus du fichier unifié.

Un artefact minimal distinct existe également pour le déploiement : `QuantizedNetworkSerialization` (magic `GLOOMYQN`, même schéma de robustesse — version, dimensions, checksum FNV-1a) persiste uniquement un `QuantizedNetwork` (poids/biais int8 + paramètres de calibration), sans aucune métadonnée d'entraînement. Il n'est pas destiné à reprendre l'entraînement, contrairement à `GLOOMY_MODEL` — voir [Quantification](quantization.md), « Artefact compact : QuantizedNetworkSerialization ».

### Métriques et benchmark

Métriques disponibles :

- MAE ;
- RMSE ;
- forgetting.

Le runner benchmark compare actuellement :

- baseline naïve `baseline_last_value` (dernière valeur connue, sans apprentissage) ;
- dataset complet (référence 100 epochs) ;
- FIFO, Reservoir, Prioritized, Novelty, Hybrid, chacune à 4 capacités (`32`, `64`, `128`, `256`) et 3 seeds ;
- FIFO int16 et FIFO int8, désormais aux mêmes 4 capacités et 3 seeds que le balayage float64 ;
- SGD, Momentum et Adam ;
- MSE, MAE et Huber ;
- temps d'entraînement, débit (samples/sec, updates/sec) et coût MAC approximatif ;
- latence d'inférence ;
- mémoire des paramètres et de l'état optimiseur ;
- mémoire d'apprentissage ;
- MAE, RMSE et pertes ;
- ratio MAE au dataset complet, par optimiseur et par perte ;
- moyenne, écart-type et intervalle de confiance à 95% du MAE sur les 3 seeds, par scénario borné.

Il contient aussi une expérience synthétique de catastrophic forgetting avec et sans replay FIFO. Chaque expérience produit son propre fichier CSV (`benchmark_baseline.csv`, `benchmark_full_dataset.csv`, `benchmark_memory_capacity.csv`, `benchmark_quantization.csv`, `benchmark_forgetting.csv`), en plus du `benchmark_results.csv` combiné.

## 3. Ce qui n'est pas encore fait

### Priorité haute : runtime d'apprentissage complet

1. **Format `GLOOMY_MODEL` unifié — fait via `ModelSerialization`**

   Le format unifié existe désormais dans `ModelSerialization` : il regroupe dans un seul fichier versionné le réseau, la normalisation, l'optimiseur et la mémoire d'apprentissage, avec un checksum global et un chargement vérifié.

   Les runtimes online et training l'utilisent désormais via `model_path` pour sauvegarder l'état entraîné au terme d'une exécution, et pour le relire au démarrage afin de reprendre un entraînement (voir point 4 ci-dessous). Les mémoires quantifiées sont désormais couvertes elles aussi (voir point 3) ; reste hors périmètre : les paramètres de quantification d'un réseau (poids/biais), qui ont leur propre artefact minimal distinct (`QuantizedNetworkSerialization`), volontairement séparé de `GLOOMY_MODEL` — voir [Quantification](quantization.md).

2. **Persistance de l'état des optimiseurs — fait**

   `OptimizerSerialization::save`/`load` persiste :

   - type d'optimiseur ;
   - learning rate ;
   - hyperparamètres (`momentum`, `beta1`, `beta2`, `epsilon`) ;
   - compte d'updates Adam ;
   - vitesses Momentum ;
   - premiers et seconds moments Adam.

   L'état est vérifié contre la forme des couches (nombre de couches, dimensions d'entrée/sortie) fournies à `load()` pour éviter de restaurer un buffer incompatible ; un fichier tronqué, corrompu ou de version différente est également rejeté. Le format `GLOOMY_MODEL` unifié couvre désormais ce composant, et les runtimes online et training l'utilisent via `model_path` pour sauvegarder (et relire au démarrage) l'état entraîné.

3. **Persistance des mémoires — fait, float64 et quantifiées**

   `LearningMemorySerialization::save`/`load` persiste, pour FIFO, Reservoir, Prioritized, Novelty et Hybrid :

   - stratégie utilisée et capacité ;
   - état RNG complet (pas seulement la seed) et compteur d'observations vues pour Reservoir, Prioritized et Hybrid ;
   - échantillons avec toutes leurs métadonnées (priorité, erreur, novelty, rarity, recency, diversity, âge, usage_count) ;
   - partitions Hybrid ;
   - paramètres propres à chaque stratégie (`alpha` pour Prioritized, `novelty_threshold` pour Novelty et Hybrid, ratios pour Hybrid).

   Fait également (format bumpé en version 3) : `QuantizedFIFOMemory` (int16) et `QuantizedInt8FIFOMemory` (int8), avec leurs paramètres de calibration `scale`/`zero_point` (partagés par tous les échantillons de la mémoire, écrits une seule fois) et leurs échantillons quantifiés. Voir [Mémoire d'apprentissage](memory.md), « Mémoires quantifiées ».

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

   **État** : `INFERENCE_RUNTIME` (comportement historique, inchangé) et `ONLINE_LEARNING_RUNTIME` sont faits. `runOnlineLearning()` (`src/headers/OnlineLearningRuntime.h`, `src/OnlineLearningRuntime.cpp`) implémente exactement la boucle ci-dessus en réutilisant les composants déjà testés séparément : `NeuralNetwork` (entrée de dimension `window_size`, sortie scalaire), `StreamingNormalizer`, une perte/un optimiseur/une mémoire construits depuis `GloomyConfig` (`loss`, `optimizer`, `memory_strategy` et leurs hyperparamètres), un `TrainingScheduler` (`EverySampleScheduler` ou `EveryNScheduler(train_every)`), et `LearningEngine::learn()` pour le replay et la mise à jour. Par défaut (`window_size=1`), chaque valeur consécutive de la séquence d'entrée devient une observation (`x[i]`) et sa cible (`x[i+1]`) ; avec `window_size > 1`, l'observation devient une fenêtre (`x[i..i+window_size-1]`) et la cible reste `x[i+window_size]`.

   Le CLI sélectionne ce runtime via la clé `runtime=online_learning` d'un fichier `-f`/`--config` (voir section « Configuration fichier » ci-dessous) ; `main.cpp` a été réorganisé pour faire circuler un unique `GloomyConfig` du parsing jusqu'au dispatch (`runInference`/`runOnline`), au lieu de cinq variables locales dispersées. Un runtime inconnu est rejeté avec un message explicite, de même qu'une perte, un optimiseur ou une stratégie de mémoire inconnus (tous les cas testés).

   `TRAINING_RUNTIME` (entraînement par epochs sur un jeu de données complet, via `LearningEngine::train()`) est maintenant exposé dans le CLI via `runtime=training`. Le runtime construit un réseau (même fenêtre `window_size`), normalise la séquence, entraîne sur le dataset complet puis sauvegarde l'état complet si `model_path` est fourni.

   Fait depuis : `window_size` rend la fenêtre d'entrée configurable (défaut `1`, comportement historique inchangé) ; `bin/gloomy` charge désormais `model_path` au démarrage pour **online_learning et training**, et reprend le réseau/la normalisation/l'optimiseur/la mémoire sauvegardés plutôt que d'en construire des neufs — un `window_size` incompatible avec le modèle repris est rejeté (`std::invalid_argument`) plutôt que mélangé silencieusement. Limites restantes : la sortie reste un flux `stdout` ligne par ligne, pas encore un format structuré ; la sortie du réseau reste toujours scalaire (seule la fenêtre d'entrée est configurable, pas de prédiction multi-pas).

### Configuration fichier

Aucune nouvelle option CLI n'a été ajoutée pour piloter l'architecture d'inférence (toujours `-c`/`-l`/`-n`/`-a`/`-A`) ; en revanche `-f`/`--config` sélectionne maintenant le runtime et ses hyperparamètres (voir point 6 ci-dessus et [Configurations et limites](configurations.md)).

À terme, un fichier de configuration est préférable à une commande contenant des dizaines d'options. L'idée proposée est :

```bash
./bin/gloomy -f gloomy.config
```

Le fichier pourrait contenir :

```ini
runtime=online_learning
window_size=1
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

**État** : les défauts sont maintenant centralisés dans `GloomyConfig` (`src/headers/GloomyConfig.h`, `src/GloomyConfig.cpp`). La structure reprend, champ par champ, le défaut déjà utilisé par chaque composant existant quand il en a un (`HuberLoss`, `MomentumOptimizer`, `AdamOptimizer`, `PrioritizedMemory`, `HybridMemoryRatios`, seed partagé de `std::mt19937`) et établit un défaut central documenté pour les champs qui n'en avaient pas encore (`learning_rate`, `memory_capacity`, `batch_size`, `window_size`, chemins de persistance). Le CLI (`src/main.cpp`) lit désormais ses cinq défauts actuels (`predictions`, `hidden_layers`, `neurons`, `activation`, `post_activation`) depuis `GloomyConfig::defaults()` au lieu de littéraux dupliqués ; le comportement du CLI est inchangé (vérifié manuellement). Un test caractérise chaque valeur pour empêcher une dérive silencieuse. Le parseur `-f/--config` couvre désormais tous les champs de `GloomyConfig`, et les deux runtimes (`online_learning`, `training`) consomment explicitement `model_path`/`optimizer_path`/`memory_path`/`metrics_path` pour sauvegarder l'état entraîné au terme de l'exécution, et relisent `model_path` au démarrage pour reprendre un entraînement s'il existe déjà.

Avant de l'implémenter, il faudra décider :

- format INI simple, JSON sans dépendance externe, ou format clé-valeur propriétaire ;
- priorité entre valeurs du fichier et options CLI ;
- validation et messages d'erreur ;
- valeurs par défaut centralisées ;
- support des secrets ou chemins externes ;
- compatibilité de version du fichier ;
- distinction entre configuration d'entraînement et configuration d'inférence.

Recommandation : commencer par un parseur clé-valeur INI minimal sans dépendance externe, avec priorité `CLI > fichier > défauts`. Ne pas appeler ce fichier `.env` au sens strict si ses valeurs ne sont pas destinées à être des variables d'environnement ; `gloomy.config` ou `gloomy.ini` serait plus explicite. Un alias `-f` peut néanmoins accepter n'importe quel chemin.

**État** : fait. `GloomyConfigFile::load()` (`src/headers/GloomyConfigFile.h`, `src/GloomyConfigFile.cpp`) implémente ce parseur clé-valeur minimal (pas de sections, pas de guillemets ; `#`/`;` en commentaire ; espaces trimés) et applique toutes les clés de `GloomyConfig` reconnues. `main.cpp` accepte `-f`/`--config PATH` avec la priorité `CLI > fichier > défauts` : le fichier est appliqué en première passe par-dessus les défauts, puis les flags `-c`/`-l`/`-n`/`-a`/`-A` explicites sont appliqués en seconde passe et l'emportent toujours. Une clé inconnue, une ligne malformée, une valeur numérique invalide ou un fichier introuvable sont rejetés avec un message explicite. `runtime=online_learning` utilise désormais les hyperparamètres déclarés dans le fichier, et `model_path` est consommé pour sauvegarder le modèle unifié au terme de l'exécution. Voir [Configurations et limites](configurations.md) pour le détail et des exemples.

### Priorité moyenne : robustesse mathématique

- ~~gradient checking généralisé à toutes les activations et plusieurs couches~~ fait : `checkNetworkGradient` (`tests/loss_tests.cpp`) compare gradient analytique et différence centrée sur un réseau à 3 couches, pour `none`/`sigmoid`/`relu`/`leaky_relu`/`tanh` (`testGradientCheckingAllActivations`). `sigmoid_derivative`/`tanh_derivative` restent exclues car `DenseLayer::backward` les rejette déjà comme activations d'entraînement ;
- ~~tests de gradient softmax multi-sortie~~ fait : `testSoftmaxGradientCheck` exerce une couche de sortie à 3 neurones avec `post_algorithm=softmax`, y compris le terme croisé du gradient softmax, jamais couvert par les tests à une seule sortie ;
- ~~tests de stabilité avec très grandes valeurs~~ fait : `testActivationStabilityWithLargeValues` vérifie que sigmoid/tanh/relu/leaky_relu restent finis (forward et gradients) pour des entrées `±1e8` ;
- ~~test de reproductibilité avec seed injectable~~ fait : `DenseLayer::seedWeightInitialization(seed)` (voir [Couches et neurones](architecture.md), section « Initialisation des poids et reproductibilité »), appelé par le CLI avec `GloomyConfig::seed` une fois la configuration résolue. `testDenseLayerSeededInitialization` vérifie qu'un même seed produit des poids identiques et que deux seeds différents en produisent des différents ;
- ~~remplacement de `rand()` par un générateur contrôlable~~ fait, dans le même changement : `DenseLayer` utilise maintenant `std::mt19937` + `std::uniform_real_distribution` au lieu de `rand()`/`RAND_MAX` process-global. `BenchmarkRunner` a été mis à jour pour utiliser `DenseLayer::seedWeightInitialization(1234)` à la place de `std::srand(1234)`, avec le même effet (vérifié : les colonnes de perte/MAE/RMSE du benchmark restent bit-identiques d'un run à l'autre, seules les colonnes de temps varient, comme avant) ;
- ~~validation systématique des valeurs non finies dans tous les composants~~ fait : `LossFunction::compute`/`gradient` et `DenseLayer::forward`/`backward` rejettent `NaN`/infini en entrée avec un message clair (`testNonFiniteValuesRejected`), comme le faisait déjà `StreamingNormalizer`. `SGDOptimizer`/`MomentumOptimizer`/`AdamOptimizer::update` rejettent désormais aussi tout gradient de poids/biais non fini et tout `gradient_scale` non fini ou non positif (`validateFiniteGradients`, partagée entre les trois — voir [Optimiseurs](optimizers.md)) : un cas réel non couvert par la validation d'entrée de `backward()` est le dépassement de capacité du `double` pendant ses propres multiplications/accumulations internes, avec des opérandes finis mais extrêmes. Toutes les stratégies de mémoire (FIFO, Reservoir, Prioritized, Novelty, Hybrid, `QuantizedFIFOMemory`, `QuantizedInt8FIFOMemory`) rejettent désormais aussi, dans `add()`, un `TrainingSample` dont `input`/`target` est vide ou non fini (`validateTrainingSampleVectors`, partagée — voir [Mémoire d'apprentissage](memory.md)) ; avant ce changement, seule `NoveltyMemory` validait la non-vacuité/dimension, et `PrioritizedMemory` ne validait que `priority`.

### Priorité moyenne : mémoire et continual learning

- ~~mise à jour de toutes les composantes du score d'importance~~ fait : `LearningEngine` calcule désormais réellement `recency` (`1/(1+age)`) et `rarity` (`1/(1+usage_count)`) à l'ajout, plus `novelty`/`diversity` (par rapport aux autres échantillons du même batch de replay) dans `trainFromMemory()`. Sous les poids par défaut (`error=1.0`, le reste à `0.0`), la priorité reste exactement l'erreur seule — comportement historique inchangé ; un `LearningEngine` construit avec des `ImportanceWeights` non par défaut peut désormais réellement s'appuyer sur les quatre autres composantes. Voir [Mémoire d'apprentissage](memory.md), « Score d'importance », et `testImportanceScoreComponents` ;
- ~~vraie récence fondée sur l'âge plutôt que l'alternance actuelle de Hybrid~~ fait : `choosePartition()` n'alterne plus par `seen_samples % 2` — toute observation générique (ni erreur, ni nouveauté) rejoint directement `recent`. Lorsque `recent` est pleine, elle évince son membre le plus ancien **par `TrainingSample::age` réel** (pas par position dans le vecteur, qui ne reflétait plus l'âge après un premier remplacement en place — un vrai bug corrigé au passage) et le **promeut** dans `historical` au lieu de le perdre ; `error`/`novelty` bénéficient de la même correction d'éviction par âge réel. Supprimé au passage : `removeOldestFromPartition`, du code mort jamais appelé. Voir [Mémoire d'apprentissage](memory.md), « Vraie récence fondée sur l'âge », et `testHybridMemoryTrueRecency` ;
- ~~mise à jour des priorités avec correction de biais d'échantillonnage~~ fait : `PrioritizedMemory` calcule un poids d'importance-sampling `(N·P(i))^(-beta)` (normalisé au maximum du batch) pour chaque échantillon tiré par `sampleIndexed()` ; `beta` est un paramètre de construction (défaut `0.4`, `0` désactive la correction). `LearningEngine::trainFromMemory` applique ce poids à la contribution de chaque échantillon au gradient via `MemoryEntry::importance_weight` (`1.0` par défaut, donc neutre pour les autres stratégies) et la méthode privée `trainWeightedBatch`, dont `trainBatch` est un cas particulier (tous les poids à `1.0`). Persisté par `LearningMemorySerialization`. Voir [Mémoire d'apprentissage](memory.md), « Correction de biais d'échantillonnage », et les tests `testPrioritizedMemoryBiasCorrection`/`testLearningEngineTrainBatchWeighting` ;
- ~~annealing de `beta` au fil de l'entraînement~~ fait : nouveau paramètre `beta_annealing_rate` (défaut `0.0`, comportement inchangé), ajouté à `beta` après chaque `sampleIndexed()`, plafonné à `1.0`. Persisté (format `LearningMemorySerialization` bumpé en version 4). Voir [Mémoire d'apprentissage](memory.md), « Annealing de beta », et `testPrioritizedMemoryBetaAnnealing` ;
- ~~exploration contrôlée des échantillons de faible priorité~~ fait : nouveau paramètre `exploration_epsilon` (défaut `0.0`, comportement inchangé — `discrete_distribution` est invariant à un facteur d'échelle uniforme, donc `epsilon=0` reproduit le même tirage RNG qu'avant) qui mélange la distribution priorisée avec une distribution uniforme, garantissant qu'aucun échantillon n'a une probabilité de tirage strictement nulle. Persisté (même bump de version). Voir [Mémoire d'apprentissage](memory.md), « Exploration contrôlée », et `testPrioritizedMemoryExplorationEpsilon` ;
- prototypes / coreset — pas fait ;
- mémoire par régimes — pas fait ;
- ~~détection légère de concept drift (mécanisme de détection actif)~~ fait : `ConceptDriftDetector` compare la moyenne récente d'erreur à une ligne de base historique (moyenne + écart-type, Welford) et signale activement une dérive (pas seulement une métrique observée après coup). Branché en option (`concept_drift_detection`, défaut `false`) dans `runOnlineLearning()`, qui déclenche alors un replay supplémentaire immédiat depuis la mémoire — un signal qui produit une action réelle, pas juste un indicateur. Vérifié sur un changement de régime synthétique net : la dérive est signalée peu après la transition et s'éteint une fois le réseau réadapté. Voir [Mémoire d'apprentissage](memory.md), « Détection de concept drift », et `testConceptDriftDetector`/`testOnlineLearningRuntimeConceptDriftDetection`. Reste ouvert : la réaction actuelle (un replay supplémentaire) est simple ; une politique plus riche (mémoire par régimes déclenchée par la dérive, ajustement du learning rate, etc.) reste à explorer ;
- règles adaptatives explicites avant tout mécanisme appris ;
- ~~expériences A -> B -> A plus nombreuses et reproductibles~~ fait pour un premier cas reproductible : `testCatastrophicForgettingWithoutReplay`, `testCatastrophicForgettingMitigatedByReplay` et `testConceptDriftReturnToPreviousRegime` (`tests/loss_tests.cpp`) reprennent les régimes synthétiques de `BenchmarkRunner.cpp` (droite croissante puis décroissante) avec une seed fixe (`4242`), et vérifient par assertion numérique — pas seulement en observant un CSV — que : l'oubli sans replay est réel (perte sur A après B `> 10`, contre `< 0.01` avant) ; le replay FIFO réduit cet oubli d'au moins moitié (calibré sur des valeurs observées ~4× sur plusieurs seeds avant d'écrire le test) ; et qu'un cycle complet A→B→A permet au réseau de **retrouver** une performance sur A comparable à l'origine (`< 0.01`), donc de s'adapter à un retour de régime plutôt que de rester durablement dégradé. Reste ouvert : des régimes plus variés, et un vrai cycle A→B→A→B pour observer si l'adaptation se dégrade avec les répétitions.

### Priorité moyenne : benchmark scientifique

**État : fait dans son ensemble.** `BenchmarkRunner` couvre désormais :

- ~~capacités `32`, `64`, `128`, `256`~~ fait pour les 5 stratégies float64 (FIFO, Reservoir, Prioritized, Novelty, Hybrid) ;
- ~~Novelty et Hybrid~~ fait, dans le même changement que les capacités ;
- ~~float64, int16 et int8 sur les mêmes données, même balayage~~ fait : int16/int8 comparés au même dataset, aux 3 pertes, et désormais aux **mêmes 4 capacités et 3 seeds** que float64 (4 × 2 précisions × 3 optimiseurs × 3 pertes × 3 seeds = 216 scénarios, avec moyenne/écart-type/IC95 du MAE comme le balayage float64). Les mémoires quantifiées étant déterministes (pas de tirage aléatoire interne), la seed ne fait varier ici que l'initialisation des poids, comme pour FIFO dans le balayage float64. `float32` n'est pas couvert (reste non fait, voir plus bas) ;
- ~~pertes MSE, MAE et Huber~~ fait : `makeLoss()` construit la perte demandée ; le balayage de capacités, le dataset complet et les scénarios quantifiés tournent désormais chacun sur les 3 pertes (nouvelle colonne `loss_function`). Le balayage de capacités seul est donc 4 capacités × 5 stratégies × 3 optimiseurs × 3 pertes × 3 seeds = 540 scénarios (contre 60 avant) ; l'ensemble du runner reste sous la seconde ;
- ~~plusieurs seeds~~ fait pour le balayage de capacités float64 **et** désormais quantifié (3 seeds : `1234`, `2345`, `3456`). Reste non fait : dataset complet, toujours à seed unique (sa perte est déjà proche de zéro, moins sensible à la variance d'initialisation) ;
- ~~moyenne et écart-type~~ fait : `mae_mean`/`mae_stddev` (écart-type population, diviseur N=3), calculées sur le MAE des 3 seeds d'un même scénario (capacité × stratégie × optimiseur × perte) et dupliquées sur chaque ligne du groupe ;
- ~~intervalles de confiance~~ fait : nouvelle colonne `mae_ci95_margin` (demi-largeur de l'IC à 95%, loi de Student `df=2`, écart-type d'**échantillon** — diviseur N-1, différent de `mae_stddev` qui reste population). La valeur critique de Student est codée en dur pour `N=3` seeds spécifiquement (`t_critical_95_df2` dans `BenchmarkRunner.cpp`) ; elle vaut `0` si le nombre de seeds change un jour sans mise à jour de cette constante, plutôt que d'utiliser silencieusement une valeur fausse. À lire comme un ordre de grandeur (échantillon de taille 3), pas une garantie statistique forte ;
- ~~baseline dernière valeur connue~~ fait : ligne `baseline_last_value` (calculée avant toute chose, sans RNG donc sans effet sur les scénarios suivants), évaluée avec les 3 pertes. Sert de plancher de comparaison — voir [Benchmark](benchmark.md) ;
- ~~ratio `performance_memory_limited / performance_full_dataset`~~ fait : colonne `mae_ratio_to_full_dataset`, calculée par rapport au MAE `full_dataset` du même optimiseur **et de la même perte**. **Limite importante inchangée** : `full_dataset` entraîne 100 epochs en batch (MAE proche de zéro) alors que les scénarios bornés font un seul passage online ; le ratio mélange donc l'effet du nombre de passages et celui de la capacité mémoire (valeurs parfois de l'ordre du million). Isoler l'effet de la seule capacité, à nombre de passages égal, reste à faire ;
- ~~coût CPU et nombre d'opérations approximatif~~ fait : colonne `approximate_macs` (MAC du forward × 3 comme approximation forward+backward, grossier par construction — ignore le coût de l'optimiseur et des activations) ;
- ~~samples/sec et updates/sec~~ fait : colonnes `samples_per_second`/`updates_per_second`, dérivées de `training_time_ms` ;
- ~~fichiers CSV séparés par expérience~~ fait : `benchmark_baseline.csv`, `benchmark_full_dataset.csv`, `benchmark_memory_capacity.csv`, `benchmark_quantization.csv`, `benchmark_forgetting.csv`, en plus du `benchmark_results.csv` combiné conservé pour compatibilité (`docs/examples.md` l'utilise).

Reste non fait : dataset complet à seed unique (pas de moyenne/écart-type/IC pour cette section) ; baseline `float32` ; jeux de données réels (aucun disponible dans cet environnement sans accès réseau).

Au passage : le format CSV (colonnes `loss_function`, `mae_ci95_margin`, `approximate_macs`, `samples_per_second`, `updates_per_second`) reste rétrocompatible en ajout de colonnes, sauf `loss_function` insérée en 4e position (avant les colonnes numériques) — la vérification par motif dans [Benchmark](benchmark.md) démarre donc à la colonne 5, pas 4. `testBenchmarkCsv` couvre tous les nouveaux champs. Chaque extension du benchmark scientifique décale à nouveau les valeurs de l'expérience de forgetting utilisées dans [docs/examples.md](examples.md) (le point de réinitialisation `DenseLayer::seedWeightInitialization(1234)` reste juste avant cette expérience, mais tout code exécuté avant ce point change l'historique de tirages qui y mène) — mis à jour et revérifié stable après le balayage capacités × seeds des scénarios quantifiés ajouté ici.

### Priorité moyenne : compression et embarqué

- ~~compression différentielle des séries temporelles~~ fait : `DeltaQuantizer` quantifie les différences consécutives d'une série plutôt que ses valeurs brutes. Mesuré sur trois scénarios réels : un gain net sur une série lisse (offset type capteur, ~15× moins d'erreur) et sur une rampe parfaitement linéaire (deltas constants, erreur quasi nulle) ; un léger désavantage sur une série bruitée/erratique (deltas aussi larges que la série, plus l'erreur s'accumule le long de la reconstruction) — un compromis réel, pas un gain systématique, à mesurer sur les données réellement visées. Voir [Quantification](quantization.md), « Compression différentielle » ;
- ~~stockage sans allocations pendant la boucle online~~ partiel : la boucle de `runOnlineLearning()` et celle de `runTraining()` (`OnlineLearningRuntime.cpp`) réutilisent désormais leurs buffers (`raw_observation`, `raw_target`, `observation`, `target`, et `TrainingSample sample` pour le runtime online) d'une itération à l'autre au lieu de les reconstruire — une fois leur capacité établie à la première itération, `vector::assign`/`resize`/`operator=` sur une taille identique ne réallouent plus (même principe déjà utilisé par `DenseLayer` pour ses buffers internes). Ajouté à cette occasion : `StreamingNormalizer::normalize(values, out)`, une surcharge en place (la surcharge par valeur existante délègue désormais à celle-ci, sans changement de comportement). Vérifié par la suite de tests inchangée (comportement identique, y compris la persistance) et un nouveau test `testOnlineLearningRuntimeLongSequenceStability` (500 pas). **Ce qui reste alloué à chaque pas** : le retour par valeur de `NeuralNetwork::forward()` (et le chaînage interne couche par couche), `LossFunction::gradient()`, et les vecteurs internes à `LearningEngine`/`LearningMemory` (`entries`, `batch`, `sample_weights` dans `trainFromMemory`) — non touchés cette fois, plus risqués à changer sans revoir leurs signatures publiques ;
- arena allocator ou capacité statique ;
- buffers contigus (partiellement gagné pour les buffers ci-dessus ; `DenseLayer` utilise toujours des `std::vector<std::vector<double>>` imbriqués pour ses poids, non contigus — voir « Limites connues ») ;
- ~~quantification des poids~~ fait : `NetworkQuantization` quantifie en int8, couche par couche, les poids et biais d'un `NeuralNetwork` déjà entraîné (calibration séparée poids/biais). Mesuré sur un réseau 1-8-1 (seed `4242`, 80 epochs, y=2x+1) : erreur absolue max ≈ `0.049`, erreur relative max ≈ `0.23 %`, mais ratio mémoire réel de seulement **2.25×** (pas 8×) — le coût fixe de calibration (`scale`/`zero_point` par vecteur) domine sur un réseau aussi petit ; ce compromis doit être remesuré pour toute architecture visée. Voir [Quantification](quantization.md), « Quantification des poids d'un réseau entraîné » ;
- ~~runtime inference-only minimal~~ fait : `bin/gloomy_infer` (`src/InferenceOnlyMain.cpp`), compilé via `make infer`, ne dépend que de `DenseLayer`/`NeuralNetwork`/`NetworkSerialization`/`Quantization`/`Int8Quantization`/`NetworkQuantization`/`QuantizedNetworkSerialization` — pas de `LearningEngine`, `Optimizer`, `LearningMemory` ni `GloomyConfig`. Mesuré : `bin/gloomy` 379648 octets (317752 stripped) contre `bin/gloomy_infer` 97656 octets (80184 stripped), soit ~74–79 % de réduction, portée par le segment `.text` (302429 → 69169 octets). Vérifié de bout en bout : entraînement → export float64 et int8 → chargement et inférence réels via ce binaire pour les deux formats. Voir [Quantification](quantization.md), « Runtime d'inférence minimal » ;
- ~~génération d'un artefact modèle sans métadonnées inutiles~~ fait : `QuantizedNetworkSerialization` (magic `GLOOMYQN`) ne persiste que les poids/biais int8 et leurs paramètres de calibration, sans état d'optimiseur ni de mémoire d'apprentissage — distinct du format unifié `GLOOMY_MODEL` qui, lui, embarque tout pour reprendre l'entraînement. Voir section 2, « Persistance » ;
- compilation et tests sur une cible embarquée réelle — non fait : aucune chaîne de compilation croisée (`arm-none-eabi-gcc`, `arm-linux-gnueabihf-gcc`, `avr-gcc`, `riscv64-unknown-elf-gcc`) n'est disponible dans cet environnement ;
- mesure RAM, Flash, CPU et énergie — non fait, pour la même raison (nécessite un matériel réel).

### Priorité basse : extensions

- optimisations SIMD — **investigué, pas appliqué** : passer `-O3` (au lieu de `-O2`) sur l'ensemble du projet donne une sortie de benchmark bit-identique (colonnes non temporelles, vérifié sur l'ensemble de `benchmark_results.csv`) et un gain de vitesse mesuré modeste (~6%, 0.421s contre 0.447s pour `make benchmark`, un run déjà sous la demi-seconde). En contrepartie, `-O3` fait grossir sensiblement les binaires (`bin/gloomy` stripped : 420 152 octets contre 317 752 ; `bin/gloomy_infer` stripped : 88 376 contre 80 184), ce qui contredirait directement l'objectif « le plus petit possible » du binaire d'inférence embarqué (point 12, voir [Quantification](quantization.md)). Non appliqué : le gain ne justifie pas le compromis de taille pour ce projet, et le vrai levier SIMD (vecteurs de poids non contigus) est le même que celui déjà différé au point « arena allocator / buffers contigus » ;
- multithreading PC — **investigué, pas appliqué** : les scénarios du benchmark sont indépendants, mais `DenseLayer::seedWeightInitialization` repose sur **un générateur statique partagé** entre tous — le paralléliser tel quel introduirait une course (data race) sur cet état partagé et casserait la reproductibilité (l'ordre des tirages ne serait plus déterministe), la propriété la plus soigneusement préservée dans tout ce projet. Le paralléliser correctement demanderait d'abord de découpler l'initialisation des poids de ce générateur global partagé — un changement d'architecture de la même nature (et du même risque) que « arena allocator / buffers contigus », pas fait ici. Le volume de travail actuel (~0.45s au total) ne justifie de toute façon pas le risque pour l'instant ;
- ~~Adam quantifié ou optimiseur à état compressé~~ fait : `CompressedAdamOptimizer` conserve les moments Adam quantifiés en int16 entre deux appels à `update()` (recalibrés à chaque pas, le calcul lui-même reste en double). Mesuré sur un réseau 1-8-1 : convergence quasi identique à Adam natif (perte finale `~0.000001` pour les deux), mais un gain mémoire de seulement **1.75×** (228 contre 400 octets), pas le 4× théorique — même constat de coût fixe de calibration que `NetworkQuantization`. Voir [Optimiseurs](optimizers.md), « Adam à état compressé » ;
- classification multi-classe avec sortie `K` neurones — **reste non fait** : le réseau supporte déjà une sortie à `K` neurones avec `softmax`, et dispose désormais d'une perte adaptée (`CrossEntropyLoss`, voir ci-dessous), mais aucun runtime CLI ne les combine — les runtimes `online_learning`/`training` sont structurellement des runtimes de régression scalaire (`window_size` valeurs en entrée, un seul scalaire en sortie). Une vraie pipeline de classification (encodage/décodage des classes, métrique d'exactitude) resterait un chantier distinct ;
- ~~pertes classification et cross-entropy~~ fait : `CrossEntropyLoss`, vérifiée par différence finie sur un réseau à sortie softmax (même principe que `testSoftmaxGradientCheck`). Non branchée dans le CLI, pour la raison ci-dessus. Voir [Fonctions de perte](losses.md), « Cross-entropy » ;
- ~~détection de dérive plus avancée~~ fait : `PageHinkleyDetector`, un test séquentiel de détection de rupture classique de la littérature (Page-Hinkley), distinct de `ConceptDriftDetector` dans son principe. Détecte le même changement de régime synthétique en 0 à 4 pas (contre plusieurs dizaines pour `ConceptDriftDetector`, qui doit d'abord remplir sa fenêtre récente), sans faux positif sur un bruit stable. Pas encore branché dans `runOnlineLearning()` (existe comme primitive testée indépendante). Voir [Mémoire d'apprentissage](memory.md), « Détection de dérive plus avancée » ;
- adaptation dynamique apprise — **reste non fait** : un contrôleur qui apprendrait (plutôt que suivrait des règles fixes) comment réagir à une dérive détectée est un chantier de recherche distinct, non commencé.

## 4. Ordre recommandé pour la suite

### Étapes déjà réalisées (historique)

1. ~~Stabiliser la sérialisation unifiée du modèle.~~ Fait : `ModelSerialization` regroupe réseau, normalisation, optimiseur et mémoire dans un fichier `GLOOMY_MODEL` versionné et protégé par checksum, avec round-trip et rejet de corruption testés, et est branché dans le CLI via `model_path` (voir section 2, « Persistance »). Les mémoires quantifiées y sont désormais couvertes aussi (voir « Prochaines étapes recommandées », point 1 ci-dessous).
2. ~~Ajouter la persistance de l'état Optimizer.~~ Fait (`OptimizerSerialization`, voir section 2 et 3).
3. ~~Ajouter la persistance des mémoires~~ Fait pour la représentation float64 **et** quantifiée (int16/int8) — voir `LearningMemorySerialization`, section 2 et 3, et « Prochaines étapes recommandées », point 1.
4. ~~Centraliser les défauts dans une configuration C++.~~ Fait (`GloomyConfig`, voir section 2 et 3).
5. ~~Ajouter `-f/--config` au CLI avec priorité CLI > fichier > défauts.~~ Fait (`GloomyConfigFile`, voir section 2 et 3).
6. ~~Exposer un mode online fonctionnel dans le CLI.~~ Fait pour `ONLINE_LEARNING_RUNTIME` (`OnlineLearningRuntime`, voir section 2 et 3).
7. ~~Brancher la persistance du modèle entraîné dans le runtime online.~~ Fait via `ModelSerialization` et `model_path` dans le CLI.
8. ~~Ajouter `TRAINING_RUNTIME` dans le CLI.~~ Fait via `runtime=training`, avec `epochs` et sauvegarde de `model_path`.
9. ~~Étendre les benchmarks aux capacités et stratégies restantes.~~ Fait dans son ensemble : capacités `32`/`64`/`128`/`256`, stratégies Novelty/Hybrid, 3 pertes (MSE/MAE/Huber), 3 seeds avec moyenne/écart-type/IC95 du MAE, ratio au dataset complet, baseline naïve `baseline_last_value`, coûts CPU/débit approximatifs, et fichiers CSV séparés par expérience (voir section 3, « Priorité moyenne : benchmark scientifique »). Restent : seeds multiples et balayage de capacités pour le dataset complet et les scénarios quantifiés, baseline `float32`, jeux de données réels.
10. ~~Ajouter les tests de concept drift et catastrophic forgetting.~~ Fait : `testCatastrophicForgettingWithoutReplay`, `testCatastrophicForgettingMitigatedByReplay` et `testConceptDriftReturnToPreviousRegime` (`tests/loss_tests.cpp`), voir section 3, « Priorité moyenne : mémoire et continual learning ».
11. ~~Optimiser les allocations et la représentation mémoire.~~ Fait pour la boucle du runtime online : `OnlineLearningRuntime.cpp` réutilise ses buffers d'une itération à l'autre au lieu de les reconstruire, voir section 3, « Priorité moyenne : compression et embarqué ». Le reste de la représentation mémoire (couches en vecteurs imbriqués, `TrainingSample` à deux `std::vector` même pour un scalaire) n'est pas touché.
12. ~~Préparer le runtime embarqué et la quantification des poids.~~ Fait pour les parties réalisables sans matériel dédié : quantification int8 des poids/biais post-entraînement (`NetworkQuantization`), artefact compact dédié (`QuantizedNetworkSerialization`, magic `GLOOMYQN`) et runtime d'inférence minimal (`bin/gloomy_infer`, ~75 % plus petit que `bin/gloomy`), voir section 3, « Priorité moyenne : compression et embarqué » et [Quantification](quantization.md). Restent non faits, faute de matériel/outillage embarqué disponible ici : buffers contigus/arena allocator pour le chemin d'inférence, compilation et tests sur une cible embarquée réelle, mesure RAM/Flash/CPU/énergie ; la compression différentielle des séries temporelles reste également hors périmètre de ce point.

### Prochaines étapes recommandées

Cette liste remplace l'ancienne numérotation 1-12 ci-dessus (désormais entièrement réalisée) comme référence pour la suite. Elle reprend, triés par priorité décroissante, les points encore ouverts identifiés section 3.

**Priorité haute — toutes faites dans cette itération**

1. ~~Étendre `GLOOMY_MODEL`/`LearningMemorySerialization` aux mémoires quantifiées~~ Fait : `QuantizedFIFOMemory` et `QuantizedInt8FIFOMemory` sont persistées (format version 3) avec leurs paramètres `scale`/`zero_point`, écrits une seule fois par mémoire — voir [Mémoire d'apprentissage](memory.md), « Mémoires quantifiées ».
2. ~~Brancher `optimizer_path`, `memory_path` et `metrics_path` dans le CLI principal~~ En réalité déjà fait avant cette itération : `saveOnlineArtifacts`/`saveTrainingArtifacts` (`OnlineLearningRuntime.cpp`) consommaient déjà les quatre chemins pour les deux runtimes — cette entrée de la liste précédente reposait sur une lecture incorrecte du code, corrigée ici.
3. ~~Permettre à `bin/gloomy` de charger un modèle sauvegardé au démarrage~~ Fait pour `runtime=training` (déjà fait avant cette itération pour `runtime=online_learning`, via la même lecture incorrecte que le point précédent) : `runTraining()` accepte désormais une variante à trois arguments (`model_path`) utilisée par `main.cpp`, qui reprend réseau/normalisation/optimiseur/mémoire sauvegardés plutôt que d'en construire des neufs. `bin/gloomy_infer` reste le seul binaire à charger un modèle pour de l'inférence pure, sans reprendre l'entraînement.
4. ~~Rendre configurable la fenêtre d'entrée/sortie du runtime online~~ Fait pour la fenêtre d'**entrée** : nouveau champ `window_size` (défaut `1`, comportement historique inchangé) dans `GloomyConfig`, consommé par `runOnlineLearning()` et `runTraining()` pour construire un réseau à `window_size` entrées et faire glisser une fenêtre sur la séquence. Un `window_size` incompatible avec un modèle repris (`model_path`) est rejeté explicitement. La sortie reste un scalaire (un seul pas de sortie) : la fenêtre de **sortie** (prédiction multi-pas) n'a pas été touchée.
5. ~~Compléter la validation des valeurs non finies~~ Fait : `Optimizer::update` (SGD/Momentum/Adam) rejette les gradients de poids/biais non finis et un `gradient_scale` non fini/non positif ; les sept stratégies de mémoire (y compris les deux quantifiées) rejettent un `TrainingSample` vide ou non fini à l'ajout — voir section 3, « Priorité moyenne : robustesse mathématique ».

**Priorité moyenne — 6, 8, 9, 11 faits dans cette itération ; 7 partiellement fait ; 10 et 12 restent ouverts**

6. ~~Détection active de concept drift~~ Fait : `ConceptDriftDetector`, branché en option (`concept_drift_detection`) dans `runOnlineLearning()`, déclenche un replay supplémentaire dès qu'une dérive est signalée — un signal qui produit une action réelle. Voir section 3, « Priorité moyenne : mémoire et continual learning », et [Mémoire d'apprentissage](memory.md).
7. Mémoire par régimes, prototypes/coreset — **restent non faits** (nécessiteraient une nouvelle stratégie `LearningMemory`, plus substantielle qu'un ajout de paramètre). ~~Exploration contrôlée des échantillons de faible priorité~~ et ~~mise à jour de toutes les composantes du score d'importance~~ sont faites (voir section 3, même section).
8. ~~Annealing de `beta` (Prioritized Replay)~~ Fait : `beta_annealing_rate` (défaut `0.0`, comportement inchangé) fait croître `beta` vers `1.0` après chaque replay. Voir [Mémoire d'apprentissage](memory.md).
9. Étendre le benchmark scientifique aux angles morts restants — **partiellement fait** : ~~seeds multiples et balayage de capacités pour int16/int8~~ fait (mêmes 4 capacités × 3 seeds que float64, 216 scénarios). Restent non faits : baseline `float32`, et surtout des jeux de données réels (aucun disponible dans cet environnement sans accès réseau — les données synthétiques actuelles ne remplacent pas une évaluation réelle).
10. Arena allocator ou buffers contigus pour le reste du chemin chaud : `DenseLayer` (poids en vecteurs imbriqués), le retour par valeur de `NeuralNetwork::forward()` et `LossFunction::gradient()`, et les vecteurs internes de `LearningEngine`/`LearningMemory` (`entries`, `batch`, `sample_weights`) — **reste non fait**, volontairement : un changement invasif qui toucherait `Optimizer`, tous les sérialiseurs, `BenchmarkRunner` et une bonne partie de la suite de tests, à faire dans une étape dédiée et prudente plutôt qu'en marge d'une itération déjà large — voir section 3, « Priorité moyenne : compression et embarqué ».
11. ~~Compression différentielle des séries temporelles~~ Fait : `DeltaQuantizer`, un gain réel mais pas systématique (mesuré sur trois scénarios) — voir [Quantification](quantization.md).
12. Compilation/tests sur une cible embarquée réelle et mesure RAM/Flash/CPU/énergie — **reste bloqué** dans cet environnement (aucune chaîne de compilation croisée ni matériel disponible) ; à reprendre dès qu'un environnement adapté existe.

**Priorité basse — 15, 16 (perte) et 17 (détection) faits ; 13, 14 investigués et volontairement pas appliqués ; 16 (pipeline classification) et 17 (adaptation apprise) restent ouverts**

13. Optimisations SIMD — **investigué, pas appliqué** : `-O3` donne une sortie bit-identique mais des binaires ~30% plus gros, contradictoire avec l'objectif de taille de `bin/gloomy_infer` (point 12). Voir section 3, « Priorité basse : extensions ».
14. Multithreading PC — **investigué, pas appliqué** : paralléliser les scénarios indépendants du benchmark demanderait d'abord de découpler `DenseLayer::seedWeightInitialization` de son générateur statique partagé (actuellement une source de course de données et de non-déterminisme si parallélisé tel quel) — un changement de la même nature que l'arena allocator, pas fait ici. Voir section 3, « Priorité basse : extensions ».
15. ~~Adam quantifié ou optimiseur à état compressé~~ Fait : `CompressedAdamOptimizer`. Voir [Optimiseurs](optimizers.md).
16. Classification multi-classe (sortie `K` neurones) — **reste non fait** (pipeline CLI complète) ; ~~pertes classification/cross-entropy~~ Fait : `CrossEntropyLoss`, vérifiée par différence finie. Voir [Fonctions de perte](losses.md).
17. ~~Détection de dérive plus avancée~~ Fait : `PageHinkleyDetector` (Page-Hinkley), voir [Mémoire d'apprentissage](memory.md). Adaptation dynamique apprise — **reste non fait**, chantier de recherche distinct.

## 5. Limites connues à ne pas oublier

- le CLI (`bin/gloomy`) recrée actuellement le réseau entre prédictions autorégressives, avec de nouveaux poids aléatoires (`INFERENCE_RUNTIME` uniquement ; sans effet sur `ONLINE_LEARNING_RUNTIME`, qui garde un seul réseau du début à la fin) ;
- le réseau des runtimes `online_learning`/`training` a une sortie toujours scalaire (un seul pas de prédiction) ; l'entrée est configurable via `window_size` (défaut `1`, comportement historique), mais il n'existe pas de prédiction multi-pas ni de séries multivariées ;
- `softmax` sur la sortie actuelle à un neurone vaut toujours `1` ;
- les couches utilisent encore des vecteurs imbriqués et des allocations dynamiques ;
- les mémoires natives stockent encore des `double` ;
- le réseau s'entraîne toujours en float64 ; seule une copie post-entraînement peut être quantifiée en int8 via `NetworkQuantization`, avec un ratio mémoire réel dépendant fortement de la taille du réseau (mesuré à 2.25× sur un petit réseau 1-8-1, loin du 8× théorique) — voir [Quantification](quantization.md) ;
- `bin/gloomy_infer` ne couvre que le chemin `forward()` ; les couches y utilisent toujours des vecteurs imbriqués non contigus, et aucune cible embarquée réelle n'a pu être testée (pas de chaîne de compilation croisée ni de matériel disponible) ;
- le format `GLOOMY_MODEL` unifié (`ModelSerialization`) couvre réseau, normalisation, optimiseur et mémoire d'apprentissage (natives et quantifiées) ; il ne couvre pas les poids quantifiés d'un réseau, qui ont leur propre artefact minimal et distinct (`QuantizedNetworkSerialization`, magic `GLOOMYQN`), volontairement dépourvu de métadonnées d'entraînement et donc non destiné à reprendre un entraînement — voir [Quantification](quantization.md) ;
- `DeltaQuantizer` (compression différentielle) n'est pas un gain systématique : sur une série bruitée/erratique dont les deltas n'ont pas une plage plus étroite que les valeurs elles-mêmes, il fait légèrement pire qu'une quantification directe, et sa reconstruction par sommes cumulées accumule l'erreur le long de la série — voir [Quantification](quantization.md) ;
- la détection de concept drift (`ConceptDriftDetector`) réagit par un seul mécanisme simple (un replay supplémentaire immédiat) ; ni mémoire par régimes, ni prototypes/coreset, ni règles adaptatives plus riches ne sont implémentés ; `PageHinkleyDetector`, une seconde méthode de détection plus réactive, existe comme primitive testée mais n'est pas encore branchée dans `runOnlineLearning()` ; aucune « adaptation dynamique apprise » (un contrôleur appris, pas seulement des règles fixes) n'est implémentée ;
- `CompressedAdamOptimizer` n'est pas encore reconnu par `OptimizerSerialization`/`ModelSerialization` : l'utiliser avec `model_path` échoue avec une erreur explicite plutôt que de sauvegarder un état incomplet — voir [Optimiseurs](optimizers.md) ;
- `CrossEntropyLoss` existe et son gradient est vérifié, mais aucun runtime CLI ne l'exploite (les runtimes `online_learning`/`training` sont structurellement des runtimes de régression scalaire, pas de classification multi-classe) — voir [Fonctions de perte](losses.md) ;
- `-O3` a été mesuré comme sûr (sortie bit-identique) mais volontairement pas adopté : le gain de vitesse (~6%, sur un run déjà sous la demi-seconde) ne justifie pas l'augmentation de taille des binaires (~30%), contraire à l'objectif d'un `bin/gloomy_infer` minimal ; le benchmark n'est pas parallélisé, `DenseLayer::seedWeightInitialization` reposant sur un générateur statique partagé qu'il faudrait d'abord découpler (même risque que l'arena allocator) ;
- les statistiques de benchmark dépendent de la machine ;
- les données synthétiques ne remplacent pas une évaluation sur données réelles (aucun jeu de données réel disponible dans cet environnement sans accès réseau) ; il n'existe pas non plus de baseline `float32` distincte d'int16/int8.

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
9. mettre à jour la documentation des [examples](examples.md) si c'est necessaire
10. mettre à jour les `README.md` si c'est necessaire
