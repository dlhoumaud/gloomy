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
- statistiques de normalisation ;
- état de l'optimiseur (SGD, Momentum, Adam) via `OptimizerSerialization` ;
- mémoire d'apprentissage native (FIFO, Reservoir, Prioritized, Novelty, Hybrid) via `LearningMemorySerialization`.

Un format unifié `GLOOMY_MODEL` est désormais également disponible via `ModelSerialization` (`src/headers/ModelSerialization.h`, `src/ModelSerialization.cpp`) : il regroupe dans un fichier versionné et protégé par checksum le réseau, la normalisation, l'optimiseur et la mémoire d'apprentissage. Le fichier réseau, le fichier optimiseur, le fichier mémoire et le fichier unifié sont tous vérifiés par un checksum FNV-1a. Les tests vérifient le round-trip et le rejet d'une corruption. Pour l'optimiseur, `load()` reconstruit le type concret à partir du fichier et refuse de restaurer un état dont la forme (nombre de couches, dimensions par couche) ne correspond pas exactement au réseau fourni. Pour la mémoire, `load()` restaure aussi l'état complet du générateur `std::mt19937` (Reservoir, Prioritized, Hybrid) et les partitions (Hybrid), afin que le replay reste reproductible après un redémarrage. Les mémoires quantifiées (`QuantizedFIFOMemory`, `QuantizedInt8FIFOMemory`) ne sont pas encore couvertes.

### Métriques et benchmark

Métriques disponibles :

- MAE ;
- RMSE ;
- forgetting.

Le runner benchmark compare actuellement :

- baseline naïve `baseline_last_value` (dernière valeur connue, sans apprentissage) ;
- dataset complet (référence 100 epochs) ;
- FIFO, Reservoir, Prioritized, Novelty, Hybrid, chacune à 4 capacités (`32`, `64`, `128`, `256`) et 3 seeds ;
- FIFO int16 et FIFO int8 (capacité `16`, seed unique) ;
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

   Le runtime online l'utilise désormais via `model_path` pour sauvegarder l'état entraîné au terme d'une exécution. Il reste à étendre ce format aux composants encore hors périmètre, en particulier les mémoires quantifiées et les paramètres de quantification.

2. **Persistance de l'état des optimiseurs — fait**

   `OptimizerSerialization::save`/`load` persiste :

   - type d'optimiseur ;
   - learning rate ;
   - hyperparamètres (`momentum`, `beta1`, `beta2`, `epsilon`) ;
   - compte d'updates Adam ;
   - vitesses Momentum ;
   - premiers et seconds moments Adam.

   L'état est vérifié contre la forme des couches (nombre de couches, dimensions d'entrée/sortie) fournies à `load()` pour éviter de restaurer un buffer incompatible ; un fichier tronqué, corrompu ou de version différente est également rejeté. Le format `GLOOMY_MODEL` unifié couvre désormais ce composant, et le CLI online l'utilise via `model_path` pour sauvegarder l'état entraîné au terme d'une exécution.

3. **Persistance des mémoires — fait pour la représentation float64**

   `LearningMemorySerialization::save`/`load` persiste, pour FIFO, Reservoir, Prioritized, Novelty et Hybrid :

   - stratégie utilisée et capacité ;
   - état RNG complet (pas seulement la seed) et compteur d'observations vues pour Reservoir, Prioritized et Hybrid ;
   - échantillons avec toutes leurs métadonnées (priorité, erreur, novelty, rarity, recency, diversity, âge, usage_count) ;
   - partitions Hybrid ;
   - paramètres propres à chaque stratégie (`alpha` pour Prioritized, `novelty_threshold` pour Novelty et Hybrid, ratios pour Hybrid).

   Reste à faire : les mémoires quantifiées (`QuantizedFIFOMemory`, `QuantizedInt8FIFOMemory`) avec leur représentation int16/int8 et leurs paramètres `scale`/`zero_point`.

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

   **État** : `INFERENCE_RUNTIME` (comportement historique, inchangé) et `ONLINE_LEARNING_RUNTIME` sont faits. `runOnlineLearning()` (`src/headers/OnlineLearningRuntime.h`, `src/OnlineLearningRuntime.cpp`) implémente exactement la boucle ci-dessus en réutilisant les composants déjà testés séparément : `NeuralNetwork` scalaire (entrée/sortie de dimension 1), `StreamingNormalizer`, une perte/un optimiseur/une mémoire construits depuis `GloomyConfig` (`loss`, `optimizer`, `memory_strategy` et leurs hyperparamètres), un `TrainingScheduler` (`EverySampleScheduler` ou `EveryNScheduler(train_every)`), et `LearningEngine::learn()` pour le replay et la mise à jour. Chaque valeur consécutive de la séquence d'entrée devient une observation (`x[i]`) et sa cible (`x[i+1]`).

   Le CLI sélectionne ce runtime via la clé `runtime=online_learning` d'un fichier `-f`/`--config` (voir section « Configuration fichier » ci-dessous) ; `main.cpp` a été réorganisé pour faire circuler un unique `GloomyConfig` du parsing jusqu'au dispatch (`runInference`/`runOnline`), au lieu de cinq variables locales dispersées. Un runtime inconnu est rejeté avec un message explicite, de même qu'une perte, un optimiseur ou une stratégie de mémoire inconnus (tous les cas testés).

   `TRAINING_RUNTIME` (entraînement par epochs sur un jeu de données complet, via `LearningEngine::train()`) est maintenant exposé dans le CLI via `runtime=training`. Le runtime construit un réseau scalaire, normalise la séquence, entraîne sur le dataset complet puis sauvegarde l'état complet si `model_path` est fourni.

   Limites connues de cette première version : le réseau du runtime online est fixé à une entrée/sortie scalaire (pas de fenêtre configurable) ; le CLI ne charge pas encore un modèle sauvegardé au démarrage, et la sortie reste un flux `stdout` ligne par ligne, pas encore un format structuré. La persistance du modèle entraîné est désormais branchée : si `model_path` est renseigné, le runtime online sauvegarde le réseau, la normalisation, l'optimiseur et la mémoire via `ModelSerialization` au terme de son exécution.

### Configuration fichier

Aucune nouvelle option CLI n'a été ajoutée pour piloter l'architecture d'inférence (toujours `-c`/`-l`/`-n`/`-a`/`-A`) ; en revanche `-f`/`--config` sélectionne maintenant le runtime et ses hyperparamètres (voir point 6 ci-dessus et [Configurations et limites](configurations.md)).

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

**État** : les défauts sont maintenant centralisés dans `GloomyConfig` (`src/headers/GloomyConfig.h`, `src/GloomyConfig.cpp`). La structure reprend, champ par champ, le défaut déjà utilisé par chaque composant existant quand il en a un (`HuberLoss`, `MomentumOptimizer`, `AdamOptimizer`, `PrioritizedMemory`, `HybridMemoryRatios`, seed partagé de `std::mt19937`) et établit un défaut central documenté pour les champs qui n'en avaient pas encore (`learning_rate`, `memory_capacity`, `batch_size`, chemins de persistance). Le CLI (`src/main.cpp`) lit désormais ses cinq défauts actuels (`predictions`, `hidden_layers`, `neurons`, `activation`, `post_activation`) depuis `GloomyConfig::defaults()` au lieu de littéraux dupliqués ; le comportement du CLI est inchangé (vérifié manuellement). Un test caractérise chaque valeur pour empêcher une dérive silencieuse. Le parseur `-f/--config` couvre désormais tous les champs de `GloomyConfig`, et le runtime online consomme explicitement `model_path` pour sauvegarder le modèle entraîné au terme de l'exécution ; les chemins `optimizer_path`, `memory_path` et `metrics_path` restent encore à utiliser proprement dans des étapes ultérieures.

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
- ~~validation systématique des valeurs non finies dans tous les composants~~ partiellement fait : `LossFunction::compute`/`gradient` et `DenseLayer::forward`/`backward` rejettent maintenant `NaN`/infini en entrée avec un message clair (`testNonFiniteValuesRejected`), comme le faisait déjà `StreamingNormalizer`. Restent à couvrir : les gradients accumulés par lot dans `Optimizer::update`, et les échantillons stockés dans les mémoires d'apprentissage (`TrainingSample::input`/`target` ne sont pas vérifiés à l'ajout, sauf dimension pour `NoveltyMemory`).

### Priorité moyenne : mémoire et continual learning

- mise à jour de toutes les composantes du score d'importance ;
- ~~vraie récence fondée sur l'âge plutôt que l'alternance actuelle de Hybrid~~ fait : `choosePartition()` n'alterne plus par `seen_samples % 2` — toute observation générique (ni erreur, ni nouveauté) rejoint directement `recent`. Lorsque `recent` est pleine, elle évince son membre le plus ancien **par `TrainingSample::age` réel** (pas par position dans le vecteur, qui ne reflétait plus l'âge après un premier remplacement en place — un vrai bug corrigé au passage) et le **promeut** dans `historical` au lieu de le perdre ; `error`/`novelty` bénéficient de la même correction d'éviction par âge réel. Supprimé au passage : `removeOldestFromPartition`, du code mort jamais appelé. Voir [Mémoire d'apprentissage](memory.md), « Vraie récence fondée sur l'âge », et `testHybridMemoryTrueRecency` ;
- ~~mise à jour des priorités avec correction de biais d'échantillonnage~~ fait : `PrioritizedMemory` calcule un poids d'importance-sampling `(N·P(i))^(-beta)` (normalisé au maximum du batch) pour chaque échantillon tiré par `sampleIndexed()` ; `beta` est un nouveau paramètre de construction (défaut `0.4`, `0` désactive la correction). `LearningEngine::trainFromMemory` applique ce poids à la contribution de chaque échantillon au gradient via `MemoryEntry::importance_weight` (`1.0` par défaut, donc neutre pour les autres stratégies) et la nouvelle méthode privée `trainWeightedBatch`, dont `trainBatch` est maintenant un cas particulier (tous les poids à `1.0`). Persisté par `LearningMemorySerialization` (format bumpé en version 2). Voir [Mémoire d'apprentissage](memory.md), « Correction de biais d'échantillonnage », et les tests `testPrioritizedMemoryBiasCorrection`/`testLearningEngineTrainBatchWeighting`. Reste ouvert : l'annealing de `beta` au fil de l'entraînement n'est pas implémenté (valeur fixe) ;
- exploration contrôlée des échantillons de faible priorité ;
- prototypes / coreset ;
- mémoire par régimes ;
- détection légère de concept drift ;
- règles adaptatives explicites avant tout mécanisme appris ;
- expériences A -> B -> A plus nombreuses et reproductibles.

### Priorité moyenne : benchmark scientifique

**État : fait dans son ensemble.** `BenchmarkRunner` couvre désormais :

- ~~capacités `32`, `64`, `128`, `256`~~ fait pour les 5 stratégies float64 (FIFO, Reservoir, Prioritized, Novelty, Hybrid) ;
- ~~Novelty et Hybrid~~ fait, dans le même changement que les capacités ;
- float64, int16 et int8 sur les mêmes données — partiel : int16/int8 comparés au même dataset et désormais aux 3 pertes, mais toujours à une seule capacité (`16`) et une seule seed, pas sur le même balayage que float64 ;
- ~~pertes MSE, MAE et Huber~~ fait : `makeLoss()` construit la perte demandée ; le balayage de capacités, le dataset complet et les scénarios quantifiés tournent désormais chacun sur les 3 pertes (nouvelle colonne `loss_function`). Le balayage de capacités seul est donc 4 capacités × 5 stratégies × 3 optimiseurs × 3 pertes × 3 seeds = 540 scénarios (contre 60 avant) ; l'ensemble du runner reste sous la seconde ;
- ~~plusieurs seeds~~ fait pour le balayage de capacités (3 seeds : `1234`, `2345`, `3456`, pilotant à la fois `DenseLayer::seedWeightInitialization` et le générateur de la mémoire). Reste non fait : dataset complet et scénarios quantifiés, toujours à seed unique ;
- ~~moyenne et écart-type~~ fait : `mae_mean`/`mae_stddev` (écart-type population, diviseur N=3), calculées sur le MAE des 3 seeds d'un même scénario (capacité × stratégie × optimiseur × perte) et dupliquées sur chaque ligne du groupe ;
- ~~intervalles de confiance~~ fait : nouvelle colonne `mae_ci95_margin` (demi-largeur de l'IC à 95%, loi de Student `df=2`, écart-type d'**échantillon** — diviseur N-1, différent de `mae_stddev` qui reste population). La valeur critique de Student est codée en dur pour `N=3` seeds spécifiquement (`t_critical_95_df2` dans `BenchmarkRunner.cpp`) ; elle vaut `0` si le nombre de seeds change un jour sans mise à jour de cette constante, plutôt que d'utiliser silencieusement une valeur fausse. À lire comme un ordre de grandeur (échantillon de taille 3), pas une garantie statistique forte ;
- ~~baseline dernière valeur connue~~ fait : ligne `baseline_last_value` (calculée avant toute chose, sans RNG donc sans effet sur les scénarios suivants), évaluée avec les 3 pertes. Sert de plancher de comparaison — voir [Benchmark](benchmark.md) ;
- ~~ratio `performance_memory_limited / performance_full_dataset`~~ fait : colonne `mae_ratio_to_full_dataset`, calculée par rapport au MAE `full_dataset` du même optimiseur **et de la même perte**. **Limite importante inchangée** : `full_dataset` entraîne 100 epochs en batch (MAE proche de zéro) alors que les scénarios bornés font un seul passage online ; le ratio mélange donc l'effet du nombre de passages et celui de la capacité mémoire (valeurs parfois de l'ordre du million). Isoler l'effet de la seule capacité, à nombre de passages égal, reste à faire ;
- ~~coût CPU et nombre d'opérations approximatif~~ fait : colonne `approximate_macs` (MAC du forward × 3 comme approximation forward+backward, grossier par construction — ignore le coût de l'optimiseur et des activations) ;
- ~~samples/sec et updates/sec~~ fait : colonnes `samples_per_second`/`updates_per_second`, dérivées de `training_time_ms` ;
- ~~fichiers CSV séparés par expérience~~ fait : `benchmark_baseline.csv`, `benchmark_full_dataset.csv`, `benchmark_memory_capacity.csv`, `benchmark_quantization.csv`, `benchmark_forgetting.csv`, en plus du `benchmark_results.csv` combiné conservé pour compatibilité (`docs/examples.md` l'utilise).

Reste non fait : dataset complet et scénarios quantifiés à seed unique (pas de moyenne/écart-type/IC pour ces sections) ; balayage de capacités pour int16/int8 ; baseline `float32` ; jeux de données réels.

Au passage : le format CSV (colonnes `loss_function`, `mae_ci95_margin`, `approximate_macs`, `samples_per_second`, `updates_per_second`) reste rétrocompatible en ajout de colonnes, sauf `loss_function` insérée en 4e position (avant les colonnes numériques) — la vérification par motif dans [Benchmark](benchmark.md) démarre donc à la colonne 5, pas 4. `testBenchmarkCsv` couvre tous les nouveaux champs. Ce changement a de nouveau décalé les valeurs de l'expérience de forgetting utilisées dans [docs/examples.md](examples.md) (retombées, par coïncidence, sur les toutes premières valeurs observées avant tout ce travail sur le benchmark) — mis à jour et revérifié stable sur plusieurs exécutions.

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
2. ~~Ajouter la persistance de l'état Optimizer.~~ Fait (`OptimizerSerialization`, voir section 2 et 3).
3. ~~Ajouter la persistance des mémoires~~ Fait pour la représentation float64 (`LearningMemorySerialization`, voir section 2 et 3). Reste : les paramètres de quantification (int16/int8).
4. ~~Centraliser les défauts dans une configuration C++.~~ Fait (`GloomyConfig`, voir section 2 et 3).
5. ~~Ajouter `-f/--config` au CLI avec priorité CLI > fichier > défauts.~~ Fait (`GloomyConfigFile`, voir section 2 et 3).
6. ~~Exposer un mode online fonctionnel dans le CLI.~~ Fait pour `ONLINE_LEARNING_RUNTIME` (`OnlineLearningRuntime`, voir section 2 et 3).
7. ~~Brancher la persistance du modèle entraîné dans le runtime online.~~ Fait via `ModelSerialization` et `model_path` dans le CLI.
8. ~~Ajouter `TRAINING_RUNTIME` dans le CLI.~~ Fait via `runtime=training`, avec `epochs` et sauvegarde de `model_path`.
9. ~~Étendre les benchmarks aux capacités et stratégies restantes.~~ Fait dans son ensemble : capacités `32`/`64`/`128`/`256`, stratégies Novelty/Hybrid, 3 pertes (MSE/MAE/Huber), 3 seeds avec moyenne/écart-type/IC95 du MAE, ratio au dataset complet, baseline naïve `baseline_last_value`, coûts CPU/débit approximatifs, et fichiers CSV séparés par expérience (voir section 3, « Priorité moyenne : benchmark scientifique »). Restent : seeds multiples et balayage de capacités pour le dataset complet et les scénarios quantifiés, baseline `float32`, jeux de données réels.
10. Ajouter les tests de concept drift et catastrophic forgetting.
11. Optimiser les allocations et la représentation mémoire.
12. Préparer le runtime embarqué et la quantification des poids.

## 5. Limites connues à ne pas oublier

- le CLI recrée actuellement le réseau entre prédictions autorégressives, avec de nouveaux poids aléatoires (`INFERENCE_RUNTIME` uniquement ; sans effet sur `ONLINE_LEARNING_RUNTIME`, qui garde un seul réseau du début à la fin) ;
- le CLI ne charge pas encore de modèle sauvegardé ; le runtime online sauvegarde désormais le réseau/l'optimiseur/la mémoire entraînés à la fin de son exécution lorsqu'un `model_path` est renseigné ;
- le CLI lance maintenant un entraînement par epochs sur un jeu de données complet via `runtime=training` ;
- le réseau du runtime online est fixé à une entrée/sortie scalaire, sans fenêtre configurable ;
- `softmax` sur la sortie actuelle à un neurone vaut toujours `1` ;
- les couches utilisent encore des vecteurs imbriqués et des allocations dynamiques ;
- les mémoires natives stockent encore des `double` ;
- les poids restent en float64 ;
- les mémoires quantifiées (int16/int8) ne sont pas encore persistées ;
- un format `GLOOMY_MODEL` unifié existe maintenant via `ModelSerialization`, en mode online il est désormais branché dans le CLI via `model_path`, et il reste à compléter les sections encore non couvertes (mémoires quantifiées, paramètres de quantification, éventuels métadonnées supplémentaires) ;
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
9. mettre à jour la documentation des [examples](examples.md) si c'est necessaire
10. mettre à jour les `README.md` si c'est necessaire
