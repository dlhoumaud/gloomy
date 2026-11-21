# Journal des changements

Ce journal récapitule les fonctionnalités déjà présentes dans le dépôt. Aucune version ou date de livraison n'est indiquée faute de version publiée déclarée.

## Fonctionnalités livrées — version non publiée

### Réseau et apprentissage

- réseau Dense avec propagation avant, rétropropagation, activations `none`, `sigmoid`, `relu`, `leaky_relu`, `tanh` et softmax final ;
- pertes MSE, MAE et Huber ;
- optimiseurs SGD, Momentum et Adam, avec mesures de taille d'état pour Momentum et Adam ;
- entraînement par epochs et mini-batches, replay indexé et recalcul des priorités après mise à jour ;
- parcours des mini-batches du dataset sans créer un vecteur de batch temporaire ;
- accès interne par référence aux sorties de propagation avant pour éviter leurs copies pendant l'entraînement, tout en conservant l'API `forward()` par valeur ;
- réutilisation des buffers de gradient des couches pendant la rétropropagation d'entraînement, avec conservation de l'API `backward()` par valeur ;
- buffers d'inférence, de gradient de perte et d'échantillons replay réutilisables; softmax calculé en place et déquantification/quantification écrivable dans des buffers existants ;
- runtime d'entraînement qui normalise les entrées du dataset en place au lieu de conserver un second dataset normalisé ;
- calibrage des mémoires quantifiées qui calcule directement les bornes min/max sans copier tout le dataset dans des vecteurs temporaires ;
- lecture ligne par ligne du dataset et entraînement par mini-batches, avec passes supplémentaires de statistiques/calibration au lieu de garder le dataset complet en RAM ;
- estimation des allocations conservées pour les stratégies mémoire intégrées, utilisée par les lignes `memory_used_bytes` du benchmark ;
- stockage contigu des poids et gradients de poids, avec vue 2D pour conserver l'accès indexé et formats de sérialisation inchangés ;
- schedulers `EverySample`, `EveryN` et `OnHighError`.

### Prototype de modèle de langage

- tokenizer byte-level BPE déterministe et versionné avec validation UTF-8, NFC Unicode 16.0, retrait du BOM initial et normalisation des fins de ligne ;
- Transformer causal CPU avec embeddings, attention multi-tête causale, résidus, LayerNorm et MLP GELU ;
- entraînement du prochain token par cross-entropy, mini-batches, AdamW, clipping du gradient, suivi cross-entropy/perplexité ;
- masques de perte sur les prompts QA pour superviser la génération des réponses sans pénaliser la reproduction de l'instruction ;
- checkpoint texte versionné regroupant tokenizer, configuration, poids, compteur d'updates, moments AdamW et état RNG de mélange, avec reprise reproductible aux frontières d'epoch ;
- CLI `lm train`, `lm evaluate` et `lm generate`, incluant température, top-k, top-p, seed et exact match QA ;
- outil de préparation locale des corpus texte avec normalisation NFC, déduplication, splits déterministes et manifeste de provenance/empreintes ;
- exemple de corpus fictif train/validation/test et documentation de l'évaluation, du profil local et des limites.
- corpus applicatif local d'assistant documentaire Gloomy, avec splits fixes par fichier source, QA train/validation/test conditionnée par contexte, test équilibré oui/non et baseline majoritaire explicite, manifeste d'empreintes et provenance, benchmark multi-seed et test de reprise bit-à-bit sur les documents.

Ce prototype de petite échelle n'a pas démontré la compréhension générale. Des corpus applicatifs autorisés, une évaluation plus robuste et un profil sur cible restent nécessaires.

### Mémoire et normalisation

- mémoires FIFO, Reservoir Sampling, Prioritized Replay, Novelty et Hybrid ;
- métadonnées d'importance, âge et nombre d'utilisations, et mises à jour ciblées par indice ;
- mémoires FIFO quantifiées int16 et int8, codecs de vecteurs et de `TrainingSample` ;
- normalisation streaming des entrées avec statistiques Welford.

Les vecteurs et métadonnées des autres stratégies mémoire, ainsi que les poids et états d'optimiseur, restent en float64. Les capacités des mémoires ordinaires sont exprimées en nombre d'échantillons.

### Runtime et persistance

- runtime configuré par fichier clé-valeur avec modes `training`, `online_learning` et `inference` ;
- dataset texte en paires entrée/cible, options de surcharge limitées et sauvegarde des modèles ;
- sérialisation versionnée des échantillons, réseau, normalisation et optimiseur ;
- format `GLOOMY_MODEL` avec sections réseau, normalisation, optimiseur et mémoire optionnelle ;
- persistance des stratégies mémoire intégrées, des états RNG des stratégies aléatoires prises en charge et des paramètres des mémoires quantifiées.
- en inférence, le chargeur vérifie le checksum global mais saute la désérialisation de la section mémoire lorsque l'appelant ne demande pas à la restaurer.
- estimation des allocations du réseau, de l'optimiseur et du moteur d'apprentissage, ainsi qu'un budget optionnel `memory_budget_bytes` pour ajuster la capacité du replay ;
- rapport runtime du pic RSS fourni par le système d'exploitation ;
- artefact `GLOOMY_INFER` versionné avec réseau et normalisation uniquement, détecté automatiquement par le runtime et exportable en complément du checkpoint d'entraînement.

Le format `GLOOMY_MODEL` sert à la reprise et inclut obligatoirement une section optimiseur; `GLOOMY_INFER` est réservé au déploiement d'inférence.

### Benchmarks et métriques

- métriques MAE, RMSE et forgetting ;
- benchmark synthétique comparant stratégies mémoire, capacités, pertes, optimiseurs et précisions de mémoire sur trois seeds, transmis aussi aux générateurs aléatoires des mémoires ;
- baseline « dernière valeur », expériences A → B et A → B → A, et export CSV des résultats, du forgetting et du drift.
- export du pic RSS dans le profil d'opérations `benchmark_profile.csv` ;
- comparaison de taille et de temps de chargement des artefacts dans `benchmark_artifacts.csv` ;
- comparaison de six stratégies mémoire sur trois régimes et transitions progressives dans `benchmark_progressive_drift.csv` ;
- profil local étendu à un réseau multi-sortie `3 → 16 → 16 → 2`, avec RSS, allocations et latence sur la machine de référence ;
- comparaison avant/après des optimisations sur mêmes données, trois seeds, architecture, 50 updates, qualité de validation et mesures mémoire via `make benchmark-comparison`.

Les benchmarks sont indicatifs et ne représentent pas des données applicatives ni une mesure sur matériel embarqué. La portée et les limites sont décrites dans [benchmark](docs/benchmark.md).

### Exemples documentés

- ajout d'exemples exécutables pour les régressions scalaire et vectorielle, le replay FIFO int8 et l'apprentissage en ligne multi-régime avec mémoire Hybrid.

## Comportement conservé du CLI historique

Le CLI de démonstration accepte une séquence de valeurs et construit un réseau neuf. Lors de prédictions successives, il reconstruit le réseau avec de nouveaux poids; ce chemin ne correspond pas au runtime configuré ni à une boucle autorégressive entraînée.
