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
- schedulers `EverySample`, `EveryN` et `OnHighError`.

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

Le format `GLOOMY_MODEL` sert à la reprise et inclut obligatoirement une section optimiseur; un artefact inference-only minimal n'est pas encore disponible.

### Benchmarks et métriques

- métriques MAE, RMSE et forgetting ;
- benchmark synthétique comparant stratégies mémoire, capacités, pertes, optimiseurs et précisions de mémoire sur trois seeds, transmis aussi aux générateurs aléatoires des mémoires ;
- baseline « dernière valeur », expériences A → B et A → B → A, et export CSV des résultats, du forgetting et du drift.

Les benchmarks sont indicatifs et ne représentent pas des données applicatives ni une mesure sur matériel embarqué. La portée et les limites sont décrites dans [benchmark](docs/benchmark.md).

## Comportement conservé du CLI historique

Le CLI de démonstration accepte une séquence de valeurs et construit un réseau neuf. Lors de prédictions successives, il reconstruit le réseau avec de nouveaux poids; ce chemin ne correspond pas au runtime configuré ni à une boucle autorégressive entraînée.
