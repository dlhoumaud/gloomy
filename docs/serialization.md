# Sérialisation

## TrainingSample

`TrainingSampleSerialization` sauvegarde et restaure une collection de `TrainingSample` dans un fichier binaire versionné.

Le format actuel contient :

```text
GLOOMYSM
version
sample_count
input vector
 target vector
metadata
```

Les métadonnées sauvegardées sont `priority`, `error`, `novelty`, `rarity`, `recency`, `diversity`, `age` et `usage_count`.

```cpp
TrainingSampleSerialization::save(path, samples);
const std::vector<TrainingSample> restored =
    TrainingSampleSerialization::load(path);
```

Le lecteur vérifie le magic, la version, les tailles maximales et les lectures complètes. Les fichiers tronqués, incompatibles ou contenant des tailles excessives sont refusés.

## Limites et évolution

Ce premier format d'échantillons ne sauvegarde pas encore l'état de l'optimiseur, les paramètres de quantification ou les statistiques de normalisation dans le même fichier. Ces composants disposent toutefois maintenant de sérialiseurs séparés, y compris l'état de l'optimiseur (voir ci-dessous).

La prochaine évolution pourra encapsuler ces sections dans un format `GLOOMY_MODEL` versionné :

```text
HEADER
ARCHITECTURE
NORMALIZATION
QUANTIZATION
WEIGHTS
BIASES
OPTIMIZER_STATE
LEARNING_MEMORY
METADATA
CHECKSUM
```

## Réseau neuronal

`NetworkSerialization` sauvegarde déjà une architecture Dense complète : activation, post-activation, dimensions, poids et biais.

```cpp
NetworkSerialization::save("model.gloomy", network);
NetworkSerialization::load("model.gloomy", restored_network);
```

Le chargement reconstruit les couches avant de restaurer leurs paramètres. Le format est versionné, protégé par un checksum FNV-1a et refuse les magic, versions, dimensions, corruptions ou fichiers tronqués invalides. L'état de l'optimiseur, la mémoire d'apprentissage et la quantification restent à intégrer dans le format modèle complet.

## Normalisation

`NormalizationSerialization` sauvegarde les statistiques nécessaires à `StreamingNormalizer` : nombre d'observations, moyenne, variance, minimum et maximum. Le chargement restaure l'état Welford sans rejouer le dataset.

```cpp
NormalizationSerialization::save("normalization.bin", normalizer);
NormalizationSerialization::load("normalization.bin", restored_normalizer);
```

## Optimiseur

`OptimizerSerialization` sauvegarde et restaure l'état de n'importe quel `Optimizer` concret (SGD, Momentum, Adam) : type, hyperparamètres, vitesses de Momentum, moments d'Adam et compte d'updates. Le format est versionné et protégé par un checksum FNV-1a comme celui du réseau.

```cpp
OptimizerSerialization::save("optimizer.bin", optimizer);
std::unique_ptr<Optimizer> restored =
    OptimizerSerialization::load("optimizer.bin", network.layers());
```

`load()` valide que la forme de l'état sauvegardé (nombre de couches, dimensions d'entrée/sortie) correspond exactement aux couches du réseau fourni avant de restaurer le buffer, afin d'éviter de réutiliser un état incompatible avec l'architecture courante. Voir [Optimiseurs](optimizers.md) pour le détail.

## Mémoire d'apprentissage

`LearningMemorySerialization` sauvegarde et restaure une mémoire complète — FIFO, Reservoir, Prioritized, Novelty ou Hybrid — avec sa capacité, ses hyperparamètres, ses échantillons (métadonnées incluses), les partitions Hybrid et l'état RNG (`std::mt19937`) des stratégies qui en utilisent un. Format versionné, protégé par un checksum FNV-1a.

```cpp
LearningMemorySerialization::save("memory.bin", memory);
std::unique_ptr<LearningMemory> restored = LearningMemorySerialization::load("memory.bin");
```

Voir [Mémoire d'apprentissage](memory.md) pour le détail par stratégie et les limites (mémoires quantifiées non couvertes).

Format actuellement en **version 2** : `correction_exponent` (`beta`, voir [Mémoire d'apprentissage](memory.md), « Correction de biais d'échantillonnage ») a été ajouté à la section `PrioritizedMemory`. Un fichier version 1 (antérieur à ce changement) est refusé par `load()` plutôt que mal interprété.

Ces fichiers (réseau, normalisation, optimiseur, mémoire) restent séparés ; leur regroupement dans un format `GLOOMY_MODEL` unique reste à faire.
