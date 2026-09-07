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

Ce premier format d'échantillons ne sauvegarde pas encore l'état de l'optimiseur, les paramètres de quantification ou les statistiques de normalisation dans le même fichier. Ces composants disposent toutefois maintenant de sérialiseurs séparés.

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
