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

Le chargement reconstruit et valide le réseau dans un objet temporaire avant de remplacer l'objet cible. Le format est versionné, protégé par un checksum FNV-1a et refuse les magic, versions, dimensions incohérentes, paramètres non finis, corruptions, données surnuméraires ou fichiers tronqués invalides. L'état de l'optimiseur, la mémoire d'apprentissage et la quantification restent à intégrer dans le format modèle complet.

## Normalisation

`NormalizationSerialization` sauvegarde les statistiques nécessaires à `StreamingNormalizer` : nombre d'observations, moyenne, variance, minimum et maximum. Le chargement restaure l'état Welford sans rejouer le dataset.

```cpp
NormalizationSerialization::save("normalization.bin", normalizer);
NormalizationSerialization::load("normalization.bin", restored_normalizer);
```

Le fichier est entièrement lu dans des buffers temporaires et toutes les statistiques sont validées avant leur application. Les données tronquées ou surnuméraires et les statistiques invalides sont refusées sans changer l'objet cible.

`TrainingSampleSerialization::load` construit et retourne une nouvelle collection ; un fichier incomplet ou contenant des données surnuméraires est refusé avant que la collection soit renvoyée.

## Optimiseur

`OptimizerSerialization` sauvegarde le type d'optimiseur, ses hyperparamètres et ses buffers (vitesses Momentum, moments et compteur d'updates Adam). Le checkpoint possède son propre magic, sa version et son checksum. Il doit être chargé dans une instance du même type, après avoir restauré le réseau ; les formes de couches sont vérifiées avant la restauration.

```cpp
OptimizerSerialization::save("optimizer.bin", optimizer);
OptimizerSerialization::load("optimizer.bin", optimizer, network.layers());
```
