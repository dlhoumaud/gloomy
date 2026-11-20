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

## Formats autonomes et format de modèle

`TrainingSampleSerialization` est un format autonome d'échantillons; il n'est pas le format de reprise complet. Les composants réseau, normalisation et optimiseur disposent aussi de sérialiseurs autonomes.

`GloomyModelSerialization` fournit le premier format unifié `GLOOMY_MODEL` versionné :

```text
HEADER (magic, version, nombre de sections)
SECTION (identifiant, taille, contenu) : NETWORK
SECTION (identifiant, taille, contenu) : NORMALIZATION
SECTION (identifiant, taille, contenu) : OPTIMIZER
SECTION optionnelle (identifiant, taille, contenu) : LEARNING_MEMORY
CHECKSUM GLOBAL
```

Chaque section est délimitée par son identifiant et sa taille. Les identifiants inconnus peuvent être ignorés pour permettre l'ajout de sections optionnelles. Les trois premières sections sont requises en version 1. La section mémoire est incluse lorsqu'une mémoire est fournie. Le checksum couvre l'en-tête et toutes les sections.

```cpp
GloomyModelSerialization::save("model.gloomy", network, normalizer, optimizer);
GloomyModelSerialization::load("model.gloomy", network, normalizer, optimizer);
```

Pour inclure la mémoire, fournir un pointeur vers celle-ci à `save`, puis un pointeur vers un `std::unique_ptr<LearningMemory>` à `load`. Le chargement recrée la stratégie et ses paramètres.

```cpp
GloomyModelSerialization::save("model.gloomy", network, normalizer, optimizer, &memory);
std::unique_ptr<LearningMemory> restored_memory;
GloomyModelSerialization::load("model.gloomy", network, normalizer, optimizer, &restored_memory);
```

FIFO, Reservoir, Prioritized, Novelty, Hybrid, FIFO int16 et FIFO int8 sont pris en charge. Reservoir, Prioritized et Hybrid sauvegardent l'état de leur générateur pseudo-aléatoire. Les mémoires quantifiées préservent leurs paramètres de calibration et les valeurs entières encodées. Les implémentations personnalisées de `LearningMemory` ne sont pas encore sérialisables.

Le chargement vérifie le checksum global, la version, les limites et l'unicité des sections, le type d'optimiseur et la cohérence des dimensions. Si l'appelant ne fournit pas de sortie `LearningMemory`, le chargeur saute la désérialisation de cette section après validation du checksum et de ses limites; c'est le chemin utilisé en inférence. Sinon, il prépare aussi la mémoire avant de remplacer les objets cibles, puis restaure l'état d'optimiseur compatible. Une normalisation sans observation ne peut pas être sauvegardée.

`GLOOMY_MODEL` est conçu comme checkpoint de reprise, pas comme artefact d'inférence minimal : la section `OPTIMIZER` est requise même en mode inference, et le lecteur reçoit une instance du type d'optimiseur correspondant. Une mémoire peut être omise à la sauvegarde; lorsqu'elle est présente, elle peut être chargée pour une reprise d'entraînement.

### Artefact dédié à l'inférence

`GLOOMY_INFER` est un format distinct contenant uniquement les sections `NETWORK` et `NORMALIZATION`, avec version, tailles bornées et checksum. Il exclut l'optimiseur, la mémoire de replay et leur état aléatoire. Utiliser `saveInference` et `loadInference` lorsque le modèle doit uniquement produire des prédictions :

```cpp
GloomyModelSerialization::saveInference("model.infer", network, normalizer);
GloomyModelSerialization::loadInference("model.infer", network, normalizer);
```

Le runtime détecte automatiquement le format à partir du magic. Un artefact d'inférence ne peut pas servir à reprendre l'entraînement. Après un entraînement, `inference_model_path=deployment.infer` ou l'option `--export-inference deployment.infer` génère cet artefact en plus du checkpoint `model_path`. Le format est versionné indépendamment du checkpoint de reprise.

## Réseau neuronal

`NetworkSerialization` sauvegarde déjà une architecture Dense complète : activation, post-activation, dimensions, poids et biais.

```cpp
NetworkSerialization::save("model.gloomy", network);
NetworkSerialization::load("model.gloomy", restored_network);
```

Le chargement reconstruit et valide le réseau dans un objet temporaire avant de remplacer l'objet cible. Le format réseau séparé est versionné, protégé par un checksum FNV-1a et refuse les magic, versions, dimensions incohérentes, paramètres non finis, corruptions, données surnuméraires ou fichiers tronqués. Pour sauvegarder ensemble le réseau, la normalisation et l'optimiseur, utiliser le format `GLOOMY_MODEL`.

## Normalisation

`NormalizationSerialization` sauvegarde les statistiques nécessaires à `StreamingNormalizer` : nombre d'observations, moyenne, variance, minimum et maximum. Le chargement restaure l'état Welford sans rejouer le dataset.

```cpp
NormalizationSerialization::save("normalization.bin", normalizer);
NormalizationSerialization::load("normalization.bin", restored_normalizer);
```

Le fichier est entièrement lu dans des buffers temporaires et toutes les statistiques sont validées avant leur application. Les données tronquées ou surnuméraires et les statistiques invalides sont refusées sans changer l'objet cible.

`TrainingSampleSerialization::load` construit et retourne une nouvelle collection ; un fichier incomplet ou contenant des données surnuméraires est refusé avant que la collection soit renvoyée.

## Optimiseur

`OptimizerSerialization` sauvegarde le type d'optimiseur, ses hyperparamètres et ses buffers (vitesses Momentum, moments et compteur d'updates Adam). Le checkpoint autonome possède son propre magic, sa version et son checksum. Il doit être chargé dans une instance du même type, après avoir restauré le réseau; les formes de couches sont vérifiées avant la restauration. Le format `GLOOMY_MODEL` utilise ce contrat d'état dans sa section `OPTIMIZER`.

```cpp
OptimizerSerialization::save("optimizer.bin", optimizer);
OptimizerSerialization::load("optimizer.bin", optimizer, network.layers());
```
