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

`GloomyModelSerialization` fournit le premier format unifié `GLOOMY_MODEL` versionné :

```text
HEADER (magic, version, nombre de sections)
SECTION (identifiant, taille, contenu) : NETWORK
SECTION (identifiant, taille, contenu) : NORMALIZATION
SECTION (identifiant, taille, contenu) : OPTIMIZER
CHECKSUM GLOBAL
```

Chaque section est délimitée par son identifiant et sa taille. Les identifiants inconnus peuvent être ignorés pour permettre l'ajout de sections optionnelles dans une évolution compatible. Les trois sections ci-dessus sont requises en version 1. Les données de quantification et la mémoire d'apprentissage seront ajoutées dans les prochaines étapes. Le checksum couvre l'en-tête et toutes les sections.

```cpp
GloomyModelSerialization::save("model.gloomy", network, normalizer, optimizer);
GloomyModelSerialization::load("model.gloomy", network, normalizer, optimizer);
```

Le chargement valide toutes les sections et prépare le réseau ainsi que le normaliseur avant de restaurer l'optimiseur et de remplacer les objets cibles. Le type d'optimiseur doit correspondre à l'instance fournie, et les dimensions du normaliseur doivent correspondre aux entrées du réseau. Une normalisation sans observation ne peut pas être sauvegardée.

## Réseau neuronal

`NetworkSerialization` sauvegarde déjà une architecture Dense complète : activation, post-activation, dimensions, poids et biais.

```cpp
NetworkSerialization::save("model.gloomy", network);
NetworkSerialization::load("model.gloomy", restored_network);
```

Le chargement reconstruit et valide le réseau dans un objet temporaire avant de remplacer l'objet cible. Le format réseau séparé est versionné, protégé par un checksum FNV-1a et refuse les magic, versions, dimensions incohérentes, paramètres non finis, corruptions, données surnuméraires ou fichiers tronqués invalides. Pour sauvegarder ensemble le réseau, la normalisation et l'optimiseur, utiliser le format `GLOOMY_MODEL`.

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
