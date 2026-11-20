# Documentation Gloomy

Gloomy est un moteur expérimental Dense en C++17. Le runtime configuré est le chemin d'entraînement batch/online et d'inférence sur modèle sauvegardé. Le CLI historique reste une démonstration : il reconstruit le réseau entre prédictions et ne représente pas un modèle autorégressif entraîné.

Le runtime configuré lit les exemples `entrée | cible` par passes successives : validation des dimensions, calcul des statistiques, puis apprentissage par mini-batches ou replay en ligne. Il ne conserve pas le dataset entier en mémoire; l'usage dépend du mini-batch, du replay configuré, du modèle et d'une ligne en cours de lecture. Les modes entraînement relisent le fichier à chaque passe/epoch, et le calibrage int16/int8 ajoute une passe de bornes. La sauvegarde utilise `GLOOMY_MODEL`.

## Entrées rapides

```bash
make
./bin/gloomy -f gloomy.config
```

Les modes sont `training`, `online_learning` et `inference`. Voir [configurations et limites](configurations.md) pour les clés prises en charge, le format dataset et les options de surcharge.

Des exemples exécutables avec leurs datasets, configurations et commandes se trouvent dans [examples](examples/README.md). L'exemple de régression linéaire entraîne `y = 2x + 1`, sauvegarde le checkpoint, puis lance des prédictions.

Le CLI historique `./bin/gloomy "10.5 11.0 12.3" -c 1 -l 2 -n 8 -a tanh` est décrit dans [architecture](architecture.md); ses limites sont distinctes du runtime configuré.

## Guides

- [Fonctions d'activation et softmax](activations.md)
- [Fonctions de perte](losses.md)
- [Architecture et pipeline](architecture.md)
- [Mémoire d'apprentissage](memory.md)
- [Quantification](quantization.md)
- [Sérialisation](serialization.md)
- [Journal des changements](../CHANGELOG.md)
- [Travaux à venir](roadmap.md)
- [Métriques](metrics.md)
- [Optimiseurs](optimizers.md)
- [Benchmark](benchmark.md)
- [Configurations et limites](configurations.md)

## Mesures et portée

`make test` exécute les tests existants; `make benchmark` lance les scénarios synthétiques reproductibles décrits dans [benchmark](benchmark.md), dont le profil inclut le pic RSS du processus. La latence et le débit dépendent de la machine. Le benchmark ne mesure ni énergie, ni performances sur matériel embarqué.
