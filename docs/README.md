# Documentation Gloomy

Gloomy est un moteur expérimental Dense en C++17. Le runtime configuré est le chemin d'entraînement batch/online et d'inférence sur modèle sauvegardé. Le CLI historique reste une démonstration : il reconstruit le réseau entre prédictions et ne représente pas un modèle autorégressif entraîné.

Le flux configuré est : lecture des exemples `entrée | cible`, normalisation des entrées, entraînement ou replay mémoire, sauvegarde `GLOOMY_MODEL`; en inférence, chargement du modèle, normalisation de l'entrée et émission des sorties. Le dataset configuré est chargé en mémoire; le traitement en flux et les budgets stricts de RAM restent à faire.

## Entrées rapides

```bash
make
./bin/gloomy -f gloomy.config
```

Les modes sont `training`, `online_learning` et `inference`. Voir [configurations et limites](configurations.md) pour les clés prises en charge, le format dataset et les options de surcharge.

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

`make test` exécute les tests existants; `make benchmark` lance les scénarios synthétiques reproductibles décrits dans [benchmark](benchmark.md). La latence et le débit dépendent de la machine. Le benchmark ne mesure ni pic RSS, ni énergie, ni performances sur matériel embarqué.
