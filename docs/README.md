# Documentation du réseau

Ce projet est un moteur de réseau neuronal dense en C++. Il supporte la propagation avant, la rétropropagation, la perte MSE, l'optimiseur SGD, l'entraînement par epochs et mini-batches, ainsi qu'une première mémoire d'apprentissage FIFO bornée.

## Utilisation

```bash
make
./bin/gloomy "10.5 11.0 12.3" -c 1 -l 2 -n 8 -a tanh
```

- `-c C` : nombre de prédictions autorisées, par défaut `1`.
- `-l L` : nombre de couches cachées, par défaut `2`. `-l 0` utilise une seule couche entrée-sortie.
- `-n N` : nombre de neurones dans chaque couche cachée, par défaut `2`.
- `-a A` : activation utilisée par chaque couche.
- `-A softmax` : normalisation de la dernière couche uniquement.

Une prédiction est ajoutée à la séquence avant la suivante. Le réseau est alors reconstruit, avec de nouveaux poids aléatoires : les prédictions successives ne constituent donc pas une vraie boucle autorégressive entraînée.

## Guides

- [Fonctions d'activation et softmax](activations.md)
- [Fonctions de perte](losses.md)
- [Couches et neurones](architecture.md)
- [Mémoire d'apprentissage](memory.md)
- [Quantification](quantization.md)
- [Sérialisation](serialization.md)
- [État du projet et feuille de route](roadmap.md)
- [Métriques](metrics.md)
- [Optimiseurs](optimizers.md)
- [Benchmark](benchmark.md)
- [Configurations et limites](configurations.md)
