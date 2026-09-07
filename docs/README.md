# Documentation du réseau

Ce projet est un prédicteur numérique C++ basé sur des couches denses. Il effectue une propagation avant avec des poids aléatoires et ne réalise ni entraînement ni rétropropagation. Une sortie est donc une estimation aléatoire tant qu'aucun mécanisme d'apprentissage n'est ajouté.

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
- [Couches et neurones](architecture.md)
- [Configurations et limites](configurations.md)
