# Régression linéaire : `y = 2x + 1`

Cet exemple montre le cycle complet : préparer les exemples, entraîner un réseau, sauvegarder son checkpoint et lancer des prédictions.

Toutes les commandes ci-dessous sont à exécuter depuis la racine du dépôt.

## 1. Compiler et préparer le dossier de sortie

```bash
make
mkdir -p artifacts
```

Le jeu de données est dans [`train.txt`](train.txt). Chaque ligne suit la forme `entrée | cible`.

## 2. Entraîner le modèle

La configuration est dans [`training.config`](training.config). Elle choisit un réseau linéaire sans couche cachée et 1 000 epochs.

```bash
./bin/gloomy -f docs/examples/regression-lineaire/training.config
```

Le checkpoint est écrit dans `artifacts/model-lineaire.gloomy`. Une nouvelle exécution avec le même chemin reprend ce checkpoint et poursuit l'entraînement. Pour repartir de zéro, supprimez ou changez ce fichier.

## 3. Prédire des valeurs

La configuration d'inférence est dans [`inference.config`](inference.config). Changez `input` pour essayer une autre entrée.

```bash
./bin/gloomy -f docs/examples/regression-lineaire/inference.config
```

Pour tester directement l'entrée `5` sans modifier le fichier :

```bash
./bin/gloomy -f docs/examples/regression-lineaire/inference.config --input 5
```

Pour les entrées `2` et `5`, les cibles attendues sont respectivement `5` et `11`. Avec la graine et les réglages fournis, les prédictions devraient être proches de ces valeurs. Le runtime n'effectue pas automatiquement d'évaluation sur un jeu de validation.

## Fichiers de l'exemple

- [`train.txt`](train.txt) : quatre exemples d'entraînement ;
- [`training.config`](training.config) : architecture, optimiseur, apprentissage et chemin du checkpoint ;
- [`inference.config`](inference.config) : chemin du checkpoint et entrée à prédire.
