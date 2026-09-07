# Réseau de Neurones en C++ - Gloomy

## Description

Ce projet est une implémentation simple d'un réseau de neurones profond (DNN) en C++. Le réseau est capable de faire des prédictions en utilisant différentes fonctions d'activation telles que **sigmoid**, **relu**, **tanh**, et leurs dérivées. De plus, il prend en charge la fonction de post-activation **softmax**. Le réseau utilise une architecture basique de couches entièrement connectées (couches denses) et peut être étendu pour inclure des fonctionnalités plus avancées.

## Fonctionnalités

- **Types de couches** : Couches denses (entièrement connectées)
- **Fonctions d'activation** :
  - Sigmoid
  - ReLU
  - Leaky ReLU
  - Tanh
  - Leurs dérivées (pour la rétropropagation)
- **Post-activation** : Softmax
- **Interface en ligne de commande** pour interagir avec le modèle :
  - Définir la séquence d'entrée
  - Spécifier le nombre de prédictions, de couches, de neurones, de fonctions d'activation, etc.

## Prérequis

- **Compilateur C++** : Le code est compatible avec C++17 et versions ultérieures.
- **Bibliothèques** :
  - Le projet ne nécessite aucune bibliothèque externe, tout est implémenté avec les bibliothèques standard de C++.
- **Outil de compilation** : Make (pour faciliter la compilation)

## Installation des dépendances

```bash
sudo apt update && sudo apt install g++ make
```

## Compilation et Utilisation

### 1. Cloner le dépôt

```bash
git clone https://github.com/dlhoumaud/gloomy.git
cd gloomy
```

### 2. Compiler le code

Vous pouvez compiler le projet avec `make` (assurez-vous que `make` est installé) :

```bash
make
```

Cela générera l'exécutable `gloomy` dans le répertoire `bin/`.

### 3. Exécuter le programme

Pour utiliser le programme, vous pouvez spécifier une séquence de valeurs d'entrée et divers paramètres via la ligne de commande.

Exemple d'utilisation :

```bash
./bin/gloomy "10.5 11.0 12.3" -c 3 -l 2 -n 4 -a sigmoid
```

#### Arguments

- **`<sequence_values>`** : Liste de valeurs séparées par des espaces (ex : `"10.5 11.0 12.3"`).
- **`-c C`** : Nombre de prédictions à générer (par défaut 1).
- **`-l L`** : Nombre de couches cachées (par défaut 2).
- **`-n N`** : Nombre de neurones par couche cachée (par défaut 2).
- **`-a [none|sigmoid|sigmoid_derivative|relu|leaky_relu|tanh|tanh_derivative]`** : Fonction d'activation (par défaut `none`).
- **`-A [none|softmax]`** : Fonction de post-activation (par défaut `none`).
- **`-h`** : Affiche l'aide.

#### Exemple de sortie

```bash
0.1258
0.1423
0.1579
0.1725
0.1850
```

Ce sont les prédictions générées en fonction de la séquence d'entrée donnée.

## Documentation détaillée

La documentation complète se trouve dans [`docs/README.md`](docs/README.md). Elle décrit les fonctions d'activation, l'effet de softmax, les couches, les neurones et les configurations recommandées.

## Aperçu du Code

### Architecture du Réseau de Neurones

L'architecture du réseau de neurones consiste en des couches entièrement connectées, chaque couche étant représentée par la classe `DenseLayer`. Cette classe gère l'initialisation des poids, la propagation avant des entrées et l'application de la fonction d'activation sélectionnée.

### Classes principales

- **DenseLayer** : Représente une couche dans le réseau de neurones. Elle gère l'initialisation des poids, la propagation avant et l'application des fonctions d'activation.
- **NeuralNetwork** : Encapsule le réseau de neurones, permettant l'ajout de couches et la prédiction des résultats.
- **Main** : Le point d'entrée de l'application, qui gère le parsing des entrées, la configuration du réseau de neurones et l'affichage des résultats.

### Fonctions d'Activation

- **Sigmoid** : `1 / (1 + exp(-x))`
- **ReLU** : `max(0, x)`
- **Leaky ReLU** : `alpha * x si x < 0 sinon x` (alpha par défaut = 0.01)
- **Tanh** : `tanh(x)`
- **Softmax** : Convertit les sorties en une distribution de probabilité.

### Flux d'Exécution

1. Les valeurs d'entrée sont extraites des arguments en ligne de commande.
2. Le réseau de neurones est configuré avec le nombre spécifié de couches, de neurones et de fonctions d'activation.
3. Le réseau effectue des prédictions basées sur les valeurs d'entrée et affiche les résultats.

## Contribution

Si vous souhaitez contribuer à ce projet, n'hésitez pas à forker le dépôt, effectuer vos modifications et soumettre une pull request.

### Licence

Ce projet est sous licence MIT.
