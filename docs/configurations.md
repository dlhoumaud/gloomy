# Configurations recommandées

Il n'existe pas de configuration universellement la plus juste. Elle dépend des données, de l'échelle des valeurs, de l'horizon de prédiction et d'une validation hors échantillon. Les recommandations ci-dessous sont des points de départ pour l'entraînement MSE + SGD disponible dans les classes C++.

## Réglages par activation

| Usage | `-a` conseillé | `-l` conseillé | `-n` conseillé |
| --- | --- | ---: | ---: |
| Régression scalaire générale | `tanh` avec données normalisées, sinon `leaky_relu` | 1 à 3 | 8 à 64 |
| Relations simples ou presque linéaires | `none` ou `tanh` | 0 à 1 | 4 à 16 |
| Réseau profond général | `leaky_relu` | 2 à 4 | 16 à 128 |
| Sortie bornée entre 0 et 1 | `sigmoid` en sortie | 1 à 3 | 8 à 64 |
| Sortie bornée entre -1 et 1 | `tanh` en sortie | 1 à 3 | 8 à 64 |
| Classification multi-classe | `relu` ou `leaky_relu` caché, `softmax` final | 1 à 3 | 16 à 128 |

Les options `sigmoid_derivative` et `tanh_derivative` ne sont pas des choix réalistes pour les couches cachées d'un réseau entraîné. Elles sont conservées pour compatibilité CLI, tandis que la rétropropagation utilise leurs dérivées en interne.

## Méthode de sélection

1. Normaliser les entrées avec les statistiques du jeu d'entraînement uniquement.
2. Commencer par `-l 1 -n 16`, puis comparer `relu`, `leaky_relu` et `tanh`.
3. Utiliser une séparation entraînement/validation/test et mesurer une métrique adaptée : MAE ou RMSE en régression, exactitude ou entropie croisée en classification.
4. Augmenter `-l` ou `-n` seulement si l'erreur d'entraînement et l'erreur de validation le justifient.
5. Fixer la graine aléatoire et répéter plusieurs essais, car l'initialisation peut changer fortement le résultat.
6. Pour une série temporelle, conserver l'ordre temporel et comparer à une baseline simple, par exemple la dernière valeur connue.

## Recommandations spécifiques à ce dépôt

Pour le comportement CLI actuel, utiliser `-A none` pour une sortie numérique. `-A softmax` est inutile puisque la sortie a un seul neurone et donnera toujours `1`. Pour obtenir des probabilités de classes, il faut encore modifier la taille de sortie et utiliser une perte adaptée.

Exemples :

```bash
# Démonstration légère, sans couche cachée
./bin/gloomy "10.5 11.0 12.3" -l 0 -a none

# Régression bornée et réseau compact
./bin/gloomy "0.2 0.4 0.6" -l 2 -n 16 -a tanh -A none

# Plusieurs prédictions autoregressives de démonstration
./bin/gloomy "10.5 11.0 12.3" -c 5 -l 2 -n 16 -a leaky_relu
```

## Runtime configuré

Le CLI propose aussi un chemin explicite pour l'inférence et l'apprentissage via un fichier clé-valeur, sans dépendance externe. Les valeurs suivent la priorité `options CLI > fichier > défauts`.

```ini
runtime=training
model_path=model.gloomy
dataset_path=series.txt
activation=tanh
hidden_layers=2
neurons=16
loss=huber
huber_delta=1.0
optimizer=adam
learning_rate=0.001
memory_strategy=hybrid
memory_capacity=256
batch_size=8
epochs=10
seed=1234
```

Lancer l'entraînement avec `./bin/gloomy -f gloomy.config`. Le fichier de données contient un exemple par ligne, sous la forme `valeurs d'entrée | valeurs cibles`; les lignes vides et celles commençant par `#` sont ignorées. Toutes les entrées doivent avoir la même dimension, ainsi que toutes les cibles.

Les modes disponibles sont `training`, `online_learning` et `inference`. Le mode training entraîne chaque epoch sur le dataset et conserve aussi les exemples dans la mémoire choisie. Le mode online_learning ajoute chaque exemple à la mémoire puis déclenche le replay selon `train_every`. Dans ces deux modes, les statistiques d'entrée sont ajustées sur les entrées du dataset initial lors de la création d'un modèle, puis conservées lors des reprises.

Pour l'inférence, configurer `runtime=inference` et `input=1.0 2.0 3.0`. Le modèle existant est chargé depuis `model_path`, l'entrée est normalisée avec ses statistiques sauvegardées, et les sorties sont écrites une par ligne. Si le fichier modèle contient une mémoire, elle peut être restaurée pour l'entraînement ; le mode inference ne la charge pas.

Les options `--runtime`, `--model`, `--data`, `--input`, `--epochs` et `--batch-size` peuvent surcharger le fichier :

```bash
./bin/gloomy -f gloomy.config --runtime inference --input "1.0 2.0 3.0"
```

Le format du dataset est volontairement simple et ne couvre pas encore les CSV avec colonnes nommées, les données manquantes ou plusieurs fichiers.
