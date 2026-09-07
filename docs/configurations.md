# Configurations recommandées

Il n'existe pas de configuration universellement la plus juste. Elle dépend des données, de l'échelle des valeurs, de l'horizon de prédiction et d'une validation hors échantillon. Les recommandations ci-dessous sont des points de départ pour une future version entraînée.

## Réglages par activation

| Usage | `-a` conseillé | `-l` conseillé | `-n` conseillé |
| --- | --- | ---: | ---: |
| Régression scalaire générale | `tanh` avec données normalisées, sinon `leaky_relu` | 1 à 3 | 8 à 64 |
| Relations simples ou presque linéaires | `none` ou `tanh` | 0 à 1 | 4 à 16 |
| Réseau profond général | `leaky_relu` | 2 à 4 | 16 à 128 |
| Sortie bornée entre 0 et 1 | `sigmoid` en sortie | 1 à 3 | 8 à 64 |
| Sortie bornée entre -1 et 1 | `tanh` en sortie | 1 à 3 | 8 à 64 |
| Classification multi-classe | `relu` ou `leaky_relu` caché, `softmax` final | 1 à 3 | 16 à 128 |

Les options `sigmoid_derivative` et `tanh_derivative` ne sont pas des choix réalistes pour les couches cachées d'un réseau entraîné. Elles doivent servir au calcul des gradients, dans une future rétropropagation.

## Méthode de sélection

1. Normaliser les entrées avec les statistiques du jeu d'entraînement uniquement.
2. Commencer par `-l 1 -n 16`, puis comparer `relu`, `leaky_relu` et `tanh`.
3. Utiliser une séparation entraînement/validation/test et mesurer une métrique adaptée : MAE ou RMSE en régression, exactitude ou entropie croisée en classification.
4. Augmenter `-l` ou `-n` seulement si l'erreur d'entraînement et l'erreur de validation le justifient.
5. Fixer la graine aléatoire et répéter plusieurs essais, car l'initialisation peut changer fortement le résultat.
6. Pour une série temporelle, conserver l'ordre temporel et comparer à une baseline simple, par exemple la dernière valeur connue.

## Recommandations spécifiques à ce dépôt

Pour le comportement actuel, utiliser `-A none` pour une sortie numérique. `-A softmax` est inutile puisque la sortie a un seul neurone et donnera toujours `1`. Pour obtenir des probabilités de classes, il faut d'abord modifier la taille de sortie, ajouter des cibles et implémenter l'entraînement.

Exemples :

```bash
# Démonstration légère, sans couche cachée
./bin/gloomy "10.5 11.0 12.3" -l 0 -a none

# Régression bornée et réseau compact
./bin/gloomy "0.2 0.4 0.6" -l 2 -n 16 -a tanh -A none

# Plusieurs prédictions autoregressives de démonstration
./bin/gloomy "10.5 11.0 12.3" -c 5 -l 2 -n 16 -a leaky_relu
```
