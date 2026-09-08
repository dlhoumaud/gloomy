# Configurations recommandées

Il n'existe pas de configuration universellement la plus juste. Elle dépend des données, de l'échelle des valeurs, de l'horizon de prédiction et d'une validation hors échantillon. Les recommandations ci-dessous sont des points de départ pour l'entraînement MSE + SGD disponible dans les classes C++.

Les valeurs par défaut effectivement utilisées par le CLI (et, à terme, par les futurs runtimes d'entraînement et d'online learning) sont centralisées dans `GloomyConfig` (`src/headers/GloomyConfig.h`).

## Fichier de configuration

Le CLI accepte `-f PATH` ou `--config PATH` pour charger un fichier `clé=valeur` minimal, parsé par `GloomyConfigFile` (`src/headers/GloomyConfigFile.h`) :

```ini
# commentaire
activation=tanh
hidden_layers=2
neurons=16
learning_rate=0.001
memory_strategy=hybrid
memory_capacity=256
```

Règles du format :

- une paire `clé=valeur` par ligne ; les espaces autour de la clé et de la valeur sont ignorés ;
- les lignes vides et celles commençant par `#` ou `;` sont des commentaires ;
- une clé inconnue, une ligne sans `=`, ou une valeur numérique invalide font échouer le chargement avec un message explicite ;
- pas de sections, pas de guillemets : chaque valeur est une chaîne brute jusqu'à la fin de la ligne (déjà suffisant pour les clés actuelles, qui sont des nombres ou des mots simples).

Priorité de résolution : **CLI > fichier > défauts**. Concrètement, le CLI applique d'abord les défauts de `GloomyConfig`, puis les valeurs du fichier passé à `-f`/`--config` si présent, puis les flags `-c`/`-l`/`-n`/`-a`/`-A` explicites, qui l'emportent toujours.

La clé `runtime` sélectionne le mode d'exécution du CLI :

- `runtime=inference` (défaut) : comportement historique, seules `activation`, `post_activation`, `hidden_layers`, `neurons` et `predictions` s'appliquent.
- `runtime=online_learning` : exécute la boucle observation → normalisation → prédiction → cible → erreur → mémoire → scheduler → replay → mise à jour sur la séquence d'entrée (voir [Mémoire d'apprentissage](memory.md) et [feuille de route](roadmap.md), point 6). Toutes les clés de `GloomyConfig` s'appliquent alors : `loss`/`huber_delta`, `optimizer`/`learning_rate`/`momentum`/`beta1`/`beta2`/`epsilon`, `memory_strategy`/`memory_capacity`/`recent_ratio`/`error_ratio`/`novelty_ratio`/`historical_ratio`/`novelty_threshold`/`prioritized_alpha`/`prioritized_beta`/`seed`, `train_every`, `batch_size`. `prioritized_beta` (défaut `0.4`) contrôle la correction de biais d'échantillonnage du prioritized replay (`0` la désactive) — voir [Mémoire d'apprentissage](memory.md).
- `runtime=training` : entraîne un réseau complet sur la séquence fournie en mode batch complet sur plusieurs `epochs` (`LearningEngine::train()`), puis sauvegarde automatiquement le modèle si `model_path` est renseigné. Les clés applicables sont les mêmes que pour `online_learning`, plus `epochs`.

Exemple :

```ini
# gloomy.config
runtime=online_learning
activation=tanh
hidden_layers=1
neurons=8
loss=huber
huber_delta=1.0
optimizer=momentum
learning_rate=0.01
momentum=0.9
memory_strategy=hybrid
memory_capacity=64
train_every=1
batch_size=4
epochs=2
```

```bash
./bin/gloomy "1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0" -f gloomy.config
```

Chaque valeur consécutive de la séquence devient une observation (`x[i]`) et sa cible (`x[i+1]`) : le réseau du runtime online a donc une entrée et une sortie de dimension `1`, quels que soient `hidden_layers`/`neurons` (qui ne dimensionnent que les couches cachées). Une ligne est affichée par observation (`index observation cible prédiction perte`, avant mise à jour des poids), suivie d'un résumé (`average_loss`, `memory_size`).

Les chemins de persistance (`model_path`, `optimizer_path`, `memory_path`, `metrics_path`) sont **vides par défaut et opt-in** : les runtimes `online_learning` et `training` ne sauvegardent chaque artefact que si son chemin est explicitement renseigné dans la configuration.

- `model_path` : sauvegarde le réseau entraîné, la normalisation, l'optimiseur et la mémoire dans un fichier `GLOOMY_MODEL` unifié (`ModelSerialization`) au terme de l'exécution. `runtime=online_learning` le lit aussi au démarrage : si le fichier existe déjà, l'exécution **reprend** l'état sauvegardé au lieu de repartir d'un réseau neuf.
- `optimizer_path` / `memory_path` : sauvegardent séparément l'optimiseur (`OptimizerSerialization`) et la mémoire d'apprentissage (`LearningMemorySerialization`), en plus du fichier unifié si `model_path` est aussi renseigné.
- `metrics_path` : écrit un résumé texte (`average_loss=...`, et `memory_size=...` pour `online_learning`).

**Ne pas confondre avec `make benchmark`**, qui produit son propre `benchmark_results.csv` indépendamment de `GloomyConfig` : ne pas régler `metrics_path=benchmark_results.csv` sur un runtime CLI dans le même répertoire, sous peine d'écraser ce fichier avec un simple résumé texte.

C'est la première brique du futur fichier `gloomy.config` décrit dans la [feuille de route](roadmap.md), section « Configuration fichier ».

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
