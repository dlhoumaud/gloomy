# Configurations et limites

Il n'existe pas de configuration universellement la plus juste. Elle dépend des données, de l'échelle des valeurs, de l'horizon de prédiction et d'une validation hors échantillon. Les réglages ci-dessous sont des heuristiques de capacité pour un petit réseau Dense entraîné, pas des valeurs optimales garanties. Le CLI historique expose les options d'architecture mais ne charge ni n'entraîne un modèle persistant. Le runtime configuré applique une activation commune aux couches; il ne permet pas de choisir séparément l'activation de sortie.

## Réglages par activation

| Usage | `-a` conseillé | `-l` conseillé | `-n` conseillé |
| --- | --- | ---: | ---: |
| Régression scalaire générale | `tanh` avec données normalisées, sinon `leaky_relu` | 1 à 3 | 8 à 64 |
| Relations simples ou presque linéaires | `none` ou `tanh` | 0 à 1 | 4 à 16 |
| Réseau profond général | `leaky_relu` | 2 à 4 | 16 à 128 |
| Sortie bornée entre 0 et 1 | `sigmoid` (également appliqué aux couches cachées dans le CLI actuel) | 1 à 3 | 8 à 64 |
| Sortie bornée entre -1 et 1 | `tanh` (également appliqué aux couches cachées dans le CLI actuel) | 1 à 3 | 8 à 64 |
| Classification multi-classe | non prise en charge comme tâche complète; sortie K et softmax seuls ne suffisent pas | — | — |

Les options `sigmoid_derivative` et `tanh_derivative` ne sont pas des choix réalistes pour un réseau entraîné. Elles ne sont acceptées que par le CLI historique; la rétropropagation utilise les dérivées en interne.

## Méthode de sélection

1. Normaliser les entrées avec les statistiques du jeu d'entraînement uniquement.
2. Commencer par `-l 1 -n 16`, puis comparer `relu`, `leaky_relu` et `tanh`.
3. Créer une séparation entraînement/validation/test adaptée au problème et mesurer une métrique pertinente. Le runtime ne crée pas automatiquement ces partitions; le benchmark intégré n'utilise qu'un dataset synthétique.
4. Augmenter `-l` ou `-n` seulement si l'erreur d'entraînement et l'erreur de validation le justifient.
5. Fixer la graine aléatoire et répéter plusieurs essais, car l'initialisation peut changer fortement le résultat.
6. Pour une série temporelle, conserver l'ordre temporel et comparer à une baseline simple, par exemple la dernière valeur connue.

## Recommandations spécifiques à ce dépôt

Pour le CLI historique, utiliser `-A none` pour une sortie numérique. `-A softmax` est inutile puisque ce chemin a un seul neurone de sortie et donnera toujours `1`. Le runtime configuré détermine la taille de sortie à partir des cibles et accepte `post_activation=softmax`, mais il ne fournit pas encore de perte classification/cross-entropy ni d'interface de décision de classe.

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

Les modes disponibles sont `training`, `online_learning` et `inference`. Le mode `training` entraîne chaque epoch sur le dataset et ajoute aussi les exemples à la mémoire choisie. Le mode `online_learning` traite les exemples dans l'ordre du fichier, les ajoute à la mémoire et déclenche un replay selon `train_every`. Les exemples sont lus ligne par ligne; le runtime ne conserve qu'un mini-batch, ou un exemple en ligne, en plus du replay borné. Le fichier est relu à chaque epoch et lors des passes de statistiques/calibration.

`memory_budget_bytes` est facultatif et borne l'estimation des allocations retenues par le replay. Lorsqu'il est présent, le runtime réduit `memory_capacity` si nécessaire en tenant compte des dimensions, métadonnées, précision et buffers scratch connus. Cette capacité est calculée par estimation et ne constitue pas une limite RSS stricte : les frais de l'allocateur et les variations de capacité des conteneurs peuvent différer. Un budget trop petit pour un exemple est rejeté. Une mémoire restaurée est refusée si ses allocations mesurées dépassent le budget. Le budget s'applique uniquement au replay; `allocated_bytes` inclut aussi réseau, optimiseur et moteur, et `peak_rss_bytes` indique le maximum rapporté par le système d'exploitation sur stderr après l'exécution.

Pour un nouveau modèle d'entraînement, les statistiques d'entrée sont calculées sur toutes les entrées du dataset fourni. Il n'y a pas de validation/test automatique : fournissez uniquement les données d'entraînement et évaluez séparément. Lorsqu'un `model_path` existe en mode entraînement, le réseau, le normaliseur et l'optimiseur sont chargés depuis ce modèle; les paramètres d'architecture de la configuration ne reconstruisent pas le réseau, et l'état de l'optimiseur sauvegardé remplace ses hyperparamètres. La mémoire est restaurée si elle est présente; dans ce cas, ses paramètres remplacent ceux utilisés pour créer une mémoire depuis la configuration.

Pour l'inférence, configurer `runtime=inference` et `input=1.0 2.0 3.0`. Le runtime détecte le format du fichier `model_path`. Avec `GLOOMY_INFER`, il charge uniquement réseau et normalisation, sans dépendre du type d'optimiseur. Avec un checkpoint `GLOOMY_MODEL`, il vérifie toujours le type d'optimiseur et le checksum, mais saute la désérialisation de la mémoire.

Pour produire l'artefact allégé pendant l'entraînement, renseigner `inference_model_path=deployment.infer` ou utiliser `--export-inference deployment.infer`. Il est enregistré en plus du checkpoint de reprise et son chemin doit être différent de `model_path`.

Les options `--runtime`, `--model`, `--data`, `--input`, `--epochs`, `--batch-size` et `--export-inference` peuvent surcharger le fichier :

```bash
./bin/gloomy -f gloomy.config --runtime inference --input "1.0 2.0 3.0"
```

Le format du dataset est volontairement simple et ne couvre pas encore les CSV avec colonnes nommées, les données manquantes ou plusieurs fichiers. Les options de surcharge ne couvrent pas tous les paramètres de configuration.
