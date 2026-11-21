# Gloomy

Gloomy est un moteur expérimental C++17 avec un runtime Dense numérique et un prototype séparé de Transformer causal pour l'apprentissage du prochain token. Ce dernier permet d'entraîner sur un petit corpus texte, de reprendre son checkpoint et de générer, mais il n'a pas démontré de compréhension générale et n'est pas un LLM de qualité production.

## Capacités actuelles

- couches Dense, rétropropagation, activations `none`, `sigmoid`, `relu`, `leaky_relu`, `tanh` et softmax final ;
- pertes MSE, MAE et Huber; optimiseurs SGD, Momentum et Adam ;
- mémoires FIFO, Reservoir, Prioritized, Novelty et Hybrid, avec replay et scheduler ;
- mémoire FIFO expérimentale en int16/int8; les poids du réseau restent en float64 ;
- normalisation streaming et format de modèle `GLOOMY_MODEL` avec réseau, normalisation, état d'optimiseur et mémoire optionnelle ;
- benchmark synthétique avec métriques de qualité, temps et estimation mémoire.
- prototype texte expérimental : tokenizer BPE, Transformer causal CPU, cross-entropy, perplexité, checkpoint avec moments AdamW, génération autoregressive et évaluation exact-match; voir [l'exemple texte](docs/examples/language-model/README.md) et ses limites.

## Prérequis et compilation

Le projet utilise un compilateur C++17 (g++) et Make, sans dépendance C++ externe. Sur Debian/Ubuntu, installer les outils avec `sudo apt install g++ make`.

```bash
git clone https://github.com/dlhoumaud/gloomy.git
cd gloomy
```

```bash
make
make test
make benchmark
```

`make benchmark` exécute une campagne synthétique et écrit des CSV ignorés par Git. Ces résultats servent à comparer des variantes dans un environnement donné, pas à établir un classement universel ni à prédire les performances sur une cible embarquée.

## Runtime configuré

Le runtime configuré accepte un fichier de configuration et expose les modes `training`, `online_learning` et `inference` :

```bash
./bin/gloomy -f gloomy.config
```

Le format de configuration, le dataset et les options de surcharge sont décrits dans [`docs/configurations.md`](docs/configurations.md). Une configuration minimale d'inférence référence un modèle existant et fournit `runtime=inference` et `input=...`.

## Prototype de langage

Le chemin de langage est distinct du runtime Dense et utilise les sous-commandes `lm train`, `lm evaluate` et `lm generate`. Il s'agit d'un prototype CPU de petite échelle. Les corpus de démonstration sont fictifs; consulter le [protocole et les limites](docs/language-model-prototype.md) avant d'interpréter ses métriques.

## Démonstration CLI historique

Ce chemin crée un réseau neuf à chaque prédiction successive; il ne charge ni n'entraîne un modèle persistant. Il est conservé pour la démonstration :

```bash
./bin/gloomy "10.5 11.0 12.3" -c 3 -l 2 -n 4 -a sigmoid
```

- `<sequence_values>` : valeurs d'entrée séparées par des espaces ;
- `-c C` : nombre de prédictions (défaut `1`) ;
- `-l L` : couches cachées (défaut `2`) ;
- `-n N` : neurones par couche cachée (défaut `2`) ;
- `-a A` : activation ;
- `-A A` : post-activation (`none` ou `softmax`) ;
- `-h` : aide.

## Documentation

Commencer par [`docs/README.md`](docs/README.md). Les fonctionnalités livrées figurent dans [`CHANGELOG.md`](CHANGELOG.md); seuls les travaux à venir figurent dans [`docs/roadmap.md`](docs/roadmap.md).
