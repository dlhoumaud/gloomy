# Apprentissage en ligne avec replay int8

Cet exemple utilise une mémoire FIFO dont les vecteurs d'entrée et de cible sont stockés en int8. Le réseau et ses paramètres d'entraînement restent en float64; seule la mémoire de replay est quantifiée.

Toutes les commandes se lancent depuis la racine du dépôt :

```bash
make
mkdir -p artifacts/examples
./bin/gloomy -f docs/examples/replay-int8/phase-a.config
./bin/gloomy -f docs/examples/replay-int8/phase-b.config
./bin/gloomy -f docs/examples/replay-int8/inference.config
```

La première phase entraîne le modèle par mini-batches et crée le checkpoint avec la mémoire. La deuxième réutilise le même `model_path`, restaure le replay, puis poursuit l'apprentissage en ligne sur de nouveaux exemples de la même relation `y = 2x + 1`. La valeur d'entrée reste dans la plage de calibration de la mémoire. `precision=int8` ne convertit pas les poids du réseau.
