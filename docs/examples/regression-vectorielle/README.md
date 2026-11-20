# Régression vectorielle

Cet exemple entraîne un réseau à deux entrées et deux sorties. Il apprend les relations `cible1 = 2 × entrée1 + entrée2` et `cible2 = entrée1 - 3 × entrée2`.

Toutes les commandes se lancent depuis la racine du dépôt. Le dataset utilise le format texte `vecteur d'entrée | vecteur cible`.

```bash
make
mkdir -p artifacts/examples
./bin/gloomy -f docs/examples/regression-vectorielle/training.config
./bin/gloomy -f docs/examples/regression-vectorielle/inference.config
```

La configuration n'a pas de couche cachée : elle convient à ces relations linéaires. Le dataset et les deux configurations sont fournis dans ce dossier. L'entraînement enregistre un checkpoint `GLOOMY_MODEL` et un artefact `GLOOMY_INFER`; l'inférence utilise ce dernier, sans charger l'optimiseur. L'entrée `0.5 -0.5` correspond aux cibles attendues `0.5 2`.
