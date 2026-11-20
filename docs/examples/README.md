# Exemples

Chaque sous-dossier contient un exemple complet avec son README, les données et les fichiers de configuration nécessaires.

- [Régression linéaire](regression-lineaire/README.md) : entraîner un modèle sur `y = 2x + 1`, sauvegarder un checkpoint et prédire de nouvelles valeurs.
- [Régression vectorielle](regression-vectorielle/README.md) : plusieurs valeurs en entrée et deux sorties numériques.
- [Replay quantifié int8](replay-int8/README.md) : apprentissage en ligne et mémoire FIFO quantifiée.
- [Apprentissage en ligne avancé](apprentissage-en-ligne-avance/README.md) : entrées/sorties vectorielles, deux régimes successifs, mémoire Hybrid et reprise du checkpoint.

Le runtime accepte actuellement un fichier texte avec une paire de vecteurs par ligne (`entrées | cibles`). Ces exemples varient les dimensions, modes et précisions de mémoire, mais ne constituent pas des lecteurs CSV ou JSON.
