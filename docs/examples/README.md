# Exemples

Les exemples numériques et de langage sont disponibles dans les dossiers dédiés. [`language-model`](language-model/README.md) entraîne un petit Transformer causal, évalue sa perplexité et son exact match, puis génère une continuation. Ce jeu synthétique n'est pas un benchmark de compréhension générale.

Chaque sous-dossier fournit un README, les données et, selon le runtime, les fichiers de configuration ou les options CLI nécessaires.

- [Régression linéaire](regression-lineaire/README.md) : entraîner un modèle sur `y = 2x + 1`, sauvegarder un checkpoint et prédire de nouvelles valeurs.
- [Régression vectorielle](regression-vectorielle/README.md) : plusieurs valeurs en entrée et deux sorties numériques.
- [Replay quantifié int8](replay-int8/README.md) : apprentissage en ligne et mémoire FIFO quantifiée.
- [Apprentissage en ligne avancé](apprentissage-en-ligne-avance/README.md) : entrées/sorties vectorielles, deux régimes successifs, mémoire Hybrid et reprise du checkpoint.
- [Prototype de modèle de langage](language-model/README.md) : tokenizer BPE, Transformer causal, checkpoints, perplexité, question-réponse exacte et génération autoregressive.

Le runtime accepte actuellement un fichier texte avec une paire de vecteurs par ligne (`entrées | cibles`). Ces exemples varient les dimensions, modes et précisions de mémoire, mais ne constituent pas des lecteurs CSV ou JSON.
