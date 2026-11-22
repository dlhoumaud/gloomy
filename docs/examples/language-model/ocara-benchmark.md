# Benchmark documentaire Ocara

Les tâches, seuils provisoires et limites d'interprétation du benchmark sont
rassemblés dans le [protocole d'acceptation applicative](acceptance-protocol.md).

Ce benchmark expérimental étend l'assistant documentaire à la documentation
technique du projet Ocara fournie localement par l'utilisateur. Les sources sont
ignorées par Git sous `docs/examples/language-model-ocara/` et ne sont ni copiées
dans le dépôt ni redistribuées. L'autorisation rapportée couvre l'entraînement
et l'évaluation locaux; aucune licence de redistribution n'est affirmée.

## Préparer et mesurer

```sh
make benchmark-language-model-ocara
```

Le script [prepare_ocara_corpus.py](../../../scripts/prepare_ocara_corpus.py)
fixe les frontières au niveau des fichiers : 26 documents train, 5 validation,
6 test. Le manifeste conserve les tailles et SHA-256 des fichiers sources et
des splits, sans incorporer le texte source. Les questions QA sont dérivées
d'extraits de leur split : 20 exemples train réutilisés 32 fois pour
l'entraînement, 20 validation et 36 test. Validation et test sont équilibrés
entre `oui` et `non`; leur baseline majoritaire est donc 50 %.

`make benchmark-language-model-ocara` entraîne trois seeds avec la
configuration indiquée dans le Makefile, publie les mesures de validation,
puis mesure la génération et le choix fermé sur le test tenu à l'écart. La
configuration doit être choisie d'après la validation avant cette campagne.
Les résultats bruts et leur manifeste sont écrits dans
[ocara-3-seeds.csv](./ocara-3-seeds.csv) et
[ocara-3-seeds.manifest.json](./ocara-3-seeds.manifest.json). Les métriques
séparent explicitement la QA générative et la QA à labels fermés; aucune
moyenne composite ne masque leurs différences.

La tâche QA est un contrôle documentaire fermé, non une évaluation générale
de compréhension ou de génération métier. Les 36 questions test restent un
échantillon limité. Les bits par octet mesurent la prédiction du prochain token
sur les documents test; ils ne sont pas directement comparables à ceux d'un
autre corpus.

## Entraîner un checkpoint local

```sh
make language-model-ocara-example
```

Le checkpoint est créé localement dans `artifacts/ocara-docs.gloomy` (chemin
ignoré par Git). On peut vérifier la prédiction documentaire avec :

```sh
./bin/gloomy lm evaluate \
  --model artifacts/ocara-docs.gloomy \
  --text data/lm/ocara/test.txt \
  --qa data/lm/ocara/qa-test.txt \
  --qa-label oui --qa-label non
```

Un exemple de génération libre est :

```sh
./bin/gloomy lm generate \
  --model artifacts/ocara-docs.gloomy \
  --prompt "Question: SQLite::open crée-t-elle une base de données ? Contexte: Ouvre ou crée une base de données SQLite. Réponse:" \
  --tokens 4 --temperature 0 --seed 42
```

La génération libre ne constitue pas un score de QA et peut être incorrecte;
utiliser les jeux référencés et les métriques du benchmark pour comparer des
versions du modèle.
