# Protocole provisoire d'évaluation applicative du modèle de langage

Ce protocole rend les essais documentaires reproductibles. Il sert de proposition
à faire valider par les utilisateurs métier; les seuils ci-dessous sont des
portes expérimentales provisoires et ne valent pas acceptation applicative.

## Tâche, entrée et sortie

La tâche actuellement disponible est une QA factuelle documentaire en français.
L'entrée contient une question et un extrait de source sous la forme
`Question: … Contexte: … Réponse:`. La sortie attendue est le verdict `oui` ou
`non`. Le test Ocara contient 36 questions équilibrées; le test Gloomy en
contient 6. Les documents sont fournis dans le contexte: cette tâche ne mesure
pas la recherche de documents, le dialogue ouvert ni le raisonnement général.

Les corpus Ocara sont fournis localement pour entraînement et évaluation. Le
manifeste conserve leurs empreintes et les fichiers sources restent ignorés
par Git. Le corpus Gloomy est constitué de documentation du dépôt. Chaque
document reste dans un seul split, les QA sont dérivées d'extraits de leur
propre split et les sources des deux corpus sont apprises uniquement depuis
train. Les tailles et empreintes sont enregistrées dans les manifestes de
résultats correspondants.

## Référence et portes expérimentales proposées

La baseline majoritaire est de 50 % sur les jeux QA équilibrés. Pour un candidat
comparé à une configuration float64 de référence, proposer les portes suivantes
sur les trois seeds `42,43,44` :

- la moyenne de l'accuracy QA à labels fermés doit dépasser 50 % et au moins
  deux seeds sur trois doivent dépasser 50 % ;
- les bits par octet sur le même split test ne doivent pas augmenter par rapport
  à la référence ;
- publier séparément exact match génératif, F1 de tokens et accuracy à labels
  fermés, sans score composite ;
- conserver les résultats bruts par seed et leurs manifestes SHA-256.

Ces seuils doivent être confirmés ou remplacés par les utilisateurs métier,
avec les questions, réponses attendues et corpus autorisés. Toute modification
du protocole est décidée sur validation; le test tenu à l'écart ne sert qu'à la
comparaison finale. Les résultats Ocara actuels n'atteignent pas la première
porte : accuracy à labels fermés de 50 % pour chacune des trois seeds. Ils ne
permettent donc pas de conclure à une amélioration applicative.

## Mesures de coût

Le runner `scripts/benchmark_language_model.py` rapporte déjà la cross-entropy,
la perplexité, les bits par octet, les temps d'entraînement et d'évaluation,
la RSS maximale, le débit de tokens, la latence QA, la taille du checkpoint et
le nombre de paramètres. Les campagnes actuelles utilisent le Ryzen 3 5300U
local comme baseline CPU de développement. La cible de déploiement n'est pas
encore définie; il faut répéter ces mesures sur cette cible avant tout seuil de
latence, débit ou RAM de production.

Le pilote de précision est disponible dans
[`weight-precision-pilot.csv`](weight-precision-pilot.csv). Il compare poids
arrondis int8/int16, calcul float32 et produits scalaires int8/int16 avec
activations quantifiées, sur un checkpoint Gloomy et deux splits documentaires.
Sur Gloomy, float64/int8 donnent respectivement une cross-entropy de `4,289` /
`4,28907`, 50 % QA dans les deux cas et `3,2354` / `3,23545` bits par octet.
Le temps du processus float64/int8 est `5,88 s` / `1,70 s`, avec RSS de
`37 820 KiB` / `11 108 KiB`. Sur Ocara, float64/int8 donnent `5,25758` /
`5,25726` cross-entropy, 50 % QA fermée dans les deux cas, et le temps/RSS
passent de `47,47 s` / `38 616 KiB` à `14,19 s` / `11 908 KiB`. Le modèle est
entraîné sur Gloomy; Ocara est ici un test de sensibilité numérique
hors-domaine, pas une évaluation applicative validée. Le checkpoint garde ses
paramètres float64 et les opérandes entiers sont quantifiés à la volée : aucune
économie persistante de stockage n'est mesurée. Ces pilotes mono-checkpoint
doivent être répétés en multi-seed et sur une cible applicative avant d'en tirer
un seuil de performance. Le float64 de référence construit des nœuds
d'autodiffération en inférence alors que les modes réduits prennent un chemin
direct pour les couches linéaires; le gain de temps/RSS observé inclut ce
changement et n'isole pas l'effet de la seule précision numérique.

## Apprentissage continu

Le pilote [`sequence-replay-pilot.csv`](sequence-replay-pilot.csv) compare sans
replay, FIFO, Reservoir et replay priorisé par cross-entropy entre une phase
Gloomy et une adaptation Ocara d'une epoch. Il évalue les deux tests complets et
calcule le biais théorique de la distribution priorisée, sans correction
d'importance. Il reste à comparer plusieurs seeds et ordres de phases, plusieurs
capacités de replay, une correction d'importance, ainsi que la qualité QA après
adaptation; les updates ne sont pas égalisés, car les documents créent des
nombres différents de fenêtres. Dans ce run unique, l'exact match génératif
reste à 0 % et la QA fermée à 50 % pour les quatre stratégies; la cross-entropy
du test Gloomy baisse de `4,966` sans replay à `4,638` avec Prioritized. Ces
chiffres sont exploratoires, et la RSS du processus ne mesure pas précisément
la seule mémoire du buffer.
