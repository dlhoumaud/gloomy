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

Un premier pilote de poids est disponible dans
[`weight-precision-pilot.csv`](weight-precision-pilot.csv). Sur un checkpoint
Gloomy et ses six QA de test, l'arrondi int8 par matrice a une erreur absolue
maximale de poids de `0,00417`; la cross-entropy est `4,28818` contre `4,289`
en float64, et l'accuracy QA reste à 50 %. En int16, l'erreur maximale est
`0,0000162` et les métriques rapportées sont identiques à float64. La RSS reste
environ 37 MiB dans les trois cas et le checkpoint stocke toujours des doubles :
ce pilote ne montre aucun gain mémoire ni de calcul entier.

La quantification réelle des poids, des activations et du calcul doit encore
être évaluée au float64 sur les mêmes splits et seeds, en rapportant l'erreur
après déquantification, les métriques de tâche, le temps, la mémoire et le coût
de conversion. Le pilote ne simule que l'arrondi/déquantification des poids;
activations et opérations restent en float64. Les modes int8/int16 du runtime
Dense quantifient uniquement les échantillons du replay et ne sont pas une
mesure de précision réduite du Transformer.

## Apprentissage continu

Le runtime du Transformer entraîne actuellement sur un corpus et sait reprendre
un checkpoint; il n'expose pas de replay de séquences pour un flux de tâches.
Si l'apprentissage continu devient un besoin métier, comparer sans replay et
avec replay uniforme/priorisé, mesurer rétention et coût mémoire, et corriger
ou quantifier le biais de Prioritized Replay. Cette expérience devra employer
des phases documentaires séparées et des questions de rétention tenues à
l'écart.
