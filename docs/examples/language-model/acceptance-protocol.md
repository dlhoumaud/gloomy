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

Le pilote de précision est publié dans
[`weight-precision-pilot.csv`](weight-precision-pilot.csv) et
[`ocara-precision-pilot.csv`](ocara-precision-pilot.csv). Sur un seul
checkpoint Gloomy, Gloomy float64/int8 donnent une cross-entropy de `4,289` /
`4,28907`, 50 % QA fermée dans les deux cas et `3,23540` / `3,23545` bits par
octet. Le temps et la RSS sont `4,71 s` / `37 812 KiB` et `1,25 s` /
`11 112 KiB` pour l'inférence int8 avec activations quantifiées. Sur Ocara,
float64/int8 donnent `5,25758` / `5,25726` de cross-entropy et `4,83807` /
`4,83778` bits par octet; le temps/RSS sont `22,83 s` / `38 632 KiB` et
`6,79 s` / `11 924 KiB`. Le checkpoint compact int8 fait `22 855` octets au
lieu de `251 575` (−90,9 %), int16 `32 071` (−87,3 %); la conversion dure
`1,4–2,0 ms`. L'erreur de quantification maximale/moyenne est `0,00417` /
`0,00175` en int8 et `0,0000162` / `0,00000677` en int16. Le chargement reconstruit toutefois des paramètres float64 et la
RSS reste autour de 38 MiB: seule l'empreinte fichier baisse. Le modèle est
entraîné sur Gloomy; Ocara est un test de sensibilité hors domaine, pas une
évaluation applicative validée. Les pilotes restent mono-checkpoint; il faut
les répéter en multi-seed et sur la cible de déploiement. Le float64 de référence construit des nœuds
d'autodiffération en inférence alors que les modes réduits prennent un chemin
direct pour les couches linéaires; le gain de temps/RSS observé inclut ce
changement et n'isole pas l'effet de la seule précision numérique.

## Apprentissage continu

Le pilote [`sequence-replay-pilot.csv`](sequence-replay-pilot.csv) compare sans
replay, FIFO, Reservoir, priorisé et priorisé avec correction d'importance
`1/(N·pᵢ)` (β=1), sur une phase Gloomy suivie d'une adaptation Ocara. Le CLI
applique ces poids par token supervisé et accepte les poids documentaires via
`--sample-weights`. Chaque document est moyenné sur ses tokens supervisés avant
d'appliquer le poids; le dénominateur du batch ne dépend pas des poids, ce qui
évite l'auto-normalisation biaisée. La campagne doit encore couvrir plusieurs seeds, ordres de
phases et capacités; égaliser les updates; et distinguer la mémoire du buffer
de la RSS du processus. L'évaluation rapporte la rétention sur les deux tests
et la QA après adaptation; ces résultats restent exploratoires.
