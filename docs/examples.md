# Du benchmark au modèle réutilisable : guide pratique

Cette page répond à une question concrète, en une seule fois : **comment mesurer, sur ma machine, quelle configuration de Gloomy est la plus adaptée, puis l'utiliser pour entraîner un modèle et le réutiliser ?**

Toutes les commandes ci-dessous ont été exécutées telles quelles avec le CLI actuel ; les sorties sont copiées telles quelles. Elles restent reproductibles à seed égale (voir [Couches et neurones](architecture.md), « Initialisation des poids et reproductibilité ») : réexécuter une commande de cette page sur une machine différente donnera les mêmes colonnes de perte/MAE, seules les colonnes de temps varieront.

## Étape 0 — Rappel : ce que Gloomy sait apprendre aujourd'hui

Gloomy n'apprend pas de texte, d'images ni de conversation. C'est un moteur de réseau dense qui sait aujourd'hui faire deux types d'apprentissage réel sur des séries **scalaires** (voir [Configurations et limites](configurations.md)) :

- `runtime=online_learning` : apprend en continu à prédire la valeur suivante d'une série (`x[i] -> x[i+1]`, ou une fenêtre `x[i..i+window_size-1] -> x[i+window_size]` si `window_size > 1`) ;
- `runtime=training` : entraîne un réseau complet sur plusieurs `epochs`, puis sauvegarde l'état si `model_path` est renseigné.

Le reste de cette page part du principe que vous avez déjà compilé le projet :

```bash
make clean
make test
make
```

## Étape 1 — Générer un benchmark complet

```bash
make benchmark
ls -1 *.csv
```

```text
benchmark_baseline.csv
benchmark_forgetting.csv
benchmark_full_dataset.csv
benchmark_memory_capacity.csv
benchmark_quantization.csv
benchmark_results.csv
```

`make benchmark` entraîne, sur une régression synthétique déterministe (`y = 2x + 1`), toutes les combinaisons utiles : 5 stratégies de mémoire × 4 capacités × 3 optimiseurs × 3 pertes × 3 seeds (`benchmark_memory_capacity.csv`, 540 lignes), les mêmes capacités en int16/int8 (`benchmark_quantization.csv`, 216 lignes), une baseline naïve et un dataset complet, ainsi que deux scénarios de catastrophic forgetting. Le détail exhaustif de chaque colonne est dans [Benchmark](benchmark.md) ; cette page n'utilise que les colonnes nécessaires pour choisir une configuration.

### Pourquoi une mémoire de replay, avant même de choisir laquelle

Avant de comparer les stratégies entre elles, `benchmark_forgetting.csv` répond à une question plus basique : est-ce qu'une mémoire de replay sert à quelque chose du tout ?

```bash
cat benchmark_forgetting.csv | cut -d, -f1-4,14
```

```text
memory_strategy,precision,optimizer,loss_function,forgetting
forgetting_no_replay,float64,sgd,mse,-19.262444948880798
forgetting_fifo_replay,float64,sgd,mse,-4.886881093972506
```

Le scénario entraîne d'abord sur un régime A (relation croissante), puis sur un régime B (relation inverse), et mesure la dégradation de la performance sur A (`Metrics::forgetting = perte_avant - perte_après` ; une valeur négative signifie que la perte a augmenté). **Sans replay, l'oubli est environ 4 fois plus important** (`-19.26` contre `-4.89`) : c'est le problème que toute stratégie de mémoire ci-dessous essaie d'atténuer, chacune à sa manière (voir [Mémoire d'apprentissage](memory.md)).

## Étape 2 — Lire le benchmark pour déterminer les modèles les plus adaptés à cette machine

### 2.1 Piège d'environnement à connaître avant de trier un CSV

Sur une machine en locale française (`LANG=fr_FR.UTF-8`, le cas le plus courant), `sort -g`/`awk` interprètent la collation des nombres différemment de l'anglais et peuvent trier un CSV silencieusement dans le mauvais ordre, sans erreur — vérifié sur cette machine :

```bash
# Ordre incorrect (locale par défaut) : 10.10 se glisse avant 1.10
sort -t, -k4,4g mon_fichier.csv | head -3

# Ordre correct
LC_ALL=C sort -t, -k4,4g mon_fichier.csv | head -3
```

Toutes les commandes de tri ci-dessous préfixent donc systématiquement `LC_ALL=C`. C'est la même famille de piège que le contournement `mawk` déjà documenté dans [Benchmark](benchmark.md).

### 2.2 Comparer les stratégies de mémoire et les optimiseurs à capacité fixée

Colonnes utiles : `memory_strategy` (1), `optimizer` (3), `loss_function` (4), `memory_capacity` (15), `seed` (17), `mae_mean` (18). `mae_mean` est identique sur les 3 lignes d'un même scénario (une par seed) : filtrer `seed==1234` suffit pour obtenir une ligne par combinaison.

```bash
LC_ALL=C awk -F, 'NR==1{next} $15==64 && $4=="mse" && $17==1234 {
    print $1","$3","$18
}' benchmark_memory_capacity.csv | LC_ALL=C sort -t, -k3,3g
```

```text
hybrid,momentum,0.12232619904977486
reservoir,momentum,0.21449506384491357
novelty,sgd,0.32306438705266211
novelty,adam,0.33943695448817462
reservoir,adam,0.43124089876872329
prioritized,sgd,0.47131195120608699
hybrid,adam,0.54982731716184785
novelty,momentum,0.66477729053319312
prioritized,adam,0.71855612191565876
fifo,adam,0.78810503640962615
prioritized,momentum,0.9926056313626862
reservoir,sgd,1.105401764568849
hybrid,sgd,6.1998596610751919
fifo,sgd,6.5726939905518362
fifo,momentum,10.101639389146948
```

Sur cette machine, à capacité 64 et perte MSE : **`memory_strategy=hybrid` + `optimizer=momentum`** obtient le plus bas MAE moyen. C'est la combinaison retenue pour la suite de cette page — mais le point important est la méthode, pas ce résultat précis : rejouez cette commande sur votre propre machine et vos propres données avant de figer un choix.

### 2.3 Vérifier l'effet de la capacité mémoire pour la stratégie retenue

```bash
LC_ALL=C awk -F, 'NR==1{next} $1=="hybrid" && $3=="momentum" && $4=="mse" && $17==1234 {
    print $15","$18","$11","$23
}' benchmark_memory_capacity.csv | LC_ALL=C sort -t, -k1,1g \
  | LC_ALL=C awk -F, '{printf "capacite=%-4s mae_mean=%-8.4f memoire=%-6s octets updates_par_seconde=%.0f\n",$1,$2,$3,$4}'
```

```text
capacite=32   mae_mean=0.4342   memoire=1840   octets updates_par_seconde=222625
capacite=64   mae_mean=0.1223   memoire=2960   octets updates_par_seconde=219600
capacite=128  mae_mean=0.7019   memoire=4560   octets updates_par_seconde=221058
capacite=256  mae_mean=0.4463   memoire=6800   octets updates_par_seconde=215703
```

À retenir : **plus de mémoire n'est pas systématiquement meilleur** ici — `capacite=64` bat `128` et `256` sur ce scénario précis (un seul passage online sur un petit jeu de données synthétique, sensible à l'ordre d'arrivée des échantillons). Le débit (`updates_par_seconde`) varie peu d'une capacité à l'autre sur cette machine : la capacité doit donc être choisie sur le MAE et le budget mémoire, pas sur la vitesse.

### 2.4 Réduire l'empreinte mémoire : comparer les précisions

Seule `FIFOMemory` dispose aujourd'hui d'une variante quantifiée (`QuantizedFIFOMemory` int16, `QuantizedInt8FIFOMemory` int8 — voir [Quantification](quantization.md)) ; la comparaison porte donc sur FIFO :

```bash
{
LC_ALL=C awk -F, 'NR==1{next} $1=="fifo" && $3=="sgd" && $4=="mse" && $17==1234 {print $15","$2","$18","$11}' benchmark_memory_capacity.csv
LC_ALL=C awk -F, 'NR==1{next} $3=="sgd" && $4=="mse" && $17==1234 {print $15","$2","$18","$11}' benchmark_quantization.csv
} | LC_ALL=C sort -t, -k1,1g -k2,2 \
  | LC_ALL=C awk -F, '{printf "capacite=%-4s precision=%-8s mae_mean=%-8.4f memoire=%s octets\n",$1,$2,$3,$4}'
```

```text
capacite=32   precision=float64  mae_mean=1.1587   memoire=2760 octets
capacite=32   precision=int16    mae_mean=1.1586   memoire=2376 octets
capacite=32   precision=int8     mae_mean=1.1972   memoire=2312 octets
capacite=64   precision=float64  mae_mean=6.5727   memoire=5320 octets
capacite=64   precision=int16    mae_mean=6.5727   memoire=4552 octets
capacite=64   precision=int8     mae_mean=6.5177   memoire=4424 octets
capacite=128  precision=float64  mae_mean=13.9855  memoire=6600 octets
capacite=128  precision=int16    mae_mean=13.9854  memoire=5640 octets
capacite=128  precision=int8     mae_mean=14.0699  memoire=5480 octets
capacite=256  precision=float64  mae_mean=13.9855  memoire=6600 octets
capacite=256  precision=int16    mae_mean=13.9854  memoire=5640 octets
capacite=256  precision=int8     mae_mean=14.0699  memoire=5480 octets
```

Deux observations, propres à ce scénario mais utiles comme méthode : la quantification ne dégrade quasiment pas le MAE ici (parfois même légèrement mieux, comme pour int8 à capacité 64) tout en réduisant la mémoire de 10 à 17 % ; et les lignes `128`/`256` sont identiques, signe que FIFO ne remplit jamais sa capacité au-delà de la taille réelle du jeu de données sur ce scénario — augmenter `memory_capacity` au-delà de ce point serait sans effet.

### 2.5 Méthode récapitulative

1. `make benchmark` sur la machine cible (les temps sont dépendants de la machine, pas les colonnes de perte).
2. Fixer une capacité candidate, filtrer une seed, trier les combinaisons `memory_strategy`/`optimizer` par `mae_mean` (2.2).
3. Vérifier que la capacité choisie est bien un optimum local, pas juste « la plus grande disponible » (2.3).
4. Si le budget mémoire est contraint, comparer les précisions disponibles pour la stratégie retenue (2.4).
5. Ne jamais figer un choix sur ce seul benchmark synthétique : le reconduire sur des données proches de l'usage réel visé (voir [Benchmark](benchmark.md), limites).

## Étape 3 — Utiliser le modèle choisi sur un exemple concret

Le résultat de l'étape 2 se traduit directement en fichier de configuration. `novelty_threshold=1.0` est explicité ici parce que c'est la valeur utilisée par le benchmark pour `hybrid`/`novelty` (le défaut de `GloomyConfig` seul est `0.0` — voir [Configurations et limites](configurations.md)) :

```ini
# ex_best.config
runtime=online_learning
activation=tanh
hidden_layers=1
neurons=8
loss=mse
optimizer=momentum
learning_rate=0.01
momentum=0.9
memory_strategy=hybrid
memory_capacity=64
novelty_threshold=1.0
seed=1234
```

Appliqué à un relevé de température horaire (montée puis descente dans la journée) :

```bash
./bin/gloomy "18 19 19.5 21 23 25 26 24 22 20 19 18.5" -f ex_best.config
```

```text
0	18	19	0	0	0
1	19	19.5	-0.339116	2.73573	0
2	19.5	21	-0.296587	6.46938	0
3	21	23	-0.214105	7.51028	0
4	23	25	0.130094	6.29485	0
5	25	26	0.563817	4.34373	0
6	26	24	0.815546	3.01992	0
7	24	22	0.871359	2.4984	0
8	22	20	0.838885	2.42661	0
9	20	19	0.661895	2.85482	0
10	19	18.5	0.508799	2.65116	0
average_loss=3.70953 memory_size=11
```

Chaque ligne est `index observation cible prédiction perte dérive` (perte avant la mise à jour de ce pas ; `dérive` vaut `1`/`0`, voir `concept_drift_detection` dans [Mémoire d'apprentissage](memory.md)).

### Variante — que faire face à une valeur aberrante ?

Le choix de la perte fait partie du même exercice de sélection. Un capteur globalement stable avec un pic ponctuel (défaut de mesure), même architecture, seule `loss` change :

```ini
# ex_mse.config
runtime=online_learning
activation=tanh
hidden_layers=1
neurons=8
loss=mse
optimizer=sgd
learning_rate=0.05
memory_strategy=fifo
memory_capacity=16
```

```ini
# ex_huber.config — identique, sauf loss=huber, huber_delta=1.0
runtime=online_learning
activation=tanh
hidden_layers=1
neurons=8
loss=huber
huber_delta=1.0
optimizer=sgd
learning_rate=0.05
memory_strategy=fifo
memory_capacity=16
```

```bash
./bin/gloomy "20 20.5 21 45 21.5 22 22.5 23" -f ex_mse.config   # loss=mse
./bin/gloomy "20 20.5 21 45 21.5 22 22.5 23" -f ex_huber.config # loss=huber, huber_delta=1.0
```

```text
# loss=mse : le pic (45) fait exploser la perte, qui reste ~500 plusieurs pas après
average_loss=548.845 memory_size=7

# loss=huber : le même pic ne dépasse jamais ~21 de perte, retour à un chiffre dès le pas suivant
average_loss=9.8784 memory_size=7
```

C'est le compromis documenté dans [Fonctions de perte](losses.md) : Huber est quadratique près de la cible et linéaire pour les grandes erreurs, donc moins dominée par un aberrant.

## Étape 4 — Générer (entraîner) un modèle et le sauvegarder

`runtime=training` entraîne sur l'ensemble de la séquence pendant `epochs` passes, puis sauvegarde l'état complet (réseau, normalisation, optimiseur, mémoire) dans le fichier unifié `GLOOMY_MODEL` si `model_path` est renseigné :

```ini
# train.config
runtime=training
activation=tanh
hidden_layers=1
neurons=8
loss=mse
optimizer=momentum
learning_rate=0.01
momentum=0.9
memory_strategy=hybrid
memory_capacity=64
novelty_threshold=1.0
seed=1234
epochs=50
model_path=model_demo.gloomy
```

```bash
./bin/gloomy "18 19 19.5 21 23 25 26 24 22 20 19 18.5" -f train.config
```

```text
average_loss=1.6865
```

```bash
ls -la model_demo.gloomy
```

```text
-rw-r--r-- 1 ... 7539 ... model_demo.gloomy
```

`model_demo.gloomy` contient désormais le réseau entraîné, les statistiques de normalisation, l'état de l'optimiseur Momentum et la mémoire hybride — le tout protégé par un checksum FNV-1a (voir [Sérialisation](serialization.md)). `*.gloomy` est ignoré par Git (`.gitignore`).

## Étape 5 — Réutiliser un modèle sauvegardé

### 5.1 Reprendre l'entraînement en ligne à partir du modèle sauvegardé

`runtime=online_learning` (et `runtime=training`) relisent `model_path` au démarrage : si le fichier existe déjà, l'exécution reprend le réseau/la normalisation/l'optimiseur/la mémoire sauvegardés au lieu de repartir de zéro. Même configuration, deux exécutions sur la suite de la séquence (le lendemain, par exemple) — la seule différence est la présence de `model_path` :

```ini
# resume.config — reprend model_demo.gloomy
runtime=online_learning
activation=tanh
hidden_layers=1
neurons=8
loss=mse
optimizer=momentum
learning_rate=0.01
momentum=0.9
memory_strategy=hybrid
memory_capacity=64
novelty_threshold=1.0
seed=1234
model_path=model_demo.gloomy
```

```bash
./bin/gloomy "18 18.5 19 20.5 22.5 24.5 25.5 23.5 21.5 19.5" -f resume.config
```

```text
0	18	18.5	-0.997173	0.000791576	0
1	18.5	19	-0.994855	0.028442	0
2	19	20.5	-0.989461	0.260814	0
3	20.5	22.5	-0.815138	0.74304	0
4	22.5	24.5	0.917035	0.648695	0
5	24.5	25.5	0.995503	0.624251	0
6	25.5	23.5	0.997325	0.536275	0
7	23.5	21.5	0.963517	0.580276	0
8	21.5	19.5	-0.510299	0.574937	0
average_loss=0.444169 memory_size=9
```

Même configuration mais **sans** `model_path` (réseau reparti de zéro), sur exactement la même suite de valeurs, pour comparaison :

```text
0	18	18.5	0	0	0
1	18.5	19	-0.339116	5.57485	0
...
average_loss=6.56063 memory_size=9
```

**`average_loss` passe de `6.56` (à froid) à `0.44` (repris)** — un facteur ~15 sur ce scénario. C'est la valeur concrète de `model_path` : le modèle repris démarre déjà adapté, au lieu de réapprendre depuis une initialisation aléatoire. `model_demo.gloomy` est réécrit à la fin de cette exécution : la relancer une troisième fois reprendrait l'état encore plus avancé.

### 5.2 Limite actuelle : inférence seule via `bin/gloomy_infer`

`bin/gloomy_infer` (voir [Quantification](quantization.md), « Runtime d'inférence minimal ») ne sait charger qu'un fichier `NetworkSerialization` (magic `GLOOMYNN`) ou `QuantizedNetworkSerialization` (magic `GLOOMYQN`) — pas le format unifié `GLOOMY_MODEL` produit par `model_path`. Le tenter donne une erreur explicite, pas un plantage silencieux :

```bash
./bin/gloomy_infer model_demo.gloomy "20.0"
```

```text
Error during inference: Invalid network file magic
```

**Aucune commande CLI ne produit aujourd'hui un fichier `NetworkSerialization`/`QuantizedNetworkSerialization` à partir d'un `GLOOMY_MODEL` entraîné** : ce chemin n'existe que dans les tests (`tests/loss_tests.cpp`), pas dans `bin/gloomy`. C'est une limite réelle, pas encore comblée — voir [feuille de route](roadmap.md). Pour l'instant, la seule façon de réutiliser un modèle entraîné en ligne de commande est celle de l'étape 5.1 (`model_path` + `online_learning`/`training`), qui reste la voie recommandée puisque c'est aussi celle qui permet de continuer l'apprentissage plutôt que de figer le modèle.

## Étape 6 — Obtenir une prédiction pour la valeur suivante, après apprentissage

On reste sur le **même exemple de température** que les étapes 4 et 5 : `model_demo.gloomy`, entraîné puis repris jusqu'à la dernière valeur réelle connue, **19.5°C**. Objectif de cette étape : demander au modèle « et après 19.5, il fait combien ? ».

### 6.1 Pourquoi `-c` ne donne pas ça

`-c` appartient à `runtime=inference` (le mode par défaut, sans `-f`). Ce mode reconstruit le réseau **avec des poids neufs et aléatoires à chaque prédiction** — il ne connaît même pas `model_path`. Lancer `-c 5` après avoir entraîné `model_demo.gloomy` donne 5 prédictions d'un réseau qui n'a jamais vu vos températures. Ce n'est pas la bonne option ici.

### 6.2 Forcer une prédiction, sur une copie du modèle

```bash
# Copie jetable de model_demo.gloomy : la commande ci-dessous va relire ET
# reecrire model_path avec une fausse valeur (voir pourquoi plus bas) ;
# on travaille sur une copie pour ne pas abimer le modele reel.
cp model_demo.gloomy model_demo_predict.gloomy
```

```ini
# predict.config — memes hyperparametres que resume.config, model_path different
runtime=online_learning
activation=tanh
hidden_layers=1
neurons=8
loss=mse
optimizer=momentum
learning_rate=0.01
momentum=0.9
memory_strategy=hybrid
memory_capacity=64
novelty_threshold=1.0
seed=1234
model_path=model_demo_predict.gloomy
```

```bash
./bin/gloomy "19.5 0" -f predict.config
```

```text
0	19.5	0	-0.98803	7.4711	0
average_loss=7.4711 memory_size=10
```

Ce que veut dire chaque colonne de cette seule ligne :

| colonne | valeur | signification |
| --- | --- | --- |
| index | `0` | un seul pas |
| observation | `19.5` | la vraie dernière température connue |
| cible | `0` | **valeur bidon** — voir pourquoi ci-dessous |
| **prédiction** | **`-0.98803`** | ce qu'on veut — mais pas encore en °C, voir 6.3 |
| perte | `7.4711` | sans intérêt : calculée à partir de la cible bidon |

Pourquoi `"19.5 0"` et pas juste `"19.5"` : le CLI exige au moins 2 valeurs pour former une paire (observation, cible). `19.5` est la vraie observation ; `0` est une valeur bidon qu'on n'utilisera jamais, juste pour compléter la paire. Preuve que ce `0` n'a aucune influence sur la prédiction : même commande, seule la valeur bidon change (à chaque fois en repartant d'une copie fraîche de `model_demo.gloomy`) :

```text
"19.5 0"    ->  prediction=-0.98803  perte=7.4711
"19.5 999"  ->  prediction=-0.98803  perte=18705.1
"19.5 -500" ->  prediction=-0.98803  perte=5243.5
```

Même prédiction (`-0.98803`) dans les 3 cas ; seule la perte change. C'est normal : le réseau calcule sa prédiction **avant** de regarder la cible ([src/OnlineLearningRuntime.cpp](../src/OnlineLearningRuntime.cpp)) — la cible ne sert qu'à la mise à jour des poids d'après, jamais au calcul de la prédiction affichée.

### 6.3 `-0.98803` n'est pas une température : la convertir en °C

Le réseau n'a jamais vu `19.5` directement. Avant d'entrer dans le réseau, chaque température est centrée-réduite par `StreamingNormalizer` :

```text
valeur_normalisee = (valeur - moyenne) / ecart_type
```

`-0.98803` est donc `19.5` **après** cette transformation — la prédiction du réseau sort dans ce même espace normalisé, jamais reconvertie par le CLI. Pour repasser en °C, on inverse la formule :

```text
valeur_en_degres = valeur_normalisee * ecart_type + moyenne
```

Il faut `moyenne` et `ecart_type`. Ce sont ceux calculés par `StreamingNormalizer` sur **toutes** les températures déjà vues, dans l'ordre où elles ont été apprises depuis l'étape 4 :

```text
18, 19, 19.5, 21, 23, 25, 26, 24, 22, 20, 19        <- etape 4 (entrainement)
18, 18.5, 19, 20.5, 22.5, 24.5, 25.5, 23.5, 21.5     <- etape 5.1 (reprise)
19.5                                                  <- observation de cette etape 6
```

21 valeurs au total, d'où :

```text
moyenne    = 21.4047619047619
ecart_type = 2.53367344655677
```

On remplace maintenant dans la formule, avec les vrais chiffres de cet exemple :

```text
valeur_en_degres = valeur_normalisee * ecart_type + moyenne
valeur_en_degres = -0.98803         * 2.53367344655677 + 21.4047619047619
valeur_en_degres = -2.50335                            + 21.4047619047619
valeur_en_degres = 18.90 °C
```

**Le modèle prédit donc `~18.90°C` pour l'heure suivant 19.5°C.** C'est cohérent avec la tendance réellement observée à l'étape 5.1 (`25.5 -> 23.5 -> 21.5 -> 19.5`, en baisse) : le modèle prédit que la baisse continue, vers `~18.9`.

Point important à ne pas oublier : **`moyenne`/`ecart_type` ne sont imprimés par aucune commande du CLI aujourd'hui.** Pour cet exemple, ils ont été obtenus en rejouant ces 21 valeurs à part, directement contre la classe C++ `StreamingNormalizer` :

```cpp
// check_norm.cpp — a compiler avec :
// g++ -std=c++17 -I./src/headers -o check_norm check_norm.cpp src/Normalization.cpp
#include "Normalization.h"
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    StreamingNormalizer normalizer(1);
    std::vector<double> temperatures = {
        18, 19, 19.5, 21, 23, 25, 26, 24, 22, 20, 19,
        18, 18.5, 19, 20.5, 22.5, 24.5, 25.5, 23.5, 21.5,
        19.5
    };
    for (double t : temperatures) {
        normalizer.update({t});
    }
    std::cout << "moyenne=" << normalizer.mean()[0] << std::endl;
    std::cout << "ecart_type=" << std::sqrt(normalizer.variance()[0]) << std::endl;
}
```

C'est une limite réelle du projet, pas un oubli de ce guide : convertir une prédiction en unité d'origine demande aujourd'hui d'écrire ce petit programme, pas juste une commande CLI — voir [feuille de route](roadmap.md).

### 6.4 Ce que ça ne donne pas : plusieurs prédictions à la suite

Ajouter deux valeurs bidon au lieu d'une (`"19.5 0 0"`) ne donne **pas** deux prédictions dans le futur : le deuxième pas utiliserait le premier `0` comme s'il s'agissait d'une vraie température, pas la prédiction du modèle pour l'heure suivante. Le CLI ne reboucle jamais une prédiction sur elle-même. Pour un vrai enchaînement (« et l'heure d'après ? », « et encore après ? »), il faudrait répéter manuellement, hors CLI : reconvertir la prédiction en °C (6.3), la rajouter à la séquence comme si c'était une vraie observation, relancer la commande — un pas à la fois, à la main. Ce n'est pas automatisé aujourd'hui — voir [feuille de route](roadmap.md), « Limites connues » (« pas de prédiction multi-pas »).

## Ce que Gloomy ne sait pas (encore) apprendre

- **Une seule série à la fois**, sortie toujours scalaire : `window_size` (défaut `1`) rend l'entrée configurable, pas la sortie. Pas d'image, pas de texte, pas de séries multivariées.
- **Pas de classification** : `softmax`/`CrossEntropyLoss` existent et sont vérifiés (voir [Fonctions d'activation](activations.md), [Fonctions de perte](losses.md)), mais aucun runtime CLI ne les exploite avec une sortie à plusieurs neurones et des cibles de classe.
- **Pas de conversion `GLOOMY_MODEL` → artefact d'inférence minimal** en ligne de commande (voir Étape 5.2).
- **Pas de reconversion de la prédiction dans l'échelle d'origine** en ligne de commande, et pas de prédiction multi-pas (voir Étape 6).

## Pour aller plus loin

Les autres clés de configuration (`optimizer`, `memory_strategy` et leurs paramètres, `seed`, `batch_size`, `train_every`...) sont documentées avec leur effet dans [Configurations et limites](configurations.md), et le détail de chaque composant (pertes, mémoires, optimiseurs, quantification, sérialisation, benchmark) dans les pages correspondantes du [sommaire](README.md).
