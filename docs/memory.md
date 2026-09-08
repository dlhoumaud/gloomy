# Mémoire d'apprentissage

## Contrat

`LearningMemory` sépare la politique de conservation du réseau neuronal et du moteur d'apprentissage. Elle manipule uniquement des `TrainingSample` composés d'un vecteur `input` et d'un vecteur `target`.

L'interface fournit :

- `add(sample)` : proposer un échantillon à la mémoire ;
- `remove(index)` : retirer un échantillon ;
- `sample(batch_size)` : produire un batch de replay ;
- `size()` et `capacity()` : observer l'occupation ;
- `clear()` : vider la mémoire.

Le réseau ne connaît pas cette abstraction et la mémoire ne connaît pas les détails des couches.

## FIFO / fenêtre glissante

`FIFOMemory` possède une capacité fixe en nombre d'échantillons. Lorsqu'elle est pleine, l'ajout d'un nouvel échantillon supprime le plus ancien.

Exemple avec une capacité de `3` :

```text
ajout : A B C  -> mémoire : A B C
ajout : D      -> mémoire : B C D
```

`sample(n)` renvoie au maximum `n` éléments dans l'ordre de la mémoire, en commençant par le plus ancien restant. Ce choix est déterministe et pratique pour les tests, mais ne constitue pas encore un replay aléatoire.

## Reservoir Sampling

`ReservoirMemory` conserve un échantillon de taille fixe d'un flux dont la longueur totale peut être inconnue. Les premiers éléments remplissent le réservoir. Pour chaque élément suivant, l'algorithme tire une position aléatoire dans le préfixe déjà observé ; si cette position appartient au réservoir, l'ancien élément est remplacé.

Après `T` observations et une capacité `K`, chaque observation a une probabilité théorique proche de `K / T` d'être conservée. La mémoire ne stocke jamais plus de `K` échantillons. `sample(n)` tire ensuite jusqu'à `n` éléments sans remise.

La seed du générateur est configurable afin de rendre les expériences reproductibles. Le coût d'ajout est constant en moyenne et le stockage reste borné, mais le générateur aléatoire ajoute un petit coût CPU par observation.

## Choix initial

- FIFO est préférable si les données récentes sont les plus pertinentes et si un comportement totalement déterministe est recherché.
- Reservoir est préférable si le flux est long ou de durée inconnue et si l'on veut conserver une représentation historique globale.

Ces deux stratégies peuvent maintenant être transmises au même `LearningEngine` sans modifier le réseau, la perte ou l'optimiseur.

## Limites actuelles

La capacité est limitée en nombre de `TrainingSample`, pas en octets. Les vecteurs `input` et `target` utilisent actuellement des `double` et leurs tailles peuvent varier ; le budget RAM réel n'est donc pas encore strictement contrôlé.

`QuantizedFIFOMemory` constitue une exception expérimentale : ses vecteurs `input` et `target` sont stockés en `int16` après calibration et la classe expose `memoryUsedBytes()`. Les autres stratégies utilisent encore leur représentation `double` native.

`QuantizedInt8FIFOMemory` fournit maintenant la variante int8, avec la même politique FIFO et le même contrat de replay. Elle réduit davantage les octets par valeur, au prix d'une erreur de quantification potentiellement supérieure.

La mémoire FIFO favorise les données récentes et peut provoquer du catastrophic forgetting. Reservoir limite ce biais temporel, mais ne tient pas encore compte de l'erreur, de la nouveauté ou des régimes de données.

Le `LearningEngine` peut maintenant appeler `trainFromMemory(memory, batch_size)` pour sélectionner un batch avec `memory.sample()` puis le transmettre à `trainBatch()`. La méthode `learn(memory, sample, batch_size)` ajoute une observation et déclenche immédiatement un replay.

Cette première intégration entraîne après chaque observation lorsqu'elle est utilisée avec `learn()`. Un `TrainingScheduler` peut maintenant différer l'entraînement pour réduire le coût CPU.

## Prioritized Replay

`PrioritizedMemory` utilise le champ `TrainingSample::priority`. Lorsqu'elle est pleine, elle remplace l'échantillon de plus faible priorité si le nouvel échantillon est plus important. Lors d'un replay, la probabilité de sélection est proportionnelle à `priority^alpha`, avec une petite valeur minimale pour conserver une exploration des priorités nulles.

`alpha = 0` rend la sélection uniforme. Une valeur plus élevée renforce la préférence pour les exemples prioritaires. La valeur par défaut est `0.6`.

La méthode `LearningEngine::learn()` calcule automatiquement une priorité égale à l'erreur absolue maximale entre la prédiction et la cible. Les appels directs à `LearningMemory::add()` peuvent définir eux-mêmes la priorité dans `TrainingSample`.

La priorité est recalculée après chaque replay indexé.

### Correction de biais d'échantillonnage (importance sampling)

Un replay prioritisé sur-échantillonne les observations à forte priorité : sans correction, la mise à jour des poids est biaisée vers elles. `PrioritizedMemory` calcule maintenant, pour chaque échantillon tiré par `sampleIndexed()`, un poids d'importance-sampling qui corrige ce biais :

```text
P(i) = priority_i^alpha / somme_j(priority_j^alpha)   (sur toute la mémoire, au moment du tirage)
w_i  = (N * P(i))^(-beta)
w_i  = w_i / max_du_batch(w_i)                          (le plus grand poids du batch vaut 1)
```

`beta` (nouveau paramètre du constructeur, défaut `0.4`) contrôle l'intensité de la correction : `beta = 0` la désactive complètement (tous les poids valent `1.0`, comportement identique à avant ce changement) ; `beta = 1` corrige entièrement le biais. `LearningEngine::trainFromMemory` applique ce poids à la contribution de chaque échantillon au gradient avant la mise à jour des poids (`MemoryEntry::importance_weight`, `1.0` par défaut et donc neutre pour les autres stratégies).

```cpp
PrioritizedMemory memory(256, /*alpha=*/0.6, /*seed=*/5489u, /*beta=*/0.4);
```

Cette version calcule `P(i)` sur l'ensemble de la mémoire au moment du tirage (avant tout retrait pour l'échantillonnage sans remise), une approximation standard qui ignore la légère dépendance entre tirages successifs d'un même batch. L'annealing de `beta` au fil de l'entraînement (souvent de `0.4` vers `1.0` dans la littérature) n'est pas implémenté : `beta` reste une valeur fixe configurée à la construction.

## Novelty Memory

`NoveltyMemory` compare chaque nouvelle entrée aux entrées déjà conservées avec une distance euclidienne au carré. Le paramètre `novelty_threshold` est donc comparé à :

```text
distance2(x, y) = somme((x_i - y_i)^2)
```

Une observation dont la distance minimale est inférieure au seuil est considérée comme redondante et ignorée. Les observations suffisamment différentes sont ajoutées jusqu'à la capacité. Lorsque la mémoire est pleine, le représentant ayant le voisin le plus proche est remplacé, afin de réduire la redondance locale.

Les entrées doivent avoir une dimension constante et ne doivent pas être vides. L'insertion examine les `K` éléments conservés ; la politique est donc simple et adaptée aux petites mémoires, mais elle coûte `O(K)` par observation. Le remplacement complet recherche en plus le représentant le plus redondant.

Novelty Memory réduit les séries d'observations presque identiques, mais elle ne tient pas encore compte de l'erreur de prédiction, de la récence ou de la diversité des cibles. Une combinaison avec Prioritized Replay ou une mémoire hybride sera étudiée ultérieurement.

## Mémoire hybride

`HybridMemory` divise une capacité globale entre quatre partitions configurables :

```cpp
HybridMemoryRatios ratios{
	0.25, // recent
	0.25, // error
	0.25, // novelty
	0.25  // historical
};
HybridMemory memory(256, ratios, novelty_threshold);
```

Les ratios sont normalisés automatiquement et les arrondis sont attribués à la partition récente. La capacité totale reste donc toujours égale à `256` au maximum.

La politique de routage est déterministe : une priorité supérieure à `0.5` dirige l'observation vers `error` ; une observation suffisamment éloignée dirige l'observation vers `novelty` ; toute autre observation (générique) rejoint directement `recent`, sans alternance — elle est par définition la plus récente au moment de son arrivée. Chaque observation n'est stockée que dans une partition, afin d'éviter les duplications.

### Vraie récence fondée sur l'âge

`recent` se comporte comme une véritable fenêtre glissante : lorsqu'elle est pleine, elle évince son membre le plus ancien **par `TrainingSample::age` réel** (mis à jour par `advanceAges()`), pas par une simple position dans le vecteur interne ni par une alternance à l'admission. L'élément évincé n'est pas perdu : il est **promu** dans `historical` (en évinçant lui-même, si besoin, un élément choisi aléatoirement dans `historical`). `historical` se peuple donc par vieillissement réel des observations qui ont transité par `recent`, plutôt que par une seconde file alimentée par une alternance `seen_samples % 2` déconnectée de l'âge — c'était la limite explicitement documentée dans une version antérieure de ce fichier. Sans partition `historical` configurée (`historical_ratio = 0`), l'élément évincé de `recent` est simplement perdu, comme avant.

`error` et `novelty` évincent elles aussi leur membre le plus ancien par âge réel lorsqu'elles sont pleines (même correctif : une position de vecteur ne reflète plus l'âge réel une fois qu'un remplacement en place a eu lieu). Le replay final mélange les partitions sans remise.

Reste ouvert : `age` est incrémenté par cycle de replay (`advanceAges()`), pas par horloge murale ; plusieurs observations ajoutées sans replay entre elles peuvent donc partager le même âge (égalité départagée de façon stable mais arbitraire). Un score d'importance normalisé combinant réellement recency/error/novelty pour le routage (plutôt que ce jeu de règles explicites) reste une politique hybride adaptative future (voir [feuille de route](roadmap.md)).

## Score d'importance

`ImportanceScorer` combine cinq composantes indépendantes, chacune bornée dans `[0, 1]` :

- `error` : erreur de prédiction ;
- `novelty` : différence avec les observations conservées ;
- `rarity` : rareté estimée d'un motif ;
- `recency` : caractère récent de l'observation ;
- `diversity` : contribution à la diversité globale.

Le score est une moyenne pondérée. Un poids égal à zéro désactive une composante, ce qui permet de comparer `error-only`, `novelty-only` ou des combinaisons sans changer la mémoire.

`TrainingSample` conserve maintenant `priority`, `error`, `novelty`, `rarity`, `recency`, `diversity`, `age` et `usage_count`. Pour l'instant, `LearningEngine::learn()` renseigne automatiquement l'erreur et calcule une priorité `error-only` normalisée. À chaque appel de replay, `advanceAges()` augmente l'âge de tous les échantillons et la mémoire incrémente `usage_count` pour les éléments sélectionnés.

Le replay indexé fournit maintenant l'indice mémoire avec chaque copie. `LearningEngine::trainFromMemory()` recalcule donc l'erreur après la mise à jour des poids et réécrit précisément l'échantillon concerné avec sa nouvelle priorité. Cela fonctionne même si deux observations ont des valeurs identiques, car leurs indices sont distincts.

## Persistance

`LearningMemorySerialization::save`/`load` sauvegarde et restaure une mémoire complète — FIFO, Reservoir, Prioritized, Novelty ou Hybrid — dans un fichier binaire versionné et protégé par un checksum FNV-1a, comme le fichier réseau et le fichier optimiseur :

```cpp
LearningMemorySerialization::save("memory.bin", memory);
std::unique_ptr<LearningMemory> restored = LearningMemorySerialization::load("memory.bin");
```

`load()` reconstruit le type concret à partir du fichier avec :

- la capacité et les hyperparamètres propres à la stratégie (`alpha` pour Prioritized, `novelty_threshold` pour Novelty et Hybrid, les ratios pour Hybrid) ;
- tous les échantillons stockés avec leurs métadonnées complètes (`priority`, `error`, `novelty`, `rarity`, `recency`, `diversity`, `age`, `usage_count`) ;
- la partition (`recent`/`error`/`novelty`/`historical`) de chaque échantillon pour Hybrid ;
- l'état complet du générateur `std::mt19937` (pas seulement la seed) et le compteur d'observations vues (`seen_samples`) pour Reservoir, Prioritized et Hybrid, afin que les tirages futurs restent reproductibles après un redémarrage ;
- une vérification que le nombre d'échantillons ne dépasse pas la capacité déclarée.

Un fichier tronqué, corrompu, de version incompatible ou dont le nombre d'échantillons dépasse la capacité est rejeté par une exception plutôt que de restaurer un état invalide.

Cette persistance reste un fichier séparé du réseau, de la normalisation et de l'optimiseur. Les mémoires quantifiées (`QuantizedFIFOMemory`, `QuantizedInt8FIFOMemory`) ne sont pas encore couvertes ; leur persistance nécessitera de réutiliser les codecs `TrainingSampleQuantization`/`Int8TrainingSampleQuantization` avec les paramètres `scale`/`zero_point`. Voir la [feuille de route](roadmap.md).

## Training Scheduler

Le `TrainingScheduler` décide si une observation doit déclencher un replay :

- `EverySampleScheduler` entraîne après chaque observation ;
- `EveryNScheduler(N)` entraîne tous les `N` échantillons ;
- `OnHighErrorScheduler(seuil)` entraîne seulement lorsque l'erreur atteint le seuil.

Exemple :

```cpp
EveryNScheduler scheduler(10);
engine.learn(memory, sample, 8, scheduler);
```

L'observation est conservée même lorsque l'entraînement est différé. Le scheduler réduit donc les mises à jour, mais ne réduit pas à lui seul la taille de la mémoire. Le nombre d'updates et le coût CPU devront être mesurés dans les futurs benchmarks.
