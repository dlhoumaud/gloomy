# Quantification

## Première implémentation : int16 affine

`Int16Quantizer` stocke une valeur réelle avec un entier signé 16 bits et deux paramètres :

```text
q = round(real / scale + zero_point)
real_approx = (q - zero_point) * scale
```

La valeur quantifiée est saturée dans `[-32768, 32767]`. La calibration calcule `scale` à partir du minimum et du maximum du vecteur fourni, **étendus pour toujours inclure 0** (voir « Bug corrigé : calibration hors zéro » ci-dessous). Le `zero_point` est ensuite ajusté pour couvrir cette plage autant que possible.

```cpp
const QuantizationParameters parameters =
    Int16Quantizer::calibrate(values);
const QuantizedVector encoded =
    Int16Quantizer::quantize(values, parameters);
const std::vector<double> restored =
    Int16Quantizer::dequantize(encoded);
```

## Règles de calibration

Les valeurs de calibration doivent représenter le flux d'entraînement. Il ne faut pas utiliser les données de test pour choisir `scale` ou `zero_point`, sinon la mesure de précision serait optimiste.

Les vecteurs vides et les valeurs non finies sont refusés. Un vecteur constant reçoit une échelle adaptée à sa magnitude afin de rester représentable.

## Bug corrigé : calibration hors zéro

`Int16Quantizer::calibrate`/`Int8Quantizer::calibrate` calculaient `zero_point = qmin - min(values)/scale`, puis saturaient le résultat dans `[qmin, qmax]` s'il en sortait. Pour des valeurs qui ne contiennent pas `0` dans leur plage (un capteur toujours positif, par exemple `17.0 20.0 23.0 26.0`, loin de `0`), ce calcul produisait systématiquement un `zero_point` hors plage, silencieusement saturé — et la saturation qui suit dans `quantize()` écrasait alors **toutes** les valeurs vers le même code quantifié (`qmax`), une perte totale d'information sans qu'aucune exception ne soit levée. Mesuré avant correction sur `{17.0, 20.0, 23.0, 26.0}` : erreur de reconstruction ≈ `17.7` (la moitié de la plage), contre une plage de valeurs de seulement `9.0`.

Ce bug touchait uniquement les données dont la plage ne contient pas `0` et n'est pas symétrique autour de `0` — c'est pourquoi il n'avait jamais été détecté : toutes les données de test existantes (poids de réseaux entraînés, à peu près centrés sur `0` par construction ; échantillons de test `{-10.0, -1.5, 0.0, 2.5, 10.0}` ou `{0.0, 1.0, 10.0, 11.0}`, qui contiennent déjà `0` ou sont symétriques) évitaient par coïncidence ce cas. Découvert en mesurant `DeltaQuantizer` (voir « Compression différentielle » ci-dessous) contre une quantification directe sur une série de type capteur, avec un offset réaliste.

Corrigé en étendant systématiquement la plage de calibration pour qu'elle contienne toujours `0` (`effective_min = min(min(values), 0)`, `effective_max = max(max(values), 0)`) avant de calculer `scale`/`zero_point` : cela garantit mathématiquement que `zero_point` reste dans `[qmin, qmax]` sans jamais avoir besoin d'être saturé, quelle que soit la plage réelle des données. Le prix est un `scale` parfois plus grossier que l'idéal (la plage effective peut être plus large que la plage réelle des données, si celle-ci ne contient pas `0`), mais la perte d'information totale disparaît. Vérifié : les tests existants passent sans modification (les cas déjà corrects restent corrects, souvent avec une précision légèrement meilleure) et deux nouveaux cas de régression (`{17.0, 20.0, 23.0, 26.0}`, int16 et int8) confirment que les valeurs restent désormais distinguables après un aller-retour.

## Coût et limites

Le stockage des valeurs passe de 8 octets par `double` à 2 octets par valeur, hors paramètres de quantification. Cette première version quantifie des vecteurs indépendants ; les poids du réseau sont désormais quantifiables séparément (voir « Quantification des poids » plus bas), mais pas encore les `TrainingSample` complets stockés par les mémoires natives (float64/int16/int8 restent des représentations distinctes, pas une conversion à la volée) ni les buffers internes du réseau pendant l'entraînement (les poids restent en float64 pendant l'apprentissage ; seul un réseau déjà entraîné se quantifie).

La quantification introduit une erreur d'arrondi et peut saturer les valeurs hors de la plage de calibration. Il faut donc mesurer l'impact sur MAE/RMSE avant de l'utiliser pour l'entraînement online. L'étape recommandée est de conserver les poids float32 et de quantifier d'abord la mémoire d'apprentissage.

## TrainingSample

`TrainingSampleQuantizer` applique cette représentation séparément aux vecteurs `input` et `target` :

```cpp
const TrainingSampleQuantizer quantizer =
    TrainingSampleQuantizer::calibrate(training_samples);
const QuantizedTrainingSample compact = quantizer.encode(sample);
const TrainingSample restored = quantizer.decode(compact);
```

Les paramètres de quantification sont calibrés sur l'ensemble des échantillons fourni, puis réutilisés pour les encodages suivants. Les métadonnées d'apprentissage (`priority`, `error`, `age`, `usage_count`, etc.) sont conservées en précision native dans cette première version.

Ce codec est prêt à être utilisé par une mémoire compacte, mais les stratégies `FIFO`, `Reservoir`, `Prioritized`, `Novelty` et `Hybrid` stockent encore des `double` directement. Leur remplacement par un stockage quantifié sera une étape distincte, nécessaire pour garantir un vrai `memory_budget_bytes`.

## Quantification int8

`Int8Quantizer` applique la même représentation affine avec une plage `[-128, 127]`. À nombre de valeurs égal, le stockage brut passe à 1 octet par valeur, mais le pas `scale` est plus grand qu'en int16 et l'erreur de reconstruction peut donc augmenter.

Les résultats int8 doivent être comparés sur les mêmes valeurs de calibration que int16, avec MAE et RMSE mesurées après déquantification. Le codec int8 est disponible pour les vecteurs ; il n'est pas encore utilisé par une mémoire compacte dédiée.

## Mémoire FIFO quantifiée

`QuantizedFIFOMemory` utilise le codec int16 pour stocker `input` et `target` tout en conservant les métadonnées d'apprentissage. Elle expose :

- `bytesPerSample()` : estimation de la taille d'un échantillon ;
- `memoryUsedBytes()` : occupation des échantillons actuellement stockés ;
- `capacity()` : nombre maximal d'échantillons.

Les paramètres `scale` et `zero_point` ne sont pas dupliqués par échantillon. Ils doivent être conservés avec le modèle ou l'état de la mémoire lors de la future sérialisation. Cette mémoire constitue la première base concrète pour tester des budgets comme `4 KB`, `8 KB` ou `16 KB`.

## Mémoire FIFO int8

`QuantizedInt8FIFOMemory` applique le même principe avec 1 octet par valeur quantifiée. Elle expose également `bytesPerSample()` et `memoryUsedBytes()`, ce qui permet de comparer directement les budgets int16 et int8 pour une même capacité.

Le gain mémoire concerne uniquement les vecteurs `input` et `target` ; les métadonnées restent en précision native. La perte de précision doit être mesurée après déquantification avant de choisir int8 pour un entraînement online.

## Compression différentielle

`DeltaQuantizer` (`src/headers/DeltaQuantization.h`) compresse une série temporelle en quantifiant, non pas les valeurs brutes, mais la différence entre valeurs consécutives (« delta ») :

```cpp
const DeltaQuantizedSeries encoded = DeltaQuantizer::encode(values);
const std::vector<double> restored = DeltaQuantizer::decode(encoded);
```

`encode()` conserve la première valeur exactement (`first_value`, un `double`, l'ancre de la reconstruction) et quantifie en int16 (`Int16Quantizer`) le reste comme une suite de deltas. `decode()` reconstruit par sommes cumulées : `valeur[i] = valeur[i-1] + delta_dequantifié[i-1]`.

L'intérêt : pour une série qui varie lentement (forte corrélation temporelle), les deltas ont une plage bien plus étroite que les valeurs elles-mêmes, donc un `scale` plus fin pour le même nombre de bits — une meilleure précision de reconstruction, à budget mémoire égal.

### Coût et limites — un compromis réel, pas un gain systématique

Mesuré sur trois scénarios (100 points sauf indication contraire) :

| Scénario | Erreur max delta | Erreur max quantification directe |
| --- | --- | --- |
| Dérive lente type capteur (offset 20, amplitude 3, dérive +0.01/pas) | `0.0000114` | `0.000177` (~15×) |
| Rampe parfaitement linéaire (offset 1000, pente 0.5, deltas constants) | `~0` | `0.00745` |
| Série bruitée/erratique (10 points, sauts irréguliers) | `0.000458` | `0.000153` |

Les deux premiers scénarios montrent le gain attendu : quand les deltas sont réellement plus « serrés » que les valeurs brutes, la compression différentielle réduit nettement l'erreur de reconstruction (cas extrême de la rampe : deltas exactement constants, erreur quasi nulle). Le troisième scénario montre l'inverse : sur une série bruitée où les sauts d'un pas à l'autre sont aussi grands que la série elle-même, les deltas n'ont **pas** une plage plus étroite, et la quantification directe fait légèrement mieux — sans compter que la reconstruction par sommes cumulées **accumule** l'erreur de chaque delta le long de la série (contrairement à la quantification directe, où l'erreur de chaque valeur reste indépendante des autres). La compression différentielle doit donc être mesurée sur les données réellement visées avant d'être choisie : ce n'est pas un gain automatique, seulement pour les séries à forte corrélation temporelle.

`quantizedBytes()` calcule la taille réelle d'une série encodée (`sizeof(double)` pour `first_value`, un `int16_t` par delta, plus les paramètres de calibration). Une série à une seule valeur (aucun delta à encoder) et une série vide (rejetée) sont gérées explicitement.

## Quantification des poids d'un réseau entraîné

`NetworkQuantization` applique `Int8Quantizer` aux poids et biais d'un `NeuralNetwork` déjà entraîné, couche par couche :

```cpp
const QuantizedNetwork compact = NetworkQuantization::quantize(network);
const NeuralNetwork restored = NetworkQuantization::dequantize(compact);
```

Pour chaque `DenseLayer`, les poids (aplatis en un seul vecteur, ligne par ligne) et les biais sont calibrés et quantifiés **séparément** (deux appels à `Int8Quantizer::calibrate`/`quantize`), parce que leurs plages de magnitude diffèrent généralement. `dequantize()` reconstruit un réseau float64 complet via `addLayer`, et rejette (`std::invalid_argument`) toute incohérence de dimensions entre couches déclarées et données quantifiées. `quantizedBytes()` calcule la taille réelle d'un `QuantizedNetwork` en mémoire (1 octet par valeur + `sizeof(QuantizationParameters)` par vecteur quantifié).

Cette quantification est **post-entraînement uniquement** : le réseau continue de s'entraîner en float64 ; on ne quantifie qu'une copie destinée à l'inférence ou au stockage.

### Impact mesuré

Sur un réseau 1-8-1 entraîné (seed `4242u`, 80 époques, SGD lr=0.01, tâche y=2x+1), comparé sur x ∈ [0, 10] par pas de 0.5 :

- erreur absolue maximale après déquantification : environ `0.049` ;
- erreur relative maximale : environ `0.23 %` ;
- ratio mémoire réel : environ **2.25×** (poids+biais quantifiés vs float64), très en-deçà du ratio théorique de 8× (1 octet vs 8 octets par valeur).

Ce dernier chiffre n'est pas une anomalie : sur un réseau aussi petit, le coût fixe de calibration par vecteur (`scale` + `zero_point`, soit `sizeof(QuantizationParameters)` ≈ 16 octets, répété pour chacun des 4 vecteurs — poids et biais de 2 couches) domine le gain apporté par la quantification elle-même. Le ratio théorique de 8× ne s'approche que pour des couches avec beaucoup plus de poids par vecteur quantifié (le coût fixe est alors amorti sur plus de valeurs). Ce compromis doit être mesuré à nouveau pour toute architecture réellement visée avant de choisir la quantification comme stratégie de compression.

### Artefact compact : `QuantizedNetworkSerialization`

`QuantizedNetworkSerialization` sérialise un `QuantizedNetwork` dans un format binaire dédié, magic `GLOOMYQN`, suivant le même schéma de robustesse que les autres sérialiseurs du projet (version de format, validation des dimensions et du nombre de couches, somme de contrôle FNV-1a en fin de fichier, rejet propre — `std::runtime_error` — en cas de fichier tronqué, corrompu ou de version non supportée) :

```cpp
QuantizedNetworkSerialization::save(path, compact);
const QuantizedNetwork loaded = QuantizedNetworkSerialization::load(path);
```

Ce fichier ne contient que les poids/biais quantifiés et leurs paramètres de calibration — pas de métadonnées d'entraînement (optimiseur, mémoire d'apprentissage, historique). C'est délibérément un artefact minimal pensé pour le déploiement, distinct du format unifié `GLOOMY_MODEL` (`ModelSerialization`) qui, lui, embarque tout l'état nécessaire pour reprendre l'entraînement.

## Runtime d'inférence minimal : `bin/gloomy_infer`

`src/InferenceOnlyMain.cpp` fournit un exécutable CLI qui ne dépend que de `DenseLayer`, `NeuralNetwork`, `NetworkSerialization`, `Quantization`, `Int8Quantization`, `NetworkQuantization` et `QuantizedNetworkSerialization` — explicitement pas de `LearningEngine`, `Optimizer`, `LearningMemory` ni `GloomyConfig`. C'est le point de départ pour une cible embarquée : uniquement le chemin de calcul nécessaire pour transformer une entrée en prédiction.

```text
Usage: bin/gloomy_infer <model_path> "<sequence_values>" [--quantized]
```

- sans `--quantized` : charge un fichier `NetworkSerialization` (magic `GLOOMYNN`, float64) ;
- avec `--quantized` : charge un fichier `QuantizedNetworkSerialization` (magic `GLOOMYQN`, int8) puis le déquantifie en mémoire avant de lancer `forward()`.

La cible `make infer` compile ce binaire en une seule invocation `g++` (comme `make test`/`make benchmark`), donc indépendamment de la règle incrémentale `%.o` et de son suivi de dépendances.

### Empreinte mesurée

Comparaison avec le CLI complet `bin/gloomy` (mêmes options de compilation, `-O2`), sur cette machine :

| Binaire | Taille brute | Taille `strip`ée |
| --- | --- | --- |
| `bin/gloomy` | 379 648 octets | 317 752 octets |
| `bin/gloomy_infer` | 97 656 octets | 80 184 octets |

Soit environ 74–79 % de réduction, portée en grande partie par le segment `.text` (302 429 → 69 169 octets) : l'exécutable n'embarque plus le code d'entraînement, de gestion de mémoire d'apprentissage ni de configuration CLI.

Ce runtime a été vérifié de bout en bout (entraînement → export float64 et int8 → chargement et inférence réels via le binaire compilé) : les deux formats produisent une prédiction cohérente avec la tâche apprise, et l'erreur induite par la quantification reste dans l'ordre de grandeur mesuré ci-dessus. Une erreur de fichier manquant est également gérée proprement (message explicite, code de sortie 1).

### Ce qui reste hors de portée de cette étape

Ce runtime minimal réduit la taille du binaire et le nombre de dépendances, mais il ne règle pas à lui seul le déploiement embarqué complet. Restent notamment, non traités ici faute de matériel/outillage disponible dans cet environnement :

- la compilation et l'exécution sur une cible embarquée réelle (aucune chaîne de compilation croisée — `arm-none-eabi-gcc`, `arm-linux-gnueabihf-gcc`, `avr-gcc`, `riscv64-unknown-elf-gcc` — n'est installée ici) ;
- la mesure de RAM, Flash, CPU et énergie sur un tel matériel ;
- le remplacement des allocations dynamiques du chemin d'inférence (`std::vector` dans `NeuralNetwork`/`DenseLayer`) par des buffers contigus ou un arena allocator à capacité statique — un changement plus invasif, qui toucherait aussi `Optimizer`, les sérialiseurs et `BenchmarkRunner`, et qui n'a pas été fait ici pour rester une modification ciblée et à faible risque.

La compression différentielle des séries temporelles (`DeltaQuantizer`), une piste distincte de la quantification des poids, est désormais faite — voir « Compression différentielle » ci-dessus.
