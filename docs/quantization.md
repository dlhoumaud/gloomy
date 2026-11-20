# Quantification

## Première implémentation : int16 affine

`Int16Quantizer` stocke une valeur réelle avec un entier signé 16 bits et deux paramètres :

```text
q = round(real / scale + zero_point)
real_approx = (q - zero_point) * scale
```

La valeur quantifiée est saturée dans `[-32768, 32767]`. La calibration calcule `scale` à partir du minimum et du maximum du vecteur fourni. Le `zero_point` est ensuite ajusté pour couvrir cette plage autant que possible.

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

Les vecteurs vides et les valeurs non finies sont refusés. Un vecteur constant reçoit une échelle minimale adaptée à sa magnitude afin de rester représentable.

## Coût et limites

Le stockage des valeurs passe de 8 octets par `double` à 2 octets par valeur, hors paramètres de quantification. Le quantizer encode des vecteurs; `TrainingSampleQuantizer` encode ensemble les vecteurs `input` et `target` et conserve les métadonnées en précision native. Les poids et buffers internes du réseau ne sont pas quantifiés.

La quantification introduit une erreur d'arrondi et peut saturer les valeurs hors de la plage de calibration. Il faut mesurer son impact sur MAE/RMSE avant de l'utiliser pour l'entraînement online. Les poids restent actuellement en float64; les quantifier est un travail distinct.

## TrainingSample

`TrainingSampleQuantizer` applique cette représentation séparément aux vecteurs `input` et `target` :

```cpp
const TrainingSampleQuantizer quantizer =
    TrainingSampleQuantizer::calibrate(training_samples);
const QuantizedTrainingSample compact = quantizer.encode(sample);
const TrainingSample restored = quantizer.decode(compact);
```

Les paramètres de quantification sont calibrés sur l'ensemble des échantillons fourni, puis réutilisés pour les encodages suivants. Les métadonnées d'apprentissage (`priority`, `error`, `age`, `usage_count`, etc.) sont conservées en précision native dans cette première version.

Ce codec est utilisé par `QuantizedFIFOMemory` (int16) et `QuantizedInt8FIFOMemory` (int8). Les stratégies Reservoir, Prioritized, Novelty et Hybrid stockent encore leurs vecteurs en `double`. Les métadonnées de toutes les mémoires restent en précision native.

## Quantification int8

`Int8Quantizer` applique la même représentation affine avec une plage `[-128, 127]`. À nombre de valeurs égal, le stockage brut passe à 1 octet par valeur, mais le pas `scale` est plus grand qu'en int16 et l'erreur de reconstruction peut donc augmenter.

Les résultats int8 doivent être comparés sur les mêmes valeurs de calibration que int16, avec MAE et RMSE mesurées après déquantification. Le codec int8 est utilisé par la mémoire FIFO int8; les autres stratégies n'ont pas de stockage compact int8.

## Mémoire FIFO quantifiée

`QuantizedFIFOMemory` utilise le codec int16 pour stocker `input` et `target` tout en conservant les métadonnées d'apprentissage. Elle expose :

- `bytesPerSample()` : estimation de la taille d'un échantillon ;
- `memoryUsedBytes()` : occupation des échantillons actuellement stockés ;
- `capacity()` : nombre maximal d'échantillons.

Les paramètres `scale` et `zero_point` ne sont pas dupliqués par échantillon. Ils sont conservés dans la sérialisation de la mémoire incluse dans `GLOOMY_MODEL`. `memoryUsedBytes()` estime l'espace des échantillons encodés et leurs métadonnées; il ne représente pas le pic RSS ni l'ensemble des allocations du processus.

## Mémoire FIFO int8

`QuantizedInt8FIFOMemory` applique le même principe avec 1 octet par valeur quantifiée. Elle expose également `bytesPerSample()` et `memoryUsedBytes()`, ce qui permet de comparer directement les budgets int16 et int8 pour une même capacité.

Le gain mémoire concerne uniquement les valeurs des vecteurs `input` et `target`; les métadonnées restent en précision native. La perte de précision doit être mesurée après déquantification avant de choisir int8 pour un entraînement online. Dans le runtime configuré, les précisions int16/int8 sont réservées à la stratégie FIFO.
