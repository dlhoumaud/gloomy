#ifndef QUANTIZED_NETWORK_SERIALIZATION_H
#define QUANTIZED_NETWORK_SERIALIZATION_H

#include "NetworkQuantization.h"
#include <string>

// Persiste un reseau quantifie en int8 : l'artefact minimal destine a
// l'inference embarquee (voir docs/quantization.md et docs/roadmap.md,
// « Priorité moyenne : compression et embarqué »). Poids et biais tiennent
// sur 1 octet au lieu de 8 (float64), et le fichier ne contient aucune
// metadonnee d'entrainement (pas de normalisation, d'optimiseur ni de
// memoire d'apprentissage), contrairement au format GLOOMY_MODEL unifie.
class QuantizedNetworkSerialization {
public:
    static void save(const std::string& path, const QuantizedNetwork& network);
    static QuantizedNetwork load(const std::string& path);
};

#endif // QUANTIZED_NETWORK_SERIALIZATION_H
