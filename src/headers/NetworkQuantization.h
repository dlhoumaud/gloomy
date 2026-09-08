#ifndef NETWORK_QUANTIZATION_H
#define NETWORK_QUANTIZATION_H

#include "Int8Quantization.h"
#include "NeuralNetwork.h"
#include <cstddef>
#include <string>
#include <vector>

// Une couche dense quantifiee en int8 : poids et biais quantifies
// separement (une calibration echelle/zero_point propre a chacun, leurs
// distributions typiques different souvent), plus les dimensions
// necessaires pour reconstruire la couche flottante.
struct QuantizedDenseLayer {
    std::size_t input_size = 0;
    std::size_t output_size = 0;
    // Aplati ligne par ligne : weights.values[input_index * output_size + output_index].
    Int8QuantizedVector weights;
    Int8QuantizedVector bias;
};

struct QuantizedNetwork {
    std::string algorithm = "none";
    std::string post_algorithm = "none";
    std::vector<QuantizedDenseLayer> layers;
};

// Quantification des poids d'un reseau entraine, pour un artefact
// d'inference compact (voir docs/quantization.md et docs/roadmap.md,
// « Priorité moyenne : compression et embarqué »). N'affecte pas
// l'entrainement : un reseau se quantifie une fois entraine, se dequantifie
// pour l'inference via NeuralNetwork::forward() habituel.
class NetworkQuantization {
public:
    static QuantizedNetwork quantize(const NeuralNetwork& network);

    // Reconstruit un reseau flottant (float64) pret pour l'inference a
    // partir de sa version quantifiee.
    static NeuralNetwork dequantize(const QuantizedNetwork& quantized);

    // Octets occupes par les poids et biais quantifies (int8 + parametres
    // de calibration par couche), sans compter l'architecture elle-meme.
    static std::size_t quantizedBytes(const QuantizedNetwork& quantized);
};

#endif // NETWORK_QUANTIZATION_H
