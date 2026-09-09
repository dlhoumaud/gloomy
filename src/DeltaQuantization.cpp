#include "headers/DeltaQuantization.h"
#include <stdexcept>

DeltaQuantizedSeries DeltaQuantizer::encode(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Cannot encode an empty time series");
    }

    DeltaQuantizedSeries result;
    result.first_value = values.front();

    if (values.size() == 1) {
        // Rien a differencier : deltas reste un QuantizedVector vide.
        // Int16Quantizer::calibrate rejette un vecteur vide, donc les
        // parametres ci-dessous ne sont que des valeurs neutres valides,
        // jamais utilisees par decode() (aucun delta a dequantifier).
        result.deltas.parameters = {1.0, 0};
        return result;
    }

    std::vector<double> raw_deltas;
    raw_deltas.reserve(values.size() - 1);
    for (std::size_t index = 1; index < values.size(); ++index) {
        raw_deltas.push_back(values[index] - values[index - 1]);
    }

    const QuantizationParameters parameters = Int16Quantizer::calibrate(raw_deltas);
    result.deltas = Int16Quantizer::quantize(raw_deltas, parameters);
    return result;
}

std::vector<double> DeltaQuantizer::decode(const DeltaQuantizedSeries& encoded) {
    const std::vector<double> deltas = Int16Quantizer::dequantize(encoded.deltas);
    std::vector<double> result;
    result.reserve(deltas.size() + 1);
    result.push_back(encoded.first_value);
    // Somme cumulee : chaque valeur reconstruite depend de la precedente,
    // donc l'erreur de quantification de chaque delta s'accumule le long de
    // la serie plutot que de rester bornee independamment pour chaque
    // valeur (contrairement a une quantification directe des valeurs
    // brutes) — voir docs/quantization.md, « Compression differentielle »,
    // « Coût et limites ».
    double running_value = encoded.first_value;
    for (double delta : deltas) {
        running_value += delta;
        result.push_back(running_value);
    }
    return result;
}

std::size_t DeltaQuantizer::quantizedBytes(const DeltaQuantizedSeries& encoded) {
    return sizeof(double) +
        encoded.deltas.values.size() * sizeof(std::int16_t) +
        sizeof(QuantizationParameters);
}
