#ifndef DELTA_QUANTIZATION_H
#define DELTA_QUANTIZATION_H

#include "Quantization.h"
#include <cstddef>
#include <vector>

// Compression differentielle d'une serie temporelle : au lieu de quantifier
// les valeurs brutes (Int16Quantizer directement), on quantifie la
// difference entre valeurs consecutives ("delta"). Pour une serie qui varie
// lentement (forte correlation temporelle), les deltas ont une plage
// beaucoup plus etroite que les valeurs elles-memes : la meme largeur
// int16 leur alloue donc un pas de quantification (`scale`) plus fin, donc
// une erreur de reconstruction plus faible — voir docs/quantization.md,
// « Compression differentielle ».
struct DeltaQuantizedSeries {
    // Premiere valeur de la serie, conservee exactement (ancre de la
    // reconstruction : toutes les valeurs suivantes en derivent par sommes
    // cumulees des deltas dequantifies).
    double first_value = 0.0;
    // Deltas quantifies : deltas.values.size() == valeurs de la serie - 1.
    QuantizedVector deltas;
};

class DeltaQuantizer {
public:
    // Rejette une serie vide. Une serie a une seule valeur produit un
    // DeltaQuantizedSeries sans aucun delta (deltas.values vide).
    static DeltaQuantizedSeries encode(const std::vector<double>& values);
    static std::vector<double> decode(const DeltaQuantizedSeries& encoded);

    // Octets reellement occupes par la serie encodee (first_value en
    // double + un int16 par delta + les parametres de calibration), a
    // comparer a values.size() * sizeof(double) pour la representation
    // native, ou a Int16Quantizer applique directement aux valeurs brutes
    // pour comparer les deux strategies de quantification a bit-width egal.
    static std::size_t quantizedBytes(const DeltaQuantizedSeries& encoded);
};

#endif // DELTA_QUANTIZATION_H
