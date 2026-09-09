#ifndef COMPRESSED_ADAM_OPTIMIZER_H
#define COMPRESSED_ADAM_OPTIMIZER_H

#include "Optimizer.h"
#include "Quantization.h"
#include <cstddef>
#include <vector>

// Variante d'Adam a etat compresse : les mêmes moments (premier, second)
// qu'AdamOptimizer, mais conserves entre deux appels a update() sous forme
// quantifiee int16 (Int16Quantizer) plutot qu'en double — voir
// docs/optimizers.md, « Adam a état compressé ». A chaque update() : les
// moments sont dequantifies, mis a jour avec la formule Adam standard en
// double (la precision de calcul du pas courant n'est pas degradee), puis
// requantifies (recalibres) avant d'etre stockes. Le compromis est
// symetrique a celui de QuantizedFIFOMemory/QuantizedInt8FIFOMemory :
// ~4x moins d'octets par valeur d'etat, au prix d'une erreur de
// quantification qui se propage (et peut s'accumuler) d'un pas a l'autre.
class CompressedAdamOptimizer final : public Optimizer {
public:
    CompressedAdamOptimizer(
        double learning_rate,
        double beta1 = 0.9,
        double beta2 = 0.999,
        double epsilon = 1e-8
    );

    void update(
        std::vector<DenseLayer>& layers,
        double gradient_scale = 1.0
    ) override;

    double learningRate() const;
    double beta1() const;
    double beta2() const;

    // Octets reellement occupes par l'etat compresse (int16 + parametres de
    // calibration par vecteur), a comparer a AdamOptimizer::stateBytes()
    // (double natif) pour la meme architecture.
    size_t stateBytes() const;

private:
    struct LayerState {
        std::size_t input_size = 0;
        std::size_t output_size = 0;
        QuantizedVector weight_first_moment;
        QuantizedVector weight_second_moment;
        QuantizedVector bias_first_moment;
        QuantizedVector bias_second_moment;
    };

    void ensureState(const std::vector<DenseLayer>& layers);

    double learning_rate;
    double first_decay;
    double second_decay;
    double epsilon;
    std::size_t update_count = 0;
    std::vector<LayerState> states;
};

#endif // COMPRESSED_ADAM_OPTIMIZER_H
