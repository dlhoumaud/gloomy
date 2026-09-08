#ifndef NORMALIZATION_H
#define NORMALIZATION_H

#include <cstddef>
#include <vector>

class StreamingNormalizer {
public:
    explicit StreamingNormalizer(size_t dimensions);

    void update(const std::vector<double>& values);
    std::vector<double> normalize(const std::vector<double>& values) const;
    // Variante en place : ecrit dans `out` (redimensionne au besoin) au lieu
    // de retourner un nouveau vecteur. Reutiliser le meme `out` d'un appel a
    // l'autre evite une allocation par appel une fois sa capacite etablie —
    // utile dans une boucle d'apprentissage en continu (voir
    // OnlineLearningRuntime.cpp). Memes validations et memes resultats que
    // la surcharge par valeur, qui delegue desormais a celle-ci.
    void normalize(const std::vector<double>& values, std::vector<double>& out) const;
    void clear();
    void restore(
        size_t count,
        const std::vector<double>& means,
        const std::vector<double>& variances,
        const std::vector<double>& minimum,
        const std::vector<double>& maximum
    );

    size_t count() const;
    size_t dimensions() const;
    const std::vector<double>& mean() const;
    std::vector<double> variance() const;
    const std::vector<double>& minimum() const;
    const std::vector<double>& maximum() const;

private:
    void validate(const std::vector<double>& values) const;

    size_t value_count = 0;
    std::vector<double> means;
    std::vector<double> moments;
    std::vector<double> minimum_values;
    std::vector<double> maximum_values;
};

#endif // NORMALIZATION_H
