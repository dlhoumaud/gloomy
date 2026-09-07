#ifndef IMPORTANCE_SCORER_H
#define IMPORTANCE_SCORER_H

struct ImportanceWeights {
    double error = 1.0;
    double novelty = 0.0;
    double rarity = 0.0;
    double recency = 0.0;
    double diversity = 0.0;
};

struct ImportanceComponents {
    double error = 0.0;
    double novelty = 0.0;
    double rarity = 0.0;
    double recency = 0.0;
    double diversity = 0.0;
};

class ImportanceScorer {
public:
    explicit ImportanceScorer(ImportanceWeights weights = {});

    double score(const ImportanceComponents& components) const;
    const ImportanceWeights& weights() const;
    void setWeights(const ImportanceWeights& weights);

private:
    ImportanceWeights importance_weights;
};

#endif // IMPORTANCE_SCORER_H
