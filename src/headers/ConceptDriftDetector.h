#ifndef CONCEPT_DRIFT_DETECTOR_H
#define CONCEPT_DRIFT_DETECTOR_H

#include <cstddef>
#include <vector>

// Detection active de concept drift a partir d'un flux d'erreurs
// (typiquement loss_before_update d'un pas d'apprentissage online) : pas
// seulement une metrique observee apres coup, mais un signal booleen
// utilisable pour declencher une action immediate (voir
// OnlineLearningRuntime.cpp et docs/roadmap.md, « Priorité moyenne :
// mémoire et continual learning »).
//
// Principe : compare la moyenne d'une fenetre glissante recente ("erreur
// maintenant") a la moyenne et l'ecart-type d'une ligne de base cumulee
// depuis la construction ("erreur historique"). Une derive est signalee
// quand la moyenne recente depasse la ligne de base de plus de
// `num_std_devs` ecarts-types (regle a 3 sigma par defaut) — une fois que
// la fenetre recente est pleine et que la ligne de base a vu au moins
// `minimum_history` valeurs, pour eviter les faux positifs au demarrage.
//
// La ligne de base continue d'integrer toutes les valeurs vues, y compris
// celles d'un nouveau regime : elle finit donc par "rattraper" une derive
// durable, et le signal peut naturellement redevenir faux une fois que le
// modele s'est reellement adapte (l'erreur recente redescend), sans
// mecanisme de reinitialisation explicite necessaire pour ce cas.
class ConceptDriftDetector {
public:
    explicit ConceptDriftDetector(
        std::size_t recent_window = 10,
        std::size_t minimum_history = 20,
        double num_std_devs = 3.0
    );

    // Ajoute une nouvelle valeur d'erreur et retourne si une derive est
    // detectee apres cet ajout (la ligne de base est mise a jour apres le
    // calcul de detection, pour que la valeur qui vient d'arriver ne dilue
    // pas immediatement sa propre comparaison).
    bool update(double error);

    bool driftDetected() const;
    double recentMean() const;
    double baselineMean() const;
    double baselineStdDev() const;
    std::size_t baselineCount() const;

    void reset();

private:
    std::size_t recent_window_size;
    std::size_t minimum_history_count;
    double std_dev_threshold;

    std::vector<double> recent_errors;
    std::size_t recent_next_index = 0;

    std::size_t baseline_count = 0;
    double baseline_mean = 0.0;
    double baseline_m2 = 0.0;

    bool drift_flag = false;
};

#endif // CONCEPT_DRIFT_DETECTOR_H
