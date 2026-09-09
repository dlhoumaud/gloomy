#ifndef PAGE_HINKLEY_DETECTOR_H
#define PAGE_HINKLEY_DETECTOR_H

#include <cstddef>

// Detection de derive par le test de Page-Hinkley (Page 1954, Hinkley
// 1971) : un test sequentiel de detection de rupture, distinct dans son
// principe de ConceptDriftDetector (qui compare une moyenne recente a une
// ligne de base par ecarts-types). Le Page-Hinkley accumule un ecart
// tolere (`delta`) a la moyenne courante, et signale une derive quand cet
// ecart cumule s'eloigne trop de son minimum observe — voir
// docs/memory.md, « Détection de dérive plus avancée : Page-Hinkley ».
//
// Formule (formulation classique) : soit x_t la valeur au pas t,
// m_t = moyenne courante de x_1..x_t,
// U_t = somme_{i=1}^{t} (x_i - m_i - delta),
// M_t = min_{i<=t}(U_i).
// Une derive est signalee quand (U_t - M_t) > lambda.
class PageHinkleyDetector {
public:
    // delta : magnitude de changement toleree avant de commencer a
    // accumuler un signal (plus grand = moins sensible au bruit).
    // lambda : seuil de detection sur l'ecart cumule (plus grand = moins
    // sensible, detection plus tardive mais moins de faux positifs).
    // Defauts calibres sur un changement de regime synthetique net (voir
    // tests/loss_tests.cpp) : detection en 0 a quelques pas apres la
    // transition, aucun faux positif observe sur un bruit stable.
    explicit PageHinkleyDetector(double delta = 0.05, double lambda = 10.0);

    // Ajoute une nouvelle valeur et retourne si une derive est detectee
    // apres cet ajout.
    bool update(double value);

    bool driftDetected() const;
    double delta() const;
    double lambda() const;
    std::size_t count() const;

    void reset();

private:
    double delta_tolerance;
    double lambda_threshold;

    double running_mean = 0.0;
    std::size_t sample_count = 0;
    double cumulative_sum = 0.0;
    double minimum_cumulative_sum = 0.0;
    bool drift_flag = false;
};

#endif // PAGE_HINKLEY_DETECTOR_H
