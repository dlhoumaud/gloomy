#include "headers/GloomyConfig.h"

const GloomyConfig& GloomyConfig::defaults() {
    static const GloomyConfig instance;
    return instance;
}
