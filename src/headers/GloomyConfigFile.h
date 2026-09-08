#ifndef GLOOMY_CONFIG_FILE_H
#define GLOOMY_CONFIG_FILE_H

#include "GloomyConfig.h"
#include <string>

// Minimal key=value configuration file parser (see docs/roadmap.md,
// section « Configuration fichier »). No sections, no quoting: one
// `key=value` pair per line, blank lines and lines starting with '#' or ';'
// are treated as comments, and surrounding whitespace around the key and
// the value is trimmed. An unknown key or a malformed line is rejected.
class GloomyConfigFile {
public:
    // Reads `path` and returns a copy of `base` with every key found in the
    // file applied on top of it. Passing GloomyConfig::defaults() (the
    // default) means the result reflects "fichier > défauts"; the caller is
    // then expected to apply any explicit CLI flag on top of the result,
    // for the full "CLI > fichier > défauts" priority.
    static GloomyConfig load(
        const std::string& path,
        const GloomyConfig& base = GloomyConfig::defaults()
    );
};

#endif // GLOOMY_CONFIG_FILE_H
