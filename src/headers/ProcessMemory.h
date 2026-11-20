#ifndef PROCESS_MEMORY_H
#define PROCESS_MEMORY_H

#include <cstddef>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

inline std::size_t peakResidentSetBytes() noexcept {
#if defined(__unix__) || defined(__APPLE__)
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0 || usage.ru_maxrss < 0) return 0;
#if defined(__APPLE__)
    return static_cast<std::size_t>(usage.ru_maxrss);
#else
    return static_cast<std::size_t>(usage.ru_maxrss) * 1024;
#endif
#else
    return 0;
#endif
}

#endif // PROCESS_MEMORY_H
