#ifndef NETWORK_SERIALIZATION_H
#define NETWORK_SERIALIZATION_H

#include "NeuralNetwork.h"
#include <string>

class NetworkSerialization {
public:
    static void save(const std::string& path, const NeuralNetwork& network);
    static void load(const std::string& path, NeuralNetwork& network);
};

#endif // NETWORK_SERIALIZATION_H
