# Nom de l'exécutable
TARGET = bin/gloomy
BENCHMARK_TARGET = bin/benchmark
BENCHMARK_SRC = src/BenchmarkRunner.cpp src/DenseLayer.cpp src/NeuralNetwork.cpp src/Optimizer.cpp src/LearningEngine.cpp src/LossFunction.cpp src/Metrics.cpp src/Benchmark.cpp src/MomentumOptimizer.cpp src/AdamOptimizer.cpp src/ImportanceScorer.cpp src/TrainingScheduler.cpp src/FIFOMemory.cpp src/ReservoirMemory.cpp src/PrioritizedMemory.cpp src/Quantization.cpp src/Int8Quantization.cpp src/TrainingSampleQuantization.cpp src/Int8TrainingSampleQuantization.cpp src/QuantizedFIFOMemory.cpp src/QuantizedInt8FIFOMemory.cpp

# Cible de test de la fonction de perte
TEST_TARGET = bin/loss_tests
TEST_SRC = tests/loss_tests.cpp src/LossFunction.cpp src/DenseLayer.cpp src/NeuralNetwork.cpp src/Optimizer.cpp src/LearningEngine.cpp src/FIFOMemory.cpp src/ReservoirMemory.cpp src/PrioritizedMemory.cpp src/NoveltyMemory.cpp src/HybridMemory.cpp src/ImportanceScorer.cpp src/TrainingScheduler.cpp src/Normalization.cpp src/NormalizationSerialization.cpp src/Quantization.cpp src/Int8Quantization.cpp src/TrainingSampleQuantization.cpp src/Int8TrainingSampleQuantization.cpp src/QuantizedFIFOMemory.cpp src/QuantizedInt8FIFOMemory.cpp src/TrainingSampleSerialization.cpp src/NetworkSerialization.cpp src/Metrics.cpp src/Benchmark.cpp src/MomentumOptimizer.cpp src/AdamOptimizer.cpp

# Compilateur
CXX = g++

# Options de compilation
CXXFLAGS = -Wall -O2 -std=c++17 -I./src/headers

# Liste des fichiers source
SRC = src/main.cpp src/DenseLayer.cpp src/NeuralNetwork.cpp src/Optimizer.cpp src/LearningEngine.cpp src/FIFOMemory.cpp src/ReservoirMemory.cpp src/PrioritizedMemory.cpp src/NoveltyMemory.cpp src/HybridMemory.cpp src/ImportanceScorer.cpp src/TrainingScheduler.cpp src/Normalization.cpp src/NormalizationSerialization.cpp src/Quantization.cpp src/Int8Quantization.cpp src/TrainingSampleQuantization.cpp src/Int8TrainingSampleQuantization.cpp src/QuantizedFIFOMemory.cpp src/QuantizedInt8FIFOMemory.cpp src/TrainingSampleSerialization.cpp src/NetworkSerialization.cpp src/Metrics.cpp src/Benchmark.cpp src/MomentumOptimizer.cpp src/AdamOptimizer.cpp

# Liste des fichiers objets (transformation des fichiers .cpp en .o)
OBJ = $(SRC:.cpp=.o)

# Répertoire des fichiers objets
OBJ_DIR = src

# Répertoire de l'exécutable
TARGET_DIR = bin

# Règle par défaut : construire l'exécutable
all: $(TARGET)

benchmark: $(BENCHMARK_TARGET)
	./$(BENCHMARK_TARGET)

test: $(TEST_TARGET)
	./$(TEST_TARGET)

# Règle pour construire l'exécutable
$(TARGET): $(OBJ)
	@mkdir -p $(TARGET_DIR)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJ)

$(BENCHMARK_TARGET): $(BENCHMARK_SRC)
	@mkdir -p $(TARGET_DIR)
	$(CXX) $(CXXFLAGS) -o $(BENCHMARK_TARGET) $(BENCHMARK_SRC)

$(TEST_TARGET): $(TEST_SRC)
	@mkdir -p $(TARGET_DIR)
	$(CXX) $(CXXFLAGS) -o $(TEST_TARGET) $(TEST_SRC)

# Règle pour compiler les fichiers .cpp en fichiers .o
$(OBJ_DIR)/%.o: src/%.cpp
	@mkdir -p $(TARGET_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Règle pour nettoyer les fichiers générés
clean:
	rm -f $(TARGET)
	rm -f $(BENCHMARK_TARGET)
	rm -f $(TEST_TARGET)
	rm -f $(OBJ_DIR)/*.o
	rmdir -p $(TARGET_DIR)

