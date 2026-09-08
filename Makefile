# Nom de l'exécutable
TARGET = bin/gloomy
BENCHMARK_TARGET = bin/benchmark
BENCHMARK_SRC = src/BenchmarkRunner.cpp src/DenseLayer.cpp src/NeuralNetwork.cpp src/Optimizer.cpp src/LearningEngine.cpp src/LossFunction.cpp src/Metrics.cpp src/Benchmark.cpp src/MomentumOptimizer.cpp src/AdamOptimizer.cpp src/ImportanceScorer.cpp src/TrainingScheduler.cpp src/FIFOMemory.cpp src/ReservoirMemory.cpp src/PrioritizedMemory.cpp src/NoveltyMemory.cpp src/HybridMemory.cpp src/Quantization.cpp src/Int8Quantization.cpp src/TrainingSampleQuantization.cpp src/Int8TrainingSampleQuantization.cpp src/QuantizedFIFOMemory.cpp src/QuantizedInt8FIFOMemory.cpp

# Cible de test de la fonction de perte
TEST_TARGET = bin/loss_tests
TEST_SRC = tests/loss_tests.cpp src/GloomyConfig.cpp src/GloomyConfigFile.cpp src/OnlineLearningRuntime.cpp src/LossFunction.cpp src/DenseLayer.cpp src/NeuralNetwork.cpp src/Optimizer.cpp src/LearningEngine.cpp src/FIFOMemory.cpp src/ReservoirMemory.cpp src/PrioritizedMemory.cpp src/NoveltyMemory.cpp src/HybridMemory.cpp src/ImportanceScorer.cpp src/TrainingScheduler.cpp src/Normalization.cpp src/NormalizationSerialization.cpp src/Quantization.cpp src/Int8Quantization.cpp src/TrainingSampleQuantization.cpp src/Int8TrainingSampleQuantization.cpp src/QuantizedFIFOMemory.cpp src/QuantizedInt8FIFOMemory.cpp src/TrainingSampleSerialization.cpp src/NetworkSerialization.cpp src/OptimizerSerialization.cpp src/LearningMemorySerialization.cpp src/ModelSerialization.cpp src/Metrics.cpp src/Benchmark.cpp src/MomentumOptimizer.cpp src/AdamOptimizer.cpp

# Compilateur
CXX = g++

# Options de compilation
CXXFLAGS = -Wall -O2 -std=c++17 -I./src/headers

# Liste des fichiers source
SRC = src/main.cpp src/GloomyConfig.cpp src/GloomyConfigFile.cpp src/OnlineLearningRuntime.cpp src/LossFunction.cpp src/DenseLayer.cpp src/NeuralNetwork.cpp src/Optimizer.cpp src/LearningEngine.cpp src/FIFOMemory.cpp src/ReservoirMemory.cpp src/PrioritizedMemory.cpp src/NoveltyMemory.cpp src/HybridMemory.cpp src/ImportanceScorer.cpp src/TrainingScheduler.cpp src/Normalization.cpp src/NormalizationSerialization.cpp src/Quantization.cpp src/Int8Quantization.cpp src/TrainingSampleQuantization.cpp src/Int8TrainingSampleQuantization.cpp src/QuantizedFIFOMemory.cpp src/QuantizedInt8FIFOMemory.cpp src/TrainingSampleSerialization.cpp src/NetworkSerialization.cpp src/OptimizerSerialization.cpp src/LearningMemorySerialization.cpp src/ModelSerialization.cpp src/Metrics.cpp src/Benchmark.cpp src/MomentumOptimizer.cpp src/AdamOptimizer.cpp

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

# Règle pour compiler les fichiers .cpp en fichiers .o. -MMD -MP genere, a
# cote de chaque .o, un fichier .d listant les headers dont depend ce .cpp
# (voir le -include en bas de fichier). Sans cela, `make` ne recompile pas
# un .o dont un header inclus a change (sa seule dependance declaree serait
# le .cpp), ce qui peut lier ensemble des .o compiles contre des versions
# incompatibles d'un meme header (ODR violation) et provoquer une
# corruption memoire silencieuse a l'execution. N'est applique qu'ici : les
# cibles benchmark/test recompilent deja toutes leurs sources en une seule
# invocation g++ a chaque fois, donc ne peuvent pas etre perimees.
$(OBJ_DIR)/%.o: src/%.cpp
	@mkdir -p $(TARGET_DIR)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

# Règle pour nettoyer les fichiers générés
clean:
	rm -f $(TARGET)
	rm -f $(BENCHMARK_TARGET)
	rm -f $(TEST_TARGET)
	rm -f $(OBJ_DIR)/*.o
	rm -f $(OBJ_DIR)/*.d
	rmdir -p $(TARGET_DIR)

# Dependances des .o envers les headers qu'ils incluent, generees par
# -MMD -MP ci-dessus. Le tiret initial ignore silencieusement l'absence de
# ces fichiers avant la toute premiere compilation.
-include $(OBJ:.o=.d)

