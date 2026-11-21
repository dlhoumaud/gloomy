# Nom de l'exécutable
TARGET = bin/gloomy
BENCHMARK_TARGET = bin/benchmark
COMPARISON_SCRIPT = scripts/benchmark_optimization_comparison.sh
BENCHMARK_SRC = src/BenchmarkRunner.cpp src/GloomyModelSerialization.cpp src/LearningMemorySerialization.cpp src/Normalization.cpp src/DenseLayer.cpp src/NeuralNetwork.cpp src/Optimizer.cpp src/LearningEngine.cpp src/LossFunction.cpp src/Metrics.cpp src/Benchmark.cpp src/MomentumOptimizer.cpp src/AdamOptimizer.cpp src/ImportanceScorer.cpp src/TrainingScheduler.cpp src/FIFOMemory.cpp src/ReservoirMemory.cpp src/PrioritizedMemory.cpp src/NoveltyMemory.cpp src/HybridMemory.cpp src/Quantization.cpp src/Int8Quantization.cpp src/TrainingSampleQuantization.cpp src/Int8TrainingSampleQuantization.cpp src/QuantizedFIFOMemory.cpp src/QuantizedInt8FIFOMemory.cpp

# Cible de test de la fonction de perte
TEST_TARGET = bin/loss_tests
TEST_SRC = tests/loss_tests.cpp src/GloomyModelSerialization.cpp src/LearningMemorySerialization.cpp src/OptimizerSerialization.cpp src/LossFunction.cpp src/DenseLayer.cpp src/NeuralNetwork.cpp src/Optimizer.cpp src/LearningEngine.cpp src/FIFOMemory.cpp src/ReservoirMemory.cpp src/PrioritizedMemory.cpp src/NoveltyMemory.cpp src/HybridMemory.cpp src/ImportanceScorer.cpp src/TrainingScheduler.cpp src/Normalization.cpp src/NormalizationSerialization.cpp src/Quantization.cpp src/Int8Quantization.cpp src/TrainingSampleQuantization.cpp src/Int8TrainingSampleQuantization.cpp src/QuantizedFIFOMemory.cpp src/QuantizedInt8FIFOMemory.cpp src/TrainingSampleSerialization.cpp src/NetworkSerialization.cpp src/Metrics.cpp src/Benchmark.cpp src/MomentumOptimizer.cpp src/AdamOptimizer.cpp

# Compilateur
CXX = g++

# Options de compilation
CXXFLAGS = -Wall -O2 -std=c++17 -I./src/headers

# Liste des fichiers source
SRC = src/main.cpp src/Runtime.cpp src/LanguageRuntime.cpp src/TransformerLM.cpp src/TextTokenizer.cpp src/GloomyModelSerialization.cpp src/LearningMemorySerialization.cpp src/OptimizerSerialization.cpp src/LossFunction.cpp src/DenseLayer.cpp src/NeuralNetwork.cpp src/Optimizer.cpp src/LearningEngine.cpp src/FIFOMemory.cpp src/ReservoirMemory.cpp src/PrioritizedMemory.cpp src/NoveltyMemory.cpp src/HybridMemory.cpp src/ImportanceScorer.cpp src/TrainingScheduler.cpp src/Normalization.cpp src/NormalizationSerialization.cpp src/Quantization.cpp src/Int8Quantization.cpp src/TrainingSampleQuantization.cpp src/Int8TrainingSampleQuantization.cpp src/QuantizedFIFOMemory.cpp src/QuantizedInt8FIFOMemory.cpp src/TrainingSampleSerialization.cpp src/NetworkSerialization.cpp src/Metrics.cpp src/Benchmark.cpp src/MomentumOptimizer.cpp src/AdamOptimizer.cpp

# Liste des fichiers objets (transformation des fichiers .cpp en .o)
OBJ = $(SRC:.cpp=.o)

# Répertoire des fichiers objets
OBJ_DIR = src

# Répertoire de l'exécutable
TARGET_DIR = bin

.PHONY: all benchmark benchmark-comparison test test-language-model language-model-example clean

# Règle par défaut : construire l'exécutable
all: $(TARGET)

benchmark: $(BENCHMARK_TARGET)
	./$(BENCHMARK_TARGET)

benchmark-comparison:
	bash $(COMPARISON_SCRIPT)

test: $(TEST_TARGET) test-language-model
	./$(TEST_TARGET)

test-language-model: $(TARGET)
	python3 scripts/test_language_model.py

language-model-example: $(TARGET)
	mkdir -p artifacts
	./$(TARGET) lm train \
		--train docs/examples/language-model/train.txt \
		--validation docs/examples/language-model/validation.txt \
		--qa-train docs/examples/language-model/qa-train.txt \
		--model artifacts/language-mini.gloomy \
		--vocab 320 --context 32 --dimension 8 --heads 2 --layers 1 \
		--batch-size 2 --epochs 1 --learning-rate 0.003 --seed 42

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

src/TextTokenizer.o: src/UnicodeNormalizationData.h src/headers/TextTokenizer.h
src/TransformerLM.o: src/headers/TransformerLM.h
src/LanguageRuntime.o: src/headers/LanguageRuntime.h src/headers/TextTokenizer.h src/headers/TransformerLM.h

# Règle pour nettoyer les fichiers générés
clean:
	rm -f $(TARGET)
	rm -f $(BENCHMARK_TARGET)
	rm -f $(TEST_TARGET)
	rm -f $(OBJ_DIR)/*.o
	rmdir -p $(TARGET_DIR)

