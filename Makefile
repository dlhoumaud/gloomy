# Nom de l'exécutable
TARGET = bin/gloomy
BENCHMARK_TARGET = bin/benchmark
COMPARISON_SCRIPT = scripts/benchmark_optimization_comparison.sh
BENCHMARK_SRC = src/BenchmarkRunner.cpp src/GloomyModelSerialization.cpp src/LearningMemorySerialization.cpp src/Normalization.cpp src/DenseLayer.cpp src/NeuralNetwork.cpp src/Optimizer.cpp src/LearningEngine.cpp src/LossFunction.cpp src/Metrics.cpp src/Benchmark.cpp src/MomentumOptimizer.cpp src/AdamOptimizer.cpp src/ImportanceScorer.cpp src/TrainingScheduler.cpp src/FIFOMemory.cpp src/ReservoirMemory.cpp src/PrioritizedMemory.cpp src/NoveltyMemory.cpp src/HybridMemory.cpp src/Quantization.cpp src/Int8Quantization.cpp src/TrainingSampleQuantization.cpp src/Int8TrainingSampleQuantization.cpp src/QuantizedFIFOMemory.cpp src/QuantizedInt8FIFOMemory.cpp

# Cible de test de la fonction de perte
TEST_TARGET = bin/loss_tests
TEST_SRC = tests/loss_tests.cpp src/GloomyModelSerialization.cpp src/LearningMemorySerialization.cpp src/OptimizerSerialization.cpp src/LossFunction.cpp src/DenseLayer.cpp src/NeuralNetwork.cpp src/Optimizer.cpp src/LearningEngine.cpp src/FIFOMemory.cpp src/ReservoirMemory.cpp src/PrioritizedMemory.cpp src/NoveltyMemory.cpp src/HybridMemory.cpp src/ImportanceScorer.cpp src/TrainingScheduler.cpp src/Normalization.cpp src/NormalizationSerialization.cpp src/Quantization.cpp src/Int8Quantization.cpp src/TrainingSampleQuantization.cpp src/Int8TrainingSampleQuantization.cpp src/QuantizedFIFOMemory.cpp src/QuantizedInt8FIFOMemory.cpp src/TrainingSampleSerialization.cpp src/NetworkSerialization.cpp src/Metrics.cpp src/Benchmark.cpp src/MomentumOptimizer.cpp src/AdamOptimizer.cpp
TRANSFORMER_MATH_TARGET = bin/transformer_math_tests
TRANSFORMER_MATH_SRC = tests/transformer_math_tests.cpp src/TransformerLM.cpp

# Compilateur
CXX = g++

# Options de compilation
CXXFLAGS = -Wall -O2 -std=c++17 -I./src/headers
LDFLAGS = -static

# Liste des fichiers source
SRC = src/main.cpp src/Runtime.cpp src/LanguageRuntime.cpp src/TransformerLM.cpp src/TextTokenizer.cpp src/GloomyModelSerialization.cpp src/LearningMemorySerialization.cpp src/OptimizerSerialization.cpp src/LossFunction.cpp src/DenseLayer.cpp src/NeuralNetwork.cpp src/Optimizer.cpp src/LearningEngine.cpp src/FIFOMemory.cpp src/ReservoirMemory.cpp src/PrioritizedMemory.cpp src/NoveltyMemory.cpp src/HybridMemory.cpp src/ImportanceScorer.cpp src/TrainingScheduler.cpp src/Normalization.cpp src/NormalizationSerialization.cpp src/Quantization.cpp src/Int8Quantization.cpp src/TrainingSampleQuantization.cpp src/Int8TrainingSampleQuantization.cpp src/QuantizedFIFOMemory.cpp src/QuantizedInt8FIFOMemory.cpp src/TrainingSampleSerialization.cpp src/NetworkSerialization.cpp src/Metrics.cpp src/Benchmark.cpp src/MomentumOptimizer.cpp src/AdamOptimizer.cpp

# Liste des fichiers objets (transformation des fichiers .cpp en .o)
OBJ = $(SRC:.cpp=.o)

# Répertoire des fichiers objets
OBJ_DIR = src

# Répertoire de l'exécutable
TARGET_DIR = bin

.PHONY: all benchmark benchmark-comparison test test-language-model test-transformer-math test-language-model-docs-resume test-language-model-ocara-resume language-model-example language-model-docs-example language-model-ocara-example benchmark-language-model-docs benchmark-language-model-ocara benchmark-language-model-weight-precision benchmark-language-model-sequence-replay clean

# Règle par défaut : construire l'exécutable
all: $(TARGET)

benchmark: $(BENCHMARK_TARGET)
	./$(BENCHMARK_TARGET)

benchmark-comparison:
	bash $(COMPARISON_SCRIPT)

test: $(TEST_TARGET) test-language-model test-transformer-math
	./$(TEST_TARGET)

test-transformer-math: $(TRANSFORMER_MATH_TARGET)
	./$(TRANSFORMER_MATH_TARGET)

test-language-model: $(TARGET)
	python3 scripts/test_language_model.py

test-language-model-docs-resume: $(TARGET)
	GLOOMY_TEST_DOC_RESUME=1 python3 scripts/test_language_model.py

test-language-model-ocara-resume: $(TARGET)
	GLOOMY_TEST_OCARA_RESUME=1 python3 scripts/test_language_model.py

language-model-example: $(TARGET)
	mkdir -p artifacts
	./$(TARGET) lm train \
		--train docs/examples/language-model/train.txt \
		--validation docs/examples/language-model/validation.txt \
		--qa-train docs/examples/language-model/qa-train.txt \
		--model artifacts/language-mini.gloomy \
		--vocab 320 --context 32 --dimension 8 --heads 2 --layers 1 \
		--batch-size 2 --epochs 1 --learning-rate 0.003 --seed 42

language-model-docs-example: $(TARGET)
	python3 scripts/prepare_gloomy_docs_corpus.py
	mkdir -p artifacts
	./$(TARGET) lm train \
		--train data/lm/gloomy-docs/train.txt \
		--validation data/lm/gloomy-docs/validation.txt \
		--qa-train data/lm/gloomy-docs/qa-train.txt \
		--model artifacts/gloomy-docs.gloomy \
		--vocab 512 --context 32 --dimension 8 --heads 2 --layers 1 \
		--batch-size 2 --epochs 3 --learning-rate 0.003 --seed 42

benchmark-language-model-docs: $(TARGET)
	python3 scripts/prepare_gloomy_docs_corpus.py
	python3 scripts/benchmark_language_model.py \
		--train data/lm/gloomy-docs/train.txt \
		--validation data/lm/gloomy-docs/validation.txt \
		--qa-train data/lm/gloomy-docs/qa-train.txt \
		--qa-validation data/lm/gloomy-docs/qa-validation.txt \
		--test data/lm/gloomy-docs/test.txt \
		--qa data/lm/gloomy-docs/qa-test.txt \
		--qa-allowed-label oui --qa-allowed-label non \
		--corpus-source "First-party Gloomy repository documentation" \
		--corpus-authorization "Local training use requested by the project user; no redistribution license asserted" \
		--source-manifest data/lm/gloomy-docs/manifest.json \
		--output docs/examples/language-model/gloomy-docs-3-seeds.csv \
		--seeds 42,43,44 --vocab 512 --context 32 --dimension 8 \
		--heads 2 --layers 1 --batch-size 2 --epochs 3 --learning-rate 0.003

language-model-ocara-example: $(TARGET)
	python3 scripts/prepare_ocara_corpus.py
	mkdir -p artifacts
	./$(TARGET) lm train \
		--train data/lm/ocara/train.txt \
		--validation data/lm/ocara/validation.txt \
		--qa-train data/lm/ocara/qa-train.txt \
		--model artifacts/ocara-docs.gloomy \
		--vocab 512 --context 64 --dimension 8 --heads 2 --layers 1 \
		--batch-size 2 --epochs 3 --learning-rate 0.003 --seed 42

benchmark-language-model-ocara: $(TARGET)
	python3 scripts/prepare_ocara_corpus.py
	python3 scripts/benchmark_language_model.py \
		--train data/lm/ocara/train.txt \
		--validation data/lm/ocara/validation.txt \
		--qa-train data/lm/ocara/qa-train.txt \
		--qa-validation data/lm/ocara/qa-validation.txt \
		--test data/lm/ocara/test.txt \
		--qa data/lm/ocara/qa-test.txt \
		--qa-allowed-label oui --qa-allowed-label non \
		--corpus-source "Ocara project documentation provided by the project user" \
		--corpus-authorization "Local training and evaluation authorized by the user; no redistribution license asserted" \
		--source-manifest data/lm/ocara/manifest.json \
		--output docs/examples/language-model/ocara-3-seeds.csv \
		--seeds 42,43,44 --vocab 512 --context 64 --dimension 8 \
		--heads 2 --layers 1 --batch-size 2 --epochs 3 --learning-rate 0.003

# Compare fake-quantized weights using the existing local Gloomy documentation checkpoint.
benchmark-language-model-weight-precision: $(TARGET)
	python3 scripts/prepare_gloomy_docs_corpus.py
	python3 scripts/benchmark_lm_weight_precision.py \
		--model artifacts/gloomy-docs.gloomy \
		--text data/lm/gloomy-docs/test.txt \
		--qa data/lm/gloomy-docs/qa-test.txt \
		--output docs/examples/language-model/weight-precision-pilot.csv
	python3 scripts/prepare_ocara_corpus.py
	python3 scripts/benchmark_lm_weight_precision.py \
		--model artifacts/gloomy-docs.gloomy \
		--text data/lm/ocara/test.txt \
		--qa data/lm/ocara/qa-test.txt \
		--output docs/examples/language-model/ocara-precision-pilot.csv

benchmark-language-model-sequence-replay: language-model-docs-example
	python3 scripts/benchmark_lm_sequence_replay.py \
		--model artifacts/gloomy-docs.gloomy \
		--output docs/examples/language-model/sequence-replay-pilot.csv

# Règle pour construire l'exécutable
$(TARGET): $(OBJ) Makefile
	@mkdir -p $(TARGET_DIR)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(TARGET) $(OBJ)

$(BENCHMARK_TARGET): $(BENCHMARK_SRC) Makefile
	@mkdir -p $(TARGET_DIR)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(BENCHMARK_TARGET) $(BENCHMARK_SRC)

$(TEST_TARGET): $(TEST_SRC) Makefile
	@mkdir -p $(TARGET_DIR)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(TEST_TARGET) $(TEST_SRC)

$(TRANSFORMER_MATH_TARGET): $(TRANSFORMER_MATH_SRC) src/headers/TransformerLM.h Makefile
	@mkdir -p $(TARGET_DIR)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(TRANSFORMER_MATH_TARGET) $(TRANSFORMER_MATH_SRC)

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
	rm -f $(TRANSFORMER_MATH_TARGET)
	rm -f $(OBJ_DIR)/*.o
	rmdir -p $(TARGET_DIR)

