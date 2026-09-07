# Nom de l'exécutable
TARGET = bin/gloomy

# Compilateur
CXX = g++

# Options de compilation
CXXFLAGS = -Wall -O2 -std=c++17 -I./src/headers

# Liste des fichiers source
SRC = src/main.cpp src/DenseLayer.cpp src/NeuralNetwork.cpp

# Liste des fichiers objets (transformation des fichiers .cpp en .o)
OBJ = $(SRC:.cpp=.o)

# Répertoire des fichiers objets
OBJ_DIR = src

# Répertoire de l'exécutable
TARGET_DIR = bin

# Règle par défaut : construire l'exécutable
all: $(TARGET)

# Règle pour construire l'exécutable
$(TARGET): $(OBJ)
	@mkdir -p $(TARGET_DIR)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJ)

# Règle pour compiler les fichiers .cpp en fichiers .o
$(OBJ_DIR)/%.o: src/%.cpp
	@mkdir -p $(TARGET_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Règle pour nettoyer les fichiers générés
clean:
	rm -f $(TARGET)
	rm -f $(OBJ_DIR)/*.o
	rmdir -p $(TARGET_DIR)

