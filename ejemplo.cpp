#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <queue>
#include <limits>

typedef std::vector<double> Data;
typedef std::unordered_map<std::string, Data> InvertedIndex;

InvertedIndex originalIndex;  // Índice invertido original con nombres de canciones y características numéricas
InvertedIndex treeIndex;      // Índice invertido específico para el Ball Tree con nombres de canciones

std::vector<std::string> split(const std::string& line, char delimiter);
void fillInvertedIndices(const std::string& filename, InvertedIndex& originalIndex, InvertedIndex& treeIndex);
void printInvertedIndex(const InvertedIndex& invertedIndex);
void printInvertedIndex(const InvertedIndex& invertedIndex, int numRecords);
void printData(const Data& data);

struct BallNode {
    std::string pivot;
    Data pivotFeatures;
    double radius;
    BallNode* child1;
    BallNode* child2;

    BallNode(const std::string& pivot, const Data& pivotFeatures)
        : pivot(pivot), pivotFeatures(pivotFeatures), radius(0.0), child1(nullptr), child2(nullptr) {}

    bool isLeaf() const {
        return child1 == nullptr && child2 == nullptr;
    }
};

class BallTree {
public:
    int minLeafPoints;  // Umbral mínimo de puntos para un nodo hoja
    BallNode* root;

    BallTree(int minLeafPoints) : root(nullptr), minLeafPoints(minLeafPoints) {}

    // Función para construir el Ball Tree
    BallNode* constructBallTree(const std::vector<std::string>& points) {
        if (points.empty()) {
            return nullptr;
        }
        // Crear un nodo hoja si se cumple la condición de tener un número mínimo de puntos
        if (points.size() <= minLeafPoints) {
            BallNode* leaf = new BallNode("", Data());
            leaf->child1 = leaf->child2 = nullptr;
            std::cout << "Nodo creado" << std::endl;
            return leaf;
        }

        // Calcular la dimensión con mayor dispersión
        int maxSpreadDim = calculateMaxSpreadDimension(points);
        //std::cout << "maxSpreadDim: " << maxSpreadDim << std::endl;

        // Seleccionar el punto central considerando la dimensión con mayor dispersión
        std::string centralPoint = selectCentralPoint(points, maxSpreadDim);
        //std::cout << "Central: " << centralPoint << std::endl;
        Data centralFeatures = treeIndex[centralPoint];

        // Dividir los puntos en dos conjuntos (izquierdo y derecho) según la mediana en la dimensión seleccionada
        std::vector<std::string> leftPoints, rightPoints;
        splitPoints(points, maxSpreadDim, centralFeatures, leftPoints, rightPoints);

        // Construir recursivamente los subárboles del Ball Tree
        BallNode* node = new BallNode(centralPoint, centralFeatures);
        node->child1 = constructBallTree(leftPoints);
        node->child2 = constructBallTree(rightPoints);

        // Calcular el radio del nodo actual como la máxima distancia entre el punto central y sus hijos
        double maxDistance = std::max(calculateDistance(centralFeatures, node->child1->pivotFeatures),
                                      calculateDistance(centralFeatures, node->child2->pivotFeatures));
        node->radius = maxDistance;

        return node;
    }

    // Función para calcular la dimensión con mayor dispersión
    int calculateMaxSpreadDimension(const std::vector<std::string>& points) {
        if (points.empty()) {
            throw std::invalid_argument("Cannot calculate max spread dimension: empty list of points.");
            return -1;
        }
        int dimensions = 13;
        int maxSpreadDim = 0;
        double maxSpread = 0.0;

        for (int dim = 0; dim < dimensions; ++dim) {
            double minValue = std::numeric_limits<double>::max();
            double maxValue = std::numeric_limits<double>::lowest();

            for (const std::string& point : points) {
                double value = treeIndex[point][dim];
                if (value < minValue) {
                    minValue = value;
                }
                if (value > maxValue) {
                    maxValue = value;
                }
            }

            double spread = maxValue - minValue;
            if (spread > maxSpread) {
                maxSpread = spread;
                maxSpreadDim = dim;
            }
        }

        return maxSpreadDim;
    }

    // Función para seleccionar el punto central considerando la dimensión con mayor dispersión
    std::string selectCentralPoint(const std::vector<std::string>& points, int dimension) {
        if (points.empty()) {
            throw std::invalid_argument("Cannot select central point: empty list of points.");
        }
        double maxValue = std::numeric_limits<double>::lowest();
        std::string centralPoint;

        for (const std::string& point : points) {
            double value = treeIndex[point][dimension];
            if (value > maxValue) {
                maxValue = value;
                centralPoint = point;
            }
        }

        return centralPoint;
    }

    // Función para dividir los puntos en dos conjuntos (izquierdo y derecho) según la mediana en la dimensión seleccionada
    void splitPoints(const std::vector<std::string>& points, int dimension, const Data& centralFeatures,
                     std::vector<std::string>& leftPoints, std::vector<std::string>& rightPoints) {
//        std::cout << "Central feactures: " << std::endl;
//        printData(centralFeatures);
//        std::cout << "Dimension: " <<dimension<< std::endl;
//        std::cout << "treeIndex " << std::endl;
        bool isLessThanOrEqual;
        for (const std::string& point : points) {
            isLessThanOrEqual = (treeIndex[point][dimension] < centralFeatures[dimension]);
            //printData(treeIndex[point]);
            //std::cout << "point: " << treeIndex[point][dimension] << " < " << centralFeatures[dimension] <<" = "<<isLessThanOrEqual<< std::endl;
            if (isLessThanOrEqual) {
                leftPoints.push_back(point);
            }
            else if(!isLessThanOrEqual) {
                //std::cout<<"Entre....................."<<std::endl;
                rightPoints.push_back(point);
            }
        }
        std::cout << "size1 " << leftPoints.size() << " size2 " << rightPoints.size()<< std::endl;
    }

    // Función para calcular la distancia euclidiana entre dos puntos
    double calculateDistance(const Data& point1, const Data& point2) {
        double distance = 0.0;
        for (size_t i = 0; i < point1.size(); ++i) {
            distance += std::pow(point1[i] - point2[i], 2);
        }
        return std::sqrt(distance);
    }

    // Función para encontrar los k vecinos más cercanos de una canción específica en el Ball Tree
    std::vector<std::string> findNearestNeighbors(const std::string& querySong, int k) {
        std::priority_queue<std::pair<double, std::string>> pq;
        findNearestNeighborsHelper(root, querySong, k, pq);

        std::vector<std::string> neighbors;
        while (!pq.empty()) {
            neighbors.push_back(pq.top().second);
            pq.pop();
        }

        return neighbors;
    }

private:
    void findNearestNeighborsHelper(BallNode* node, const std::string& querySong, int k,
                                    std::priority_queue<std::pair<double, std::string>>& pq) {
        if (node == nullptr) {
            return;
        }

        if (node->isLeaf()) {
            // El nodo actual es un nodo hoja
            double distance = calculateDistance(originalIndex[querySong], node->pivotFeatures);

            // Insertar la distancia y el nombre de la canción en la cola de prioridad
            pq.push(std::make_pair(distance, node->pivot));

            // Verificar si la cola de prioridad contiene más de k elementos
            // Si es así, eliminar el elemento con la mayor distancia
            if (static_cast<int>(pq.size()) > k) {
                pq.pop();
            }
        } else {
            // El nodo actual no es un nodo hoja
            double distance = node->radius > 0 ? calculateDistance(originalIndex[querySong], node->pivotFeatures) : 0;

            // Insertar la distancia y el nombre de la canción en la cola de prioridad
            pq.push(std::make_pair(distance, node->pivot));

            // Verificar si la cola de prioridad contiene más de k elementos
            // Si es así, eliminar el elemento con la mayor distancia
            if (static_cast<int>(pq.size()) > k) {
                pq.pop();
            }

            // Calcular la distancia del punto de consulta a los subárboles de los nodos hijos
            double leftDistance = node->child1 ? calculateDistance(originalIndex[querySong], node->child1->pivotFeatures) : 0;
            double rightDistance = node->child2 ? calculateDistance(originalIndex[querySong], node->child2->pivotFeatures) : 0;

            // Determinar qué subárbol explorar primero en función de las distancias calculadas
            BallNode* firstChild = leftDistance < rightDistance ? node->child1 : node->child2;
            BallNode* secondChild = leftDistance < rightDistance ? node->child2 : node->child1;

            // Recursivamente explorar el subárbol más cercano primero
            findNearestNeighborsHelper(firstChild, querySong, k, pq);

            // Verificar si es necesario explorar el subárbol restante
            if (pq.size() < k || std::abs(distance - leftDistance) < pq.top().first) {
                findNearestNeighborsHelper(secondChild, querySong, k, pq);
            }
        }
    }
};

void printData(const Data& data) {
    for (const auto& value : data) {
        std::cout << value << " ";
    }
    std::cout << std::endl;
}

int main() {
    // Inicializar los índices invertidos
    fillInvertedIndices("songs_final.csv", originalIndex, treeIndex);

    // Construir el Ball Tree
    int minLeafPoints = 50;  // Ajusta el valor según tus necesidades
    BallTree ballTree(minLeafPoints);

    std::vector<std::string> points;
    for (const auto& entry : treeIndex) {
        points.push_back(entry.first);
    }

    ballTree.root = ballTree.constructBallTree(points);

    // Consultar los k vecinos de una canción específica
    std::string querySong = "Mercury: Retrograde";
    int k = 5;
    std::vector<std::string> neighbors = ballTree.findNearestNeighbors(querySong, k);

    // Imprimir los k vecinos encontrados
    std::cout << "K Nearest Neighbors of " << querySong << ":" << std::endl;
    for (const std::string& neighbor : neighbors) {
        std::cout << neighbor << std::endl;
    }

    return 0;
}

void printInvertedIndex(const InvertedIndex& invertedIndex) {
    for (const auto& entry : invertedIndex) {
        const std::string& songName = entry.first;
        const Data& features = entry.second;

        std::cout << "Song: " << songName << std::endl;
        std::cout << "Features: ";
        printData(features);
        std::cout << std::endl;
    }
}

void printInvertedIndex(const InvertedIndex& invertedIndex, int numRecords) {
    int count = 0;
    for (const auto& entry : invertedIndex) {
        if (count >= numRecords) {
            break;
        }
        const std::string& songName = entry.first;
        const Data& features = entry.second;

        std::cout << "Song: " << songName << std::endl;
        std::cout << "Features: ";
        printData(features);
        std::cout << std::endl;

        count++;
    }
}

void fillInvertedIndices(const std::string& filename, InvertedIndex& originalIndex, InvertedIndex& treeIndex) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "Error: Failed to open the file.\n";
        return;
    }

    std::string line;
    std::getline(file, line); // Read and discard the first header line

    while (std::getline(file, line)) {
        // Split the line into fields using comma as the delimiter
        std::vector<std::string> fields = split(line, ',');

        if (fields.size() < 14) {
            std::cout << "Error: Insufficient number of fields.\n";
            continue;
        }

        // Get the numeric features
        Data features;
        for (size_t i = 0; i < 13; i++) {
            try {
                features.push_back(std::stod(fields[i]));
            } catch (const std::exception& e) {
                throw std::runtime_error("Error: Invalid numeric field.");
            }
        }

        // Concatenate the remaining fields as the song name
        std::string songName;
        for (size_t i = 13; i < fields.size(); i++) {
            songName += fields[i];
            if (i < fields.size() - 1) {
                songName += ",";
            }
        }

        // Add the data to the inverted indices
        originalIndex[songName] = features;
        treeIndex[songName] = features;
    }
}

// Function to split a string into fields using a delimiter
std::vector<std::string> split(const std::string& line, char delimiter) {
    std::vector<std::string> fields;
    std::istringstream ss(line);
    std::string field;
    while (std::getline(ss, field, delimiter)) {
        fields.push_back(field);
    }
    return fields;
}
