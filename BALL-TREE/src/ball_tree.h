#include <experimental/filesystem>
#include <Eigen/Dense>
#include <algorithm>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include <limits>
#include <chrono>
#include <string>
#include <cmath>
#include <queue>
#include "datos.h"
#include "pca.h"


struct Neighbor {
    double distance;
    std::string songName;

    Neighbor(double distance, const std::string& songName) : distance(distance), songName(songName) {}

    // Sobrecarga del operador de comparación para mantener la cola de prioridad ordenada en orden descendente
    bool operator<(const Neighbor& other) const {
        return distance < other.distance;
    }
};


struct BallNode {
    double radius;
    Data center;
    BallNode* child1;
    BallNode* child2;
    InvertedIndex ballTreeIndex;  // Índice invertido específico para el Ball Tree *
};

class BallTree {
public:
    int minLeafPoints;
    double alpha;  // Parámetro de conocimiento de carga (workload-awareness)
    BallNode* root;

    BallTree(int minLeafPoints, double alpha) : root(nullptr), minLeafPoints(minLeafPoints), alpha(alpha) {}

    void constructBallTree(InvertedIndex& treeSongIndex) {

        // Convertir el índice invertido original en un vector de puntos
        std::vector<Data> points;
        for (const auto& entry : treeSongIndex) {
            points.push_back(entry.second);
        }

        // Realizar PCA para obtener el primer eigenvector
        pca_t<double> pca;
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> inputMatrix(points.size(), points[0].size());
        for (int i = 0; i < points.size(); i++) {
            inputMatrix.row(i) = Eigen::Map<const Eigen::VectorXd>(points[i].data(), points[i].size());
        }
        pca.set_input(inputMatrix);
        pca.compute();
        const Eigen::Matrix<double, Eigen::Dynamic, 1>& firstEigenVector = pca.get_eigen_vectors().col(0);

        // Construir el Ball*-Tree recursivamente
        std::vector<int> indices(points.size());
        for (int i = 0; i < points.size(); i++) {
            indices[i] = i;
        }
        root = constructBallTreeRecursive(points, indices, firstEigenVector, treeSongIndex);
    }

    //KNN devueltos en orden ascendente de distancia (del más cercano al menos cercano)
    std::vector<std::string> findNearestNeighbors(const std::string& querySong, int k, InvertedIndex& diccSongIndex) {
        std::vector<std::string> neighbors;

        // Obtén el punto de consulta del índice invertido original
        Data queryPoint = diccSongIndex[querySong]; 

        //std::cout<<"query: "<< queryPoint.size() <<std::endl;
        if(queryPoint.size()==0) return neighbors;

        // Realiza el proceso de búsqueda en el Ball*-Tree
        std::priority_queue<Neighbor> nearestNeighbors;
        findNearestNeighborsRecursive(root, queryPoint, k, nearestNeighbors, querySong);

        // Agrega los nombres de las canciones vecinas al vector de vecinos
        //std::cout<<"Cola: "<< (!nearestNeighbors.empty()) <<std::endl;
        while (!nearestNeighbors.empty()) {
            const std::string& neighborSong = nearestNeighbors.top().songName;
            neighbors.push_back(neighborSong);
            nearestNeighbors.pop();
        }

        return neighbors;
    }

    void printBallTree(BallNode* node, int level=0) {
        if (node == nullptr) {
            return;
        }

        // Imprimir información del nodo actual
        for (int i = 0; i < level; i++) {
            std::cout << "  ";
        }
        std::cout << "- Radius: " << node->radius << ", Center: [";
        for (int i = 0; i < node->center.size(); i++) {
            std::cout << node->center[i];
            if (i < node->center.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;

        // Imprimir información de los hijos
        printBallTree(node->child1, level + 1);
        printBallTree(node->child2, level + 1);

        // Imprimir número de puntos en la hoja
        if (node->child1 == nullptr && node->child2 == nullptr) {
            for (int i = 0; i < level; i++) {
                std::cout << "  ";
            }
            std::cout << "  * Leaf, Number of Points: " << node->ballTreeIndex.size() << std::endl;
        }
    }

    std::vector<std::string> findConstrainedNearestNeighbors(const std::string& querySong, double range, int k, InvertedIndex& diccSongIndex) {
        std::vector<std::string> neighbors;

        // Obtén el punto de consulta del índice invertido original
        Data queryPoint = diccSongIndex[querySong];

        // Realiza el proceso de búsqueda restringida de vecinos más cercanos en el Ball*-Tree
        std::priority_queue<Neighbor> nearestNeighbors;
        findConstrainedNearestNeighborsRecursive(root, queryPoint, range, k, nearestNeighbors, querySong);

        // Agrega los nombres de las canciones vecinas al vector de vecinos
        while (!nearestNeighbors.empty()) {
            const std::string& neighborSong = nearestNeighbors.top().songName;
            neighbors.push_back(neighborSong);
            nearestNeighbors.pop();
        }

        return neighbors;
    }





private:

    BallNode* constructBallTreeRecursive(const std::vector<Data>& points, const std::vector<int>& indices,
                                         const Eigen::Matrix<double, Eigen::Dynamic, 1>& splitDirection,
                                         const InvertedIndex& originalIndex) {
        BallNode* ballNode = new BallNode;

        if (indices.size() <= minLeafPoints) {
            // Nodo hoja
            ballNode->radius = 0.0;
            ballNode->center = calculateCenter(points, indices);
            ballNode->child1 = nullptr;
            ballNode->child2 = nullptr;

            // Construir el índice invertido específico del Ball Tree *
            for (int index : indices) {
                const std::string& songName = getKeyByValue(originalIndex, points[index]);
                ballNode->ballTreeIndex[songName] = points[index];
            }
        } else {
            // Calcular el centro del nodo
            ballNode->center = calculateCenter(points, indices);

            // Calcular el radio del nodo
            double maxDistance = 0.0;
            for (int index : indices) {
                double distance = calculateDistance(points[index], ballNode->center);
                maxDistance = std::max(maxDistance, distance);
            }
            ballNode->radius = maxDistance;

            // Calcular el vector de proyecciones
            std::vector<double> projections;
            for (int index : indices) {
                double projection = calculateProjection(points[index], splitDirection);
                projections.push_back(projection);
            }

            // Calcular el valor óptimo para tc que minimiza la función de objetivo F(tc)
            double tmin = *std::min_element(projections.begin(), projections.end());
            double tmax = *std::max_element(projections.begin(), projections.end());
            int numSections = 100;  // Número de secciones para evaluar F(tc)
            double bestTc = tmin;
            double bestObjective = std::numeric_limits<double>::max();

            for (int i = 0; i <= numSections; i++) {
                double tc = tmin + (tmax - tmin) * i / numSections;

                // Calcular los tamaños de las particiones y la distancia mínima entre tc y las proyecciones
                int N = indices.size();
                int N1 = 0;
                int N2 = 0;
                for (double projection : projections) {
                    if (projection < tc) {
                        N1++;
                    } else {
                        N2++;
                    }
                }
                double objective = std::abs(N2 - N1) / static_cast<double>(N) + alpha * (tc - tmin) / (tmax - tmin);

                if (objective < bestObjective) {
                    bestObjective = objective;
                    bestTc = tc;
                }
            }

            // Dividir los puntos en dos subconjuntos según el valor óptimo de tc
            std::vector<int> leftIndices;
            std::vector<int> rightIndices;
            for (int index : indices) {
                double projection = calculateProjection(points[index], splitDirection);
                if (projection < bestTc) {
                    leftIndices.push_back(index);
                } else {
                    rightIndices.push_back(index);
                }
            }
            //std::cout<<"left: "<<leftIndices.size()<<std::endl;
            //std::cout<<"right: "<<rightIndices.size()<<std::endl<<std::endl;
            // Construir recursivamente los hijos del nodo
            ballNode->child1 = constructBallTreeRecursive(points, leftIndices, splitDirection, originalIndex);
            ballNode->child2 = constructBallTreeRecursive(points, rightIndices, splitDirection, originalIndex);
        }

        return ballNode;
    }
    void findNearestNeighborsRecursive(BallNode* node, const Data& queryPoint, int k,
                                       std::priority_queue<Neighbor>& nearestNeighbors, const std::string& querySong) {
        if (node->child1 == nullptr && node->child2 == nullptr) {
            //std::cout<<"Nodo hoja"<<std::endl;
            // Nodo hoja
            for (const auto& entry : node->ballTreeIndex) {
                const std::string& songName = entry.first; //std::cout<<"song: "<< songName<<std::endl;
                const Data& point = entry.second; 

                //std::cout<<"SPoint1: "<< point.size() <<std::endl;
                //std::cout<<"SPoint2: "<< queryPoint.size() <<std::endl;

                if (songName != querySong) { // Excluir la canción de consulta por su nombre
                    double distance = calculateDistance(point, queryPoint);
                    updateNearestNeighbors(nearestNeighbors, songName, distance, k);
                }
            }
        } else {
            //std::cout<<"Nodo interno"<<std::endl;
            //std::cout<<"Query: "<< queryPoint.size() <<std::endl;
            // Nodo interno
            double distance1 = calculateDistance(queryPoint, node->child1->center);//std::cout<<"Dist1: "<< distance1 <<std::endl;
            double distance2 = calculateDistance(queryPoint, node->child2->center);//std::cout<<"Dist2: "<< distance2 <<std::endl;

            //std::cout<<"Radius1: "<<node->child1->radius<<std::endl;
            //std::cout<<"Radius2: "<<node->child2->radius<<std::endl;

            bool F1 = (distance1 <= node->child1->radius) ;
            bool F2 = (distance2 <= node->child2->radius) ;

            if (F1 && F2) { //std::cout<<"2"<<std::endl;
                // Ambos hijos pueden contener vecinos más cercanos
                findNearestNeighborsRecursive(node->child1, queryPoint, k, nearestNeighbors, querySong);
                findNearestNeighborsRecursive(node->child2, queryPoint, k, nearestNeighbors, querySong);
            } else if (F1) { //std::cout<<"Nodo1"<<std::endl;
                // Solo el hijo izquierdo puede contener vecinos más cercanos
                findNearestNeighborsRecursive(node->child1, queryPoint, k, nearestNeighbors, querySong);
            } else if (F2) { //std::cout<<"Nodo2"<<std::endl;
                // Solo el hijo derecho puede contener vecinos más cercanos
                findNearestNeighborsRecursive(node->child2, queryPoint, k, nearestNeighbors, querySong);
            } else if (!F1 && !F2){
                // Ambos hijos pueden contener vecinos más cercanos
                findNearestNeighborsRecursive(node->child1, queryPoint, k, nearestNeighbors, querySong);
                findNearestNeighborsRecursive(node->child2, queryPoint, k, nearestNeighbors, querySong);
            }
        }
    }

    void findConstrainedNearestNeighborsRecursive(BallNode* node, const Data& queryPoint, double range, int k,
                                                  std::priority_queue<Neighbor>& nearestNeighbors, const std::string& querySong) {
        if (node->child1 == nullptr && node->child2 == nullptr) {
            // Nodo hoja
            for (const auto& entry : node->ballTreeIndex) {
                const std::string& songName = entry.first;
                const Data& point = entry.second;

                if (songName != querySong) {
                    double distance = calculateDistance(point, queryPoint);
                    if (distance <= range && (nearestNeighbors.size() < k || distance < nearestNeighbors.top().distance)) {
                        nearestNeighbors.push(Neighbor(distance, songName));
                        if (nearestNeighbors.size() > k) {
                            nearestNeighbors.pop();
                        }
                    }
                }
            }
        } else {
            // Nodo interno
            double distanceToLeft = calculateDistance(queryPoint, node->child1->center);
            double distanceToRight = calculateDistance(queryPoint, node->child2->center);

            BallNode* closerNode;
            BallNode* fartherNode;

            if (distanceToLeft < distanceToRight) {
                closerNode = node->child1;
                fartherNode = node->child2;
            } else {
                closerNode = node->child2;
                fartherNode = node->child1;
            }

            // Buscar en el nodo más cercano
            findConstrainedNearestNeighborsRecursive(closerNode, queryPoint, range, k, nearestNeighbors, querySong);

            // Verificar si es necesario buscar en el nodo más lejano
            if (nearestNeighbors.size() < k || std::abs(distanceToLeft - distanceToRight) < alpha * node->radius) {
                findConstrainedNearestNeighborsRecursive(fartherNode, queryPoint, range, k, nearestNeighbors, querySong);
            }
        }
    }

    double calculateProjection(const Data& point, const Eigen::Matrix<double, Eigen::Dynamic, 1>& splitDirection) {
        double projection = 0.0;
        int dimension = point.size();

        for (int i = 0; i < dimension; i++) {
            projection += point[i] * splitDirection[i];
        }

        return projection;
    }

    double calculateDistance(const Data& point1, const Data& point2) {
        //std::cout<<"SPoint1: "<< point1.size() <<std::endl;
        //std::cout<<"SPoint2: "<< point2.size() <<std::endl;
        double distance = 0.0;
        int dimension = point1.size();

        for (int i = 0; i < dimension; i++) {
            double diff = point1[i] - point2[i];
            distance += diff * diff;
        }

        return std::sqrt(distance);
    }

    double calculateDotProduct(const Data& point1, const Eigen::Matrix<double, Eigen::Dynamic, 1>& point2) {
        double dotProduct = 0.0;
        int dimension = point1.size();

        for (int i = 0; i < dimension; i++) {
            dotProduct += point1[i] * point2[i];
        }

        return dotProduct;
    }

    void updateNearestNeighbors(std::priority_queue<Neighbor>& nearestNeighbors, const std::string& songName,
                                double distance, int k) {
        Neighbor neighbor(distance, songName);
        nearestNeighbors.push(neighbor);

        if (nearestNeighbors.size() > k) {
            nearestNeighbors.pop();
        }
    }

    std::string getKeyByValue(const InvertedIndex& index, const Data& value) {
        for (const auto& entry : index) {
            if (entry.second == value) {
                return entry.first;
            }
        }
        return "";
    }
  

    Data calculateCenter(const std::vector<Data>& points, const std::vector<int>& indices) {
        int dimension = points[0].size();
        int numPoints = indices.size();
        Data center(dimension, 0.0);

        for (int i : indices) {
            for (int j = 0; j < dimension; j++) {
                center[j] += points[i][j];
            }
        }

        for (int j = 0; j < dimension; j++) {
            center[j] /= numPoints;
        }

        return center;
    }


};

int main() {
    // Inicializar los índices invertidos
    //InvertedIndex diccSongIndex;
    InvertedIndex treeSongIndex;

    fillInvertedIndices("songs_final1.csv",treeSongIndex);
    //printInvertedIndex(diccSongIndex,5);
    std::cout<<std::endl;
    //printInvertedIndex(treeSongIndex,5);
    //std::cout<<"\n index1 "<<diccSongIndex.size()<<std::endl;
    //std::cout<<"\n index2 "<<treeSongIndex.size()<<std::endl;



    // Construir el Ball*-Tree
    int minLeafPoints = 1000;  // Ajusta el valor según lo necesario
    double alpha = 0.1;
    BallTree ballTree(minLeafPoints, alpha);

    auto start = std::chrono::steady_clock::now();
    ballTree.constructBallTree(treeSongIndex);
    auto end = std::chrono::steady_clock::now();

    std::cout << "Construction time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
              << " ns" << std::endl;

    // Imprimir el Ball Tree
    //std::cout << "Ball Tree:" << std::endl;
    //ballTree.printBallTree(ballTree.root);


    //std::cout<<"\nKNN Queries "<<std::endl;
    // Consultar los k vecinos de una canción específica
    //std::string querySong = "Andromeda";
    std::string querySong = "Mercury: Retrograde";
    int k = 5;

    start = std::chrono::steady_clock::now();
    std::vector<std::string> neighbors = ballTree.findNearestNeighbors(querySong, k, treeSongIndex);
    end = std::chrono::steady_clock::now();

    std::cout << "KNN time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
              << " ns" << std::endl;

    // Imprimir los k vecinos encontrados
    std::cout << "\nK= "<<k<<" Nearest Neighbors of <<" << querySong << ":" << ">>"<<std::endl;
    for (const std::string& neighbor : neighbors) {
        std::cout << neighbor << std::endl;
    }

/*
    std::string querySongC = "Andromeda";
    double range = 0.5;  // Rango máximo permitido

    start = std::chrono::steady_clock::now();
    std::vector<std::string> constrainedNeighbors = ballTree.findConstrainedNearestNeighbors(querySongC, range, k, treeSongIndex);
    end = std::chrono::steady_clock::now();

    std::cout << "Constrained NN time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
              << " ns" << std::endl;
              
    std::cout<<"rangR: "<< constrainedNeighbors.size()<<std::endl;

    // Imprimir los vecinos más cercanos restringidos encontrados
    std::cout << "Constrained Nearest Neighbors of " << querySongC << " within range " << range << ":" << std::endl;
    for (const std::string& neighbor : constrainedNeighbors) {
        std::cout << neighbor << std::endl;
    }
 */  
    return 0;
}

