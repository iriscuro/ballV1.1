//Compilar en linux: g++ -std=c++11 -I /usr/include/eigen3 pca_test.cpp -o pca_test

#include <iostream>
#include <fstream>
#include <random>
#include "pca.h"
#include "datos.h" 

#include <experimental/filesystem>

int main() {
    InvertedIndex diccSongIndex;
    InvertedIndex treeSongIndex;

    fillInvertedIndices("songs_final.csv", diccSongIndex, treeSongIndex);

    // Obtener los datos numéricos del segundo índice invertido
    std::vector<Data> numericData;
    for (const auto& entry : treeSongIndex) {
        const Data& features = entry.second;
        numericData.push_back(features);
    }

    // Crear la matriz pca_data_matrix con los datos numéricos
    Eigen::MatrixXf pca_data_matrix(numericData.size(), numericData[0].size());
    for (size_t i = 0; i < numericData.size(); i++) {
        for (size_t j = 0; j < numericData[i].size(); j++) {
            pca_data_matrix(i, j) = numericData[i][j];
        }
    }

    // Realizar el cálculo de PCA
    pca_t<float> pca;
    pca.set_input(pca_data_matrix);
    pca.compute();

    // Obtener el primer eigenvector
    Eigen::Matrix<float, Eigen::Dynamic, 1> first_eigenvector = pca.get_eigen_vectors().col(0);

    // Imprimir el primer eigenvector
    std::cout << "First Eigenvector:\n";
    for (int i = 0; i < first_eigenvector.rows(); ++i) {
        std::cout << first_eigenvector(i) << " ";
    }
    std::cout << std::endl;

    return 0;
}


