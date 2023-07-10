#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <cstdio>

typedef std::vector<double> Data;
typedef std::unordered_map<std::string, Data> InvertedIndex;

InvertedIndex invertedIndex;  // Declaración global de la variable invertedIndex


void printData(const Data& data) {
    for (const auto& value : data) {
        std::cout << value << " ";
    }
    std::cout << std::endl;
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

void printInvertedIndex(const InvertedIndex& invertedIndex) {
    std::cout<<std::endl;
    //std::cout<<"\n<<<<<Print starded>>>>>"<<std::endl;
    for (const auto& entry : invertedIndex) {
        const std::string& songName = entry.first;
        const Data& features = entry.second;

        std::cout << "Song: {" << songName << "}" <<std::endl;
        std::cout << "Features: ";
        for (const auto& feature : features) {
            std::cout << feature << " ";
        }
        std::cout << std::endl << std::endl;
    }
    std::cout<<"Print finished..."<<std::endl;
}

void printInvertedIndex(const InvertedIndex& invertedIndex, int numRecords) {
    std::cout<<std::endl;
    //std::cout<<"\n<<<<<Print starded>>>>>"<<std::endl;
    int count = 0;
    for (const auto& entry : invertedIndex) {
        if (count >= numRecords) {
            break;
        }
        const std::string& songName = entry.first;
        const Data& features = entry.second;

        std::cout << "Song: {" << songName << "}"  <<std::endl;
        std::cout << "Features: ";
        for (const auto& feature : features) {
            std::cout << feature << " ";
        }
        std::cout << std::endl << std::endl;

        count++;
    }
    std::cout<<"Print finished..."<<std::endl;
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
            features.push_back(std::stod(fields[i]));
        }

        // Concatenate the remaining fields as the song name
        std::string songName;
        for (size_t i = 13; i < fields.size(); i++) {
            songName += fields[i];
            if (i < fields.size() - 1) {
                songName += ",";
                //std::cout<<"-"<<songName<<" -"<<std::endl;
            }
            std::cout<<"-"<<songName<<"-";
        }
        std::cout<<std::endl;

        //std::cout<<"-"<<songName<<" -"<<std::endl;
        // Add the data to the inverted indices
        originalIndex[songName] = features;
        treeIndex[songName] = features;
    }
}

void fillInvertedIndicesP(const std::string& filename, InvertedIndex& originalIndex,  int max) {
    std::cout<<"FILL..."<<std::endl;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "Error: Failed to open the file.\n";
        return;
    }

    std::string line;
    std::getline(file, line); // Read and discard the first header line
    int cant=1;

    while (std::getline(file, line) && cant<=max) {
        // Split the line into fields using comma as the delimiter
        std::vector<std::string> fields = split(line, ',');

        if (fields.size() < 14) {
            std::cout << "Error: Insufficient number of fields.\n";
            continue;
        }

        // Get the numeric features
        Data features;
        for (size_t i = 0; i < 13; i++) {
            features.push_back(std::stod(fields[i]));
        }

        // Concatenate the remaining fields as the song name
        std::string songName;
        for (size_t i = 13; i < fields.size(); i++) {
            songName += fields[i];
            if (i < fields.size() - 1) {
                songName += ",";
                //std::cout<<"-"<<songName<<" -"<<std::endl;
            }
            //std::cout<<"-"<<songName<<"-";
        }
        std::cout<<"-"<<songName<<"-";
        std::cout<<std::endl;

        //std::cout<<"-"<<songName<<" -"<<std::endl;
        // Add the data to the inverted indices
        originalIndex[songName] = features; 

        cant++;
    }std::cout<<"Cant. :"<<cant<<std::endl;
}

void fillInvertedIndices(const std::string& filename, InvertedIndex& treeIndex) {
    std::cout<<"FILL..."<<std::endl;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "Error: Failed to open the file.\n";
        return;
    }

    std::string line;
    std::getline(file, line); // Read and discard the first header line
    int cant=0;
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
            features.push_back(std::stod(fields[i]));
        }

        // Concatenate the remaining fields as the song name
        std::string songName;
        for (size_t i = 13; i < fields.size(); i++) {
            songName += fields[i];
            if (i < fields.size() - 1) {
                songName += ",";
            }
        }
        //std::cout<<"-"<<songName<<"-";std::cout<<std::endl;
        // Add the data to the inverted indices
        treeIndex[songName] = features;
        cant++;

    }std::cout<<"Cant. :"<<cant<<std::endl;
}

void printNumericData(const InvertedIndex &invertedIndex, const std::string &songName) {
  auto it = invertedIndex.find(songName);
  if (it != invertedIndex.end()) {
    const Data &features = it->second;

    std::cout << "Numeric data for song: {" << songName << "}" << std::endl;
    for (const auto &feature : features) {
      std::cout << feature << " ";
    }
    std::cout << std::endl << std::endl;
  } else {
    std::cout << "Song not found: {" << songName << "}" << std::endl;
  }
}


void saveResultsToFile(const std::string& filename, const std::vector<std::string>& results) {
    std::ofstream outputFile(filename);

    if (outputFile.is_open()) {
        for (const std::string& result : results) {
            outputFile << result << std::endl;
        }

        outputFile.close();
        std::cout << "Results saved to " << filename << std::endl;
    } else {
        std::cout << "Error opening the file: " << filename << std::endl;
    }
}


/*
int main() {
    InvertedIndex diccSongIndex;
    InvertedIndex treeSongIndex;

    fillInvertedIndices("songs_final.csv", diccSongIndex, treeSongIndex);

    // Después de llenar los índices invertidos
    printInvertedIndex(diccSongIndex, 10);
    printInvertedIndex(treeSongIndex, 10);


    return 0;
}
*/
