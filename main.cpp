#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <stdexcept>
#include <numeric>
#include <chrono>
#include <ctime>

bool allow_duplicate = false;

void load_data(const std::string& csv_path, std::vector<std::vector<double>>& data, std::vector<int>& labels, std::vector<std::string>& dim_label) {
    std::ifstream file(csv_path);
    std::string line;
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::istringstream iss(line);
        std::string value;
        while (std::getline(iss, value, ';')) {
            try {
                row.push_back(std::stod(value));
            } catch (const std::invalid_argument& e) {
                std::cout << "Invalid argument: " << value << std::endl;
            }
            //row.push_back(std::stod(value));
        }
        if (!row.empty()) {
            data.push_back(row);
            labels.push_back(static_cast<int>(row.back())); // Convertir el valor a int
            row.pop_back();
        }
    }
}



struct Ball {
    std::vector<double> center;
    double radius;
    std::vector<std::vector<double>> points;
    Ball* left;
    Ball* right;

    Ball(const std::vector<double>& center, double radius, const std::vector<std::vector<double>>& points, Ball* left, Ball* right)
        : center(center), radius(radius), points(points), left(left), right(right) {}
};

class BallTree {
public:
    BallTree(const std::vector<std::vector<double>>& values, const std::vector<int>& labels, int train_num)
     : values(values), labels(labels), train_num(train_num) {
        if (values.empty()) {
            throw std::runtime_error("Data for Ball-Tree must not be empty.");
        }
        root = build_BallTree();
        KNN_max_now_dist = std::numeric_limits<double>::infinity();
        KNN_result = std::vector<std::pair<std::vector<double>, double>>(1, std::make_pair(std::vector<double>(), KNN_max_now_dist));
    }

    void search_KNN(const std::vector<double>& target, int K) {
        if (root == nullptr) {
            throw std::runtime_error("Ball-Tree must not be empty.");
        }
        if (K > values.size()) {
            throw std::invalid_argument("K in KNN must be greater than the length of data.");
        }
        if (target.size() != root->center.size()) {
            throw std::invalid_argument("Target must have the same dimension as the data.");
        }
        KNN_result = std::vector<std::pair<std::vector<double>, double>>(1, std::make_pair(std::vector<double>(), KNN_max_now_dist));
        nums = 0;
        search_KNN_core(root, target, K);
    }
    std::vector<std::pair<std::vector<double>, double>> get_KNN_result() const {
        return KNN_result;
    }

private:
    std::vector<std::vector<double>> values;
    std::vector<int> labels;
    int train_num;
    Ball* root;
    double KNN_max_now_dist;
    std::vector<std::pair<std::vector<double>, double>> KNN_result;
    int nums;

    Ball* build_BallTree() {
        std::vector<std::vector<double>> data;
        for (size_t i = 0; i < values.size(); i++) {
            std::vector<double> point = values[i];
            point.push_back(static_cast<double>(labels[i]));
            data.push_back(point);
        }
        return buildBallTree_core(data);
    }

    double dist(const std::vector<double>& point1, const std::vector<double>& point2) {
        double sum = 0.0;
        for (size_t i = 0; i < point1.size(); i++) {
            sum += std::pow(point1[i] - point2[i], 2);
        }
        return std::sqrt(sum);
    }

    Ball* buildBallTree_core(std::vector<std::vector<double>>& data) {
        if (data.empty()) {
            return nullptr;
        }
        if (data.size() == 1) {
            return new Ball(data[0], 0.001, data, nullptr, nullptr);
        }
        std::vector<std::vector<double>> data_disloc(data.begin() + 1, data.end());
        data_disloc.push_back(data[0]);

        // Corrección: Calcular la suma de los elementos de data_disloc correctamente
        double sum = 0.0;
        for (const auto& vector : data_disloc) {
            sum += std::accumulate(vector.begin(), vector.end(), 0.0);
        }

        if (sum == 0) {
            return new Ball(data[0], 1e-100, data, nullptr, nullptr);
        }
        std::vector<double> cur_center(data[0].size() - 1, 0.0);
        for (size_t i = 0; i < cur_center.size(); i++) {
            double sum = 0.0;
            for (size_t j = 0; j < data.size(); j++) {
                sum += data[j][i];
            }
            cur_center[i] = sum / data.size();
        }
        std::vector<double> dists_with_center(data.size(), 0.0);
        for (size_t i = 0; i < data.size(); i++) {
            dists_with_center[i] = dist(cur_center, data[i]);
        }
        size_t max_dist_index = std::max_element(dists_with_center.begin(), dists_with_center.end()) - dists_with_center.begin();
        double max_dist = dists_with_center[max_dist_index];
        Ball* root = new Ball(cur_center, max_dist, data, nullptr, nullptr);
        std::vector<double> point1 = data[max_dist_index];
        std::vector<double> dists_with_point1(data.size(), 0.0);
        for (size_t i = 0; i < data.size(); i++) {
            dists_with_point1[i] = dist(point1, data[i]);
        }
        size_t max_dist_index2 = std::max_element(dists_with_point1.begin(), dists_with_point1.end()) - dists_with_point1.begin();
        std::vector<double> point2 = data[max_dist_index2];
        std::vector<double> dists_with_point2(data.size(), 0.0);
        for (size_t i = 0; i < data.size(); i++) {
            dists_with_point2[i] = dist(point2, data[i]);
        }
        std::vector<bool> assign_point1(data.size(), false);
        for (size_t i = 0; i < data.size(); i++) {
            assign_point1[i] = dists_with_point1[i] < dists_with_point2[i];
        }
        std::vector<std::vector<double>> data_left, data_right;
        for (size_t i = 0; i < data.size(); i++) {
            if (assign_point1[i]) {
                data_left.push_back(data[i]);
            } else {
                data_right.push_back(data[i]);
            }
        }
        root->left = buildBallTree_core(data_left);
        root->right = buildBallTree_core(data_right);
        return root;
    }


    void search_KNN_core(Ball* root_ball, const std::vector<double>& target, int K) {
        if (root_ball == nullptr) {
            return;
        }
        if (root_ball->left == nullptr && root_ball->right == nullptr) {
            insert(root_ball, target, K);
        }
        if (dist(root_ball->center, target) <= root_ball->radius + KNN_result[0].second) {
            search_KNN_core(root_ball->left, target, K);
            search_KNN_core(root_ball->right, target, K);
        }
    }

    void insert(Ball* root_ball, const std::vector<double>& target, int K) {
        for (const std::vector<double>& node : root_ball->points) {
            nums++;
            bool is_duplicate = std::any_of(KNN_result.begin(), KNN_result.end(), [&](const std::pair<std::vector<double>, double>& item) {
                return dist(node, item.first) < 1e-4 && std::abs(node.back() - item.first.back()) < 1e-4;
            });
            if (is_duplicate && !allow_duplicate) {
                continue;
            }
            double distance = dist(target, node);
            if (KNN_result.size() < K) {
                KNN_result.push_back(std::make_pair(node, distance));
            } else if (distance < KNN_result.front().second) {
                KNN_result.erase(KNN_result.begin());
                KNN_result.push_back(std::make_pair(node, distance));
            }
            std::sort(KNN_result.begin(), KNN_result.end(), [](const std::pair<std::vector<double>, double>& a, const std::pair<std::vector<double>, double>& b) {
                return a.second > b.second;
            });
        }
    }
};

int main() {
    std::string csv_path = "winequality-white.csv";
    std::vector<std::vector<double>> data;
    std::vector<int> labels;
    std::vector<std::string> dim_label;
    load_data(csv_path, data, labels, dim_label);

    double split_rate = 0.8;
    int K = 5;
    int train_num = static_cast<int>(data.size() * split_rate);
    std::cout << "train_num: " << train_num << std::endl;
    std::clock_t start1 = std::clock();
    BallTree ball_tree(data, labels, train_num);
    std::clock_t end1 = std::clock();

    double diff_all = 0;
    int accuracy = 0;
    double search_all_time = 0;
    int calu_dist_nums = 0;
    for (size_t index = train_num; index < data.size(); ++index) {
        std::clock_t start2 = std::clock();
        ball_tree.search_KNN(data[index], K);
        std::clock_t end2 = std::clock();
        search_all_time += static_cast<double>(end2 - start2) / CLOCKS_PER_SEC;

        double pred_label = 0.0;
        std::unordered_map<double, int> label_count;
        for (const auto& node : ball_tree.get_KNN_result()) {
            double label = node.first.back();
            label_count[label]++;
        }
        for (const auto& pair : label_count) {
            if (pair.second > label_count[pred_label]) {
                pred_label = pair.first;
            }
        }
        diff_all += std::abs(labels[index] - pred_label);
        if (labels[index] == pred_label) {
            accuracy++;
        }
        std::cout << "accuracy: " << static_cast<double>(accuracy) / (index - train_num + 1) << std::endl;
        std::cout << "Total: " << (index - train_num + 1) << ", MSE: " << (diff_all / (index - train_num + 1)) << "    " << labels[index] << "--->" << pred_label << std::endl;
    }

    std::cout << "BallTree construction time: " << static_cast<double>(end1 - start1) / CLOCKS_PER_SEC << std::endl;
    std::cout << "Average search time: " << (search_all_time / (data.size() - train_num)) << std::endl;
    std::cout << "Average calculation count: " << (calu_dist_nums / (data.size() - train_num)) << std::endl;

    return 0;
}