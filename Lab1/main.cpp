#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <stdexcept>

using namespace std;
using namespace std::chrono;

vector<vector<double>> load_matrix(const string& filename, int& rows, int& cols) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Не удалось открыть файл: " + filename);
    }

    file >> rows >> cols;
    if (rows <= 0 || cols <= 0) {
        throw runtime_error("Некорректные размеры матрицы в файле: " + filename);
    }

    vector<vector<double>> matrix(rows, vector<double>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file >> matrix[i][j];
        }
    }

    file.close();
    return matrix;
}

void save_matrix(const string& filename, const vector<vector<double>>& matrix) {
    ofstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Не удалось открыть файл для записи: " + filename);
    }

    int rows = matrix.size();
    int cols = matrix.empty() ? 0 : matrix[0].size();

    file << rows << " " << cols << endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file << matrix[i][j] << " ";
        }
        file << endl;
    }

    file.close();
}

vector<vector<double>> multiply_matrices(const vector<vector<double>>& a, const vector<vector<double>>& b) {
    int rows_a = a.size();
    int cols_a = a[0].size();
    int rows_b = b.size();
    int cols_b = b[0].size();

    if (cols_a != rows_b) {
        throw runtime_error("Невозможно умножить матрицы: несовместимые размеры.");
    }

    vector<vector<double>> result(rows_a, vector<double>(cols_b, 0.0));

    for (int i = 0; i < rows_a; ++i) {
        for (int j = 0; j < cols_b; ++j) {
            for (int k = 0; k < cols_a; ++k) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    return result;
}


int main() {
    string matrix_a_file, matrix_b_file, result_file;
    int a_rows, a_cols, b_rows, b_cols;
    vector<vector<double>> matrix_a, matrix_b, result_matrix;
    high_resolution_clock::time_point start, end;
    duration<double> duration_sec;

    //matrix_a_file = "/Users/dariatsyganova/Desktop/Parallel_prog/input_matrix_a.txt";
    //matrix_b_file = "/Users/dariatsyganova/Desktop/Parallel_prog/input_matrix_b.txt";
    //result_file = "/Users/dariatsyganova/Desktop/Parallel_prog/output_matrix.txt";
    std::vector<int> counts = {50, 100, 200, 300, 400, 500};
    std::string base_path = "/Users/dariatsyganova/Desktop/Parallel_programming/files/";

    for (int size : counts) {
        std::string matrix_a_file = base_path + "1_" + std::to_string(size) + ".txt";
        std::string matrix_b_file = base_path + "2_" + std::to_string(size) + ".txt";
        std::string result_file = base_path + "result_" + std::to_string(size) + ".txt";
        try {
            matrix_a = load_matrix(matrix_a_file, a_rows, a_cols);
            matrix_b = load_matrix(matrix_b_file, b_rows, b_cols);

            start = high_resolution_clock::now();

            result_matrix = multiply_matrices(matrix_a, matrix_b);

            end = high_resolution_clock::now();
            duration_sec = duration_cast<duration<double>>(end - start);

            save_matrix(result_file, result_matrix);

            cout << "Матрицы успешно умножены. Размер = " << size << endl;
            cout << "Время выполнения: " << duration_sec.count() << " сек." << endl;
            cout << "Объем задачи (A: " << a_rows << "x" << a_cols << ", B: " << b_rows << "x" << b_cols << ")" << endl;

        } catch (const exception& e) {
            cerr << "Ошибка: " << e.what() << endl;
            return 1;
        }
    }
    return 0;
}