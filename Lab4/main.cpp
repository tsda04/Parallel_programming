#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <random>
#include <ctime>
#include <sstream>
#include <chrono>
#include <stdexcept>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

// CUDA ядро для умножения матриц
__global__ void matrixMulKernel(int* A, int* B, int* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

vector<vector<int>> generate_matrix(int size, int seed) {
    vector<vector<int>> matrix(size, vector<int>(size));

    mt19937 engine(static_cast<unsigned int>(time(nullptr)) + seed);
    uniform_int_distribution<int> dist(0, 100); // Ограничиваем значения для уменьшения переполнения

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix[i][j] = dist(engine);
        }
    }

    return matrix;
}

void write_to_file(const vector<vector<int>>& matrix, const string& path) {
    ofstream out(path);
    if (!out.is_open()) {
        throw runtime_error("Не удалось открыть файл для записи: " + path);
    }

    for (const auto& row : matrix) {
        for (int val : row) {
            out << val << " ";
        }
        out << endl;
    }
}

vector<vector<int>> read_from_file(const string& path) {
    ifstream in(path);
    if (!in.is_open()) {
        throw runtime_error("Не удалось открыть файл: " + path);
    }

    vector<vector<int>> matrix;
    string line;

    while (getline(in, line)) {
        istringstream iss(line);
        vector<int> row;
        int value;
        while (iss >> value) {
            row.push_back(value);
        }
        if (!row.empty()) {
            matrix.push_back(row);
        }
    }

    if (matrix.empty()) {
        throw runtime_error("Файл пуст или содержит некорректные данные: " + path);
    }

    return matrix;
}

vector<vector<int>> multiply_matrices_cuda(const vector<vector<int>>& matrix1,
                                           const vector<vector<int>>& matrix2) {
    size_t N = matrix1.size();
    if (N == 0 || matrix2.size() != N || matrix2[0].size() != N) {
        throw runtime_error("Некорректные размеры матриц для умножения");
    }

    // Проверка доступной памяти на GPU
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t required_mem = 3 * N * N * sizeof(int); // A, B и C матрицы
    if (required_mem > free_mem) {
        throw runtime_error("Недостаточно памяти на GPU. Требуется: " +
                            to_string(required_mem / (1024 * 1024)) +
                            " MB, доступно: " +
                            to_string(free_mem / (1024 * 1024)) + " MB");
    }

    // Подготовка плоских массивов
    vector<int> flatA(N * N);
    vector<int> flatB(N * N);
    vector<int> flatC(N * N, 0);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            flatA[i * N + j] = matrix1[i][j];
            flatB[i * N + j] = matrix2[j][i]; // Транспонирование B для лучшего доступа к памяти
        }
    }

    // Выделение памяти на GPU
    int *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaError_t err;

    err = cudaMalloc(&d_A, N * N * sizeof(int));
    if (err != cudaSuccess) {
        throw runtime_error("Ошибка выделения памяти для матрицы A: " + string(cudaGetErrorString(err)));
    }

    err = cudaMalloc(&d_B, N * N * sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_A);
        throw runtime_error("Ошибка выделения памяти для матрицы B: " + string(cudaGetErrorString(err)));
    }

    err = cudaMalloc(&d_C, N * N * sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        throw runtime_error("Ошибка выделения памяти для матрицы C: " + string(cudaGetErrorString(err)));
    }

    // Копирование данных на GPU
    err = cudaMemcpy(d_A, flatA.data(), N * N * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        throw runtime_error("Ошибка копирования матрицы A на GPU: " + string(cudaGetErrorString(err)));
    }

    err = cudaMemcpy(d_B, flatB.data(), N * N * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        throw runtime_error("Ошибка копирования матрицы B на GPU: " + string(cudaGetErrorString(err)));
    }

    // Настройка размеров блоков и сетки
    const int BLOCK_SIZE = 16;
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        throw runtime_error("Ошибка выполнения ядра CUDA: " + string(cudaGetErrorString(err)));
    }

    // Копирование результата обратно
    err = cudaMemcpy(flatC.data(), d_C, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        throw runtime_error("Ошибка копирования результата с GPU: " + string(cudaGetErrorString(err)));
    }

    // Освобождение памяти GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Преобразование результата обратно в матрицу
    vector<vector<int>> result(N, vector<int>(N));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            result[i][j] = flatC[i * N + j];
        }
    }

    return result;
}

int main(int argc, char** argv) {
    vector<int> counts = {50, 100, 200, 300, 400, 500, 1500};
    vector<double> times;

    try {
        // Генерация новых матриц
        for (const auto& count : counts) {
            for (int i = 1; i < 3; ++i) {
                vector<vector<int>> matrix = generate_matrix(count, i);
                string path = to_string(i) + "_" + to_string(count) + ".txt";
                write_to_file(matrix, path);
            }
        }

        for (size_t i = 0; i < counts.size(); ++i) {
            int count = counts[i];
            cout << "Умножение матриц размера " << count << "x" << count << endl;

            string path_1 = "1_" + to_string(count) + ".txt";
            string path_2 = "2_" + to_string(count) + ".txt";
            string result_path = "result_" + to_string(count) + ".txt";

            auto start_time = steady_clock::now();

            try {
                vector<vector<int>> matrix_1 = read_from_file(path_1);
                vector<vector<int>> matrix_2 = read_from_file(path_2);
                vector<vector<int>> result = multiply_matrices_cuda(matrix_1, matrix_2);
                write_to_file(result, result_path);
            } catch (const exception& e) {
                cerr << "Ошибка для размера " << count << ": " << e.what() << endl;
                times.push_back(-1.0); // Помечаем ошибку
                continue;
            }

            auto end_time = steady_clock::now();
            double duration = duration_cast<milliseconds>(end_time - start_time).count();
            times.push_back(duration);
            cout << "Время: " << duration << " мс" << endl;
        }

    } catch (const exception& e) {
        cerr << "Критическая ошибка: " << e.what() << endl;
        return 1;
    }

    return 0;
}