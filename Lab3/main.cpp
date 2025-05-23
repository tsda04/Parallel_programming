#include <mpi.h>
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    string filenameA, filenameB, filenameRes;
    int a_rows, a_cols, b_rows, b_cols;

    vector<vector<double>> A, B, C;

    if (rank == 0) {
        string base_path = "/Users/dariatsyganova/Desktop/Parallel_programming/files/";
        vector<int> counts = {50, 100, 200, 300, 400, 500};

        for (int size_mat : counts) {
            string a_file = base_path + "1_" + to_string(size_mat) + ".txt";
            string b_file = base_path + "2_" + to_string(size_mat) + ".txt";
            string res_file = base_path + "result_" + to_string(size_mat) + ".txt";

            try {
                A = load_matrix(a_file, a_rows, a_cols);
                B = load_matrix(b_file, b_rows, b_cols);

                if (a_cols != b_rows) {
                    cerr << "Несовместимые размеры матриц для размера " << size_mat << endl;
                    continue;
                }

                // Распределение данных для всех процессов
                int rows_per_process = a_rows / size;
                int remainder = a_rows % size;

                vector<int> sendcounts(size);
                vector<int> displs(size);
                int offset = 0;
                for (int i = 0; i < size; ++i) {
                    sendcounts[i] = (i < remainder) ? (rows_per_process + 1) * a_cols : rows_per_process * a_cols;
                    displs[i] = offset;
                    offset += sendcounts[i];
                }

                vector<double> A_flat(a_rows * a_cols);
                for (int i = 0; i < a_rows; ++i)
                    for (int j = 0; j < a_cols; ++j)
                        A_flat[i * a_cols + j] = A[i][j];

                vector<double> A_sub;
                if (rank == 0) {
                    A_sub.resize(sendcounts[0]);
                } else {
                    A_sub.resize(sendcounts[rank]);
                }

                MPI_Scatterv(A_flat.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                             A_sub.data(), sendcounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

                // Передача матрицы B целиком (так как её нужно знать всем)
                vector<double> B_flat(b_rows * b_cols);
                if (rank == 0) {
                    for (int i = 0; i < b_rows; ++i)
                        for (int j = 0; j < b_cols; ++j)
                            B_flat[i * b_cols + j] = B[i][j];
                }

                MPI_Bcast(B_flat.data(), b_rows * b_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

                // Каждый процесс формирует свою часть матрицы A
                int local_rows = sendcounts[rank] / a_cols;
                vector<vector<double>> local_A(local_rows, vector<double>(a_cols));
                for (int i = 0; i < local_rows; ++i)
                    for (int j = 0; j < a_cols; ++j)
                        local_A[i][j] = A_sub[i * a_cols + j];

                // Все процессы умножают свои части
                // для полного результата нужно объединить части
                // Для упрощения, все вычисляют свою часть результирующей матрицы

                vector<vector<double>> local_B(b_rows, vector<double>(b_cols));
                for (int i = 0; i < b_rows; ++i)
                    for (int j = 0; j < b_cols; ++j)
                        local_B[i][j] = B_flat[i * b_cols + j];

                auto start = high_resolution_clock::now();
                vector<vector<double>> local_C = multiply_matrices(local_A, local_B);

                int result_rows = local_C.size();
                vector<int> recvcounts(size);
                vector<int> displs_res(size);
                int total_rows = 0;

                for (int i = 0; i < size; ++i) {
                    int rows_i = (i < remainder) ? (rows_per_process + 1) : rows_per_process;
                    recvcounts[i] = rows_i * b_cols;
                    displs_res[i] = total_rows * b_cols;
                    total_rows += rows_i;
                }

                vector<double> result_flat(total_rows * b_cols);
                // Преобразуем локальный результат в плоский массив
                vector<double> local_result_flat(local_C.size() * b_cols);
                for (int i = 0; i < local_C.size(); ++i)
                    for (int j = 0; j < b_cols; ++j)
                        local_result_flat[i * b_cols + j] = local_C[i][j];

                MPI_Gatherv(local_result_flat.data(), local_result_flat.size(), MPI_DOUBLE,
                            result_flat.data(), recvcounts.data(), displs_res.data(), MPI_DOUBLE,
                            0, MPI_COMM_WORLD);
                if (rank == 0) {
                    vector<vector<double>> result_matrix(total_rows, vector<double>(b_cols));
                    int index = 0;
                    for (int i = 0; i < total_rows; ++i) {
                        for (int j = 0; j < b_cols; ++j) {
                            result_matrix[i][j] = result_flat[i * b_cols + j];
                        }
                    }
                    auto end = high_resolution_clock::now();
                    auto duration_sec = duration_cast<duration<double>>(end - start);
                    save_matrix(res_file, result_matrix);
                    cout << "Матрицы успешно умножены. Размер = " << size_mat << endl;
                    cout << "Время выполнения: " << duration_sec.count() << " сек." << endl;

                }

            } catch (const exception& e) {
                cerr << "Ошибка: " << e.what() << endl;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
