import numpy as np

def load_matrix_from_file(filename):
    """Загружает матрицу из файла, возвращает numpy array."""
    with open(filename, 'r') as f:
        rows, cols = map(int, f.readline().split())
        matrix = []
        for _ in range(rows):
            row = list(map(float, f.readline().split()))
            matrix.append(row)
    return np.array(matrix)

def verify_matrix_multiplication(result_file, matrix_a_file, matrix_b_file, tolerance=1e-6):
    """
    Проверяет умножение матриц.
    Сравнивает результат из result_file с результатом умножения матриц из matrix_a_file и matrix_b_file,
    используя numpy.
    """
    try:
        matrix_a = load_matrix_from_file(matrix_a_file)
        matrix_b = load_matrix_from_file(matrix_b_file)
        result_cpp = load_matrix_from_file(result_file)

        # Вычисляем результат умножения с помощью numpy
        result_numpy = np.dot(matrix_a, matrix_b)

        # Сравниваем результаты
        return np.allclose(result_cpp, result_numpy, atol=tolerance) # atol - допустимая погрешность

    except FileNotFoundError:
        print(f"Ошибка: Один из файлов не найден.")
        return False
    except Exception as e:
        print(f"Ошибка при верификации: {e}")
        return False


def check() -> None:
    counts = [50, 100, 200, 300, 400, 500, 1500]
    for size in counts:
        matrix_a_file = f"/Users/dariatsyganova/Desktop/Parallel_programming/files/1_{size}.txt"
        matrix_b_file = f"/Users/dariatsyganova/Desktop/Parallel_programming/files/2_{size}.txt"
        cpp_result_file = f"/Users/dariatsyganova/Desktop/Parallel_programming/files/result_{size}.txt"

        if verify_matrix_multiplication(cpp_result_file, matrix_a_file, matrix_b_file):
            print(f"Verification for matrix size {size}: PASSED")
        else:
            print(f"Verification for matrix size {size}: FAILED")

if __name__ == "__main__":
    check()