import numpy as np

def generate_matrix_file(filename, size):
    """Генерирует файл с матрицей заданного размера."""
    matrix = np.random.rand(size, size)  # Создаем матрицу случайных чисел
    with open(filename, 'w') as f:
        f.write(f"{size} {size}\n")  # Записываем размер матрицы
        for row in matrix:
            f.write(" ".join(map(str, row)) + "\n")

# Определяем размеры матриц
sizes = [50, 100, 200, 300, 400, 500]

# Генерируем файлы для матриц A и B каждого размера
for size in sizes:
    generate_matrix_file(f"1_{size}.txt", size)  # Матрица A
    generate_matrix_file(f"2_{size}.txt", size)  # Матрица B

print("Файлы матриц успешно созданы.")
