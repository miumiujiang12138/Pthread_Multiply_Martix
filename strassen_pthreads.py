import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor

def add_matrix(A, B):
    return A + B

def subtract_matrix(A, B):
    return A - B

def strassen_matrix_multiply(A, B):
    if A.shape[0] == 1:
        return A * B
    else:
        n = A.shape[0] // 2

        # 分割矩阵
        A11, A12, A21, A22 = A[:n, :n], A[:n, n:], A[n:, :n], A[n:, n:]
        B11, B12, B21, B22 = B[:n, :n], B[:n, n:], B[n:, :n], B[n:, n:]

        # 生成辅助矩阵
        M1 = strassen_matrix_multiply(add_matrix(A11, A22), add_matrix(B11, B22))
        M2 = strassen_matrix_multiply(add_matrix(A21, A22), B11)
        M3 = strassen_matrix_multiply(A11, subtract_matrix(B12, B22))
        M4 = strassen_matrix_multiply(A22, subtract_matrix(B21, B11))
        M5 = strassen_matrix_multiply(add_matrix(A11, A12), B22)
        M6 = strassen_matrix_multiply(subtract_matrix(A21, A11), add_matrix(B11, B12))
        M7 = strassen_matrix_multiply(subtract_matrix(A12, A22), add_matrix(B21, B22))

        # 合并结果
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6

        # 组合结果矩阵
        C = np.vstack([np.hstack([C11, C12]), np.hstack([C21, C22])])
        return C

def parallel_strassen_matrix_multiply(A, B):
    if A.shape[0] == 1:
        return A * B
    else:
        n = A.shape[0] // 2
        A11, A12, A21, A22 = A[:n, :n], A[:n, n:], A[n:, :n], A[n:, n:]
        B11, B12, B21, B22 = B[:n, :n], B[:n, n:], B[n:, :n], B[n:, n:]

        # 使用线程池来计算辅助矩阵
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(strassen_matrix_multiply, add_matrix(A11, A22), add_matrix(B11, B22)),
                executor.submit(strassen_matrix_multiply, add_matrix(A21, A22), B11),
                executor.submit(strassen_matrix_multiply, A11, subtract_matrix(B12, B22)),
                executor.submit(strassen_matrix_multiply, A22, subtract_matrix(B21, B11)),
                executor.submit(strassen_matrix_multiply, add_matrix(A11, A12), B22),
                executor.submit(strassen_matrix_multiply, subtract_matrix(A21, A11), add_matrix(B11, B12)),
                executor.submit(strassen_matrix_multiply, subtract_matrix(A12, A22), add_matrix(B21, B22)),
            ]
            M1, M2, M3, M4, M5, M6, M7 = [f.result() for f in futures]

        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6

        C = np.vstack([np.hstack([C11, C12]), np.hstack([C21, C22])])
        return C

# 测试并比较运行时间
matrix_sizes = [64, 128, 256]  # 选择适合的 n*n 矩阵大小
serial_times = []
parallel_times = []

for size in matrix_sizes:
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    
    # 串行 Strassen 算法时间
    start_time = time.time()
    strassen_matrix_multiply(A, B)
    serial_times.append(time.time() - start_time)
    print(f"Serial Strassen time for {size}x{size} matrix: {serial_times[-1]:.4f} seconds")
    
    # 并行 Strassen 算法时间
    start_time = time.time()
    parallel_strassen_matrix_multiply(A, B)
    parallel_times.append(time.time() - start_time)
    print(f"Parallel Strassen time for {size}x{size} matrix: {parallel_times[-1]:.4f} seconds")

# 可视化结果
import matplotlib.pyplot as plt

plt.plot(matrix_sizes, serial_times, label='Serial Strassen')
plt.plot(matrix_sizes, parallel_times, label='Parallel Strassen')
plt.xlabel('Matrix Size (n x n)')
plt.ylabel('Time (seconds)')
plt.title('Runtime Comparison of Serial and Parallel Strassen Matrix Multiplication')
plt.legend()
plt.show()
