import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor

def add_matrix(A, B):
    return A + B

def subtract_matrix(A, B):
    return A - B

def parallel_matrix_multiply_recursive(C, A, B, n):
    if n == 1:
        # 基础情况，1x1矩阵相乘
        C[0, 0] = A[0, 0] * B[0, 0]
    else:
        half_size = n // 2

        # 分割矩阵
        A11, A12, A21, A22 = A[:half_size, :half_size], A[:half_size, half_size:], A[half_size:, :half_size], A[half_size:, half_size:]
        B11, B12, B21, B22 = B[:half_size, :half_size], B[:half_size, half_size:], B[half_size:, :half_size], B[half_size:, half_size:]
        
        # 创建结果子矩阵
        C11 = C[:half_size, :half_size]
        C12 = C[:half_size, half_size:]
        C21 = C[half_size:, :half_size]
        C22 = C[half_size:, half_size:]

        # 创建临时矩阵来存储中间计算结果
        T11, T12, T21, T22 = np.zeros((half_size, half_size)), np.zeros((half_size, half_size)), np.zeros((half_size, half_size)), np.zeros((half_size, half_size))

        # 使用多线程执行递归调用
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(parallel_matrix_multiply_recursive, T11, A11, B11, half_size),
                executor.submit(parallel_matrix_multiply_recursive, T12, A12, B21, half_size),
                executor.submit(parallel_matrix_multiply_recursive, T21, A11, B12, half_size),
                executor.submit(parallel_matrix_multiply_recursive, T22, A12, B22, half_size),
                executor.submit(parallel_matrix_multiply_recursive, C11, T11, T12, half_size),
                executor.submit(parallel_matrix_multiply_recursive, C12, T21, T22, half_size),
                executor.submit(parallel_matrix_multiply_recursive, T11, A21, B11, half_size),
                executor.submit(parallel_matrix_multiply_recursive, T12, A22, B21, half_size),
            ]
            # 等待所有计算完成
            for future in futures:
                future.result()

        # 合并计算结果
        C[:half_size, :half_size] = T11 + T12
        C[:half_size, half_size:] = T21 + T22
        C[half_size:, :half_size] = T11 + T12
        C[half_size:, half_size:] = T21 + T22