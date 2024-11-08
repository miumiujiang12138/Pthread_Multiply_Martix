import numpy as np
import time

def serial_matrix_multiply(A, B):
    n = A.shape[0]                  # 获取A元组的第一个元素，也就是矩阵A的行数
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C
