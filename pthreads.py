import numpy as np
from joblib import Parallel, delayed

def parallel_matrix_multiply(A, B):
    n = A.shape[0]
    C = np.zeros((n, n))
    
    def compute_row(i):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    
    Parallel(n_jobs=-1)(delayed(compute_row)(i) for i in range(n))
    return C
