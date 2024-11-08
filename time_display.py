import numpy as np
import time
from single_thread import serial_matrix_multiply
from pthreads import parallel_matrix_multiply
from recursive_pthreads import parallel_matrix_multiply_recursive
from strassen_pthreads import parallel_strassen_matrix_multiply
matrix_sizes = [128, 256, 512]  # 可以根据实际情况调整矩阵大小
serial_times = []
parallel_times = []
multithreaded_times = []
strassen_times = []

for size in matrix_sizes:
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    
    # 测量单线程串行时间
    start_time = time.time()
    serial_matrix_multiply(A, B)
    serial_times.append(time.time() - start_time)
    print(f"Serial multiplication time for {size}x{size} matrix: {serial_times[-1]:.4f} seconds")
    
    # 测量多线程并行时间
    start_time = time.time()
    parallel_matrix_multiply(A, B)
    parallel_times.append(time.time() - start_time)
    print(f"Parallel multiplication time for {size}x{size} matrix: {parallel_times[-1]:.4f} seconds")

    #测量多线程分治并行时间
    """
    C = np.zeros((size, size)) 
    start_time = time.time()
    multithreaded_result = parallel_matrix_multiply_recursive(C, A, B, size)
    multithreaded_times.append(time.time() - start_time)
    print(f"Multithreaded divide-and-conquer time for {size}x{size} matrix: {multithreaded_times[-1]:.4f} seconds")
    """
    #测量strassen算法的时间
    """
    start_time = time.time()
    parallel_strassen_matrix_multiply(A, B)
    parallel_times.append(time.time() - start_time)
    print(f"Parallel Strassen time for {size}x{size} matrix: {parallel_times[-1]:.4f} seconds")
    """

