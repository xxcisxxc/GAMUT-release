#ifndef _M_ALLOC
#define _M_ALLOC

#include "common.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "device_launch_parameters.h"

/**
 * @brief modified from https://github.com/NVIDIA/cutlass/blob/master/examples/00_basic_gemm/basic_gemm.cu\n
	initialize each element in matrix - gpu part
 * 
 * @param matrix The matrix that need to initialized
 * @param rows The dimension of row
 * @param columns The dimension of column
 * @return __global__ 
 */
__global__ void InitializeMatrix_kernel(float *matrix, int rows, int columns);

/**
 * @brief copied from https://github.com/NVIDIA/cutlass/blob/master/examples/00_basic_gemm/basic_gemm.cu\n
	initialize a matrix - cpu part
 * 
 * @param matrix The matrix that need to initialized
 * @param rows The dimension of row
 * @param columns The dimension of column
 * @return cudaError_t 
 */
cudaError_t InitializeMatrix(float *matrix, int rows, int columns);

/**
 * @brief modified from https://github.com/NVIDIA/cutlass/blob/master/examples/00_basic_gemm/basic_gemm.cu\n
	allocate memory for a matrix
 * 
 * @param matrix The matrix that need to allocated
 * @param rows The dimension of row
 * @param columns The dimension of column
 * @param initialized Whether initilalize or not. Default: true
 * @return cudaError_t 
 */
cudaError_t AllocateMatrix(float **matrix, int rows, int columns, bool initialized = true);

#endif /* _M_ALLOC */