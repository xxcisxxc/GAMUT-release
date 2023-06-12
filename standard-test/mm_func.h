#ifndef _MM_FUNC
#define _MM_FUNC

#include "common.h"
#include "cublas_v2.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_splitk_parallel.h"
#include <stdexcept>

/**
 * @brief modified from https://solarianprogrammer.com/2012/05/31/matrix-multiplication-cuda-cublas-curand-thrust/\n
	Matrix multiplication (C = A * B) using CUBLAS
 * 
 * @param handle cublas handle. Need to initialize and remeber to destroy afterwards
 * @param A first matrix
 * @param B second matrix
 * @param C return matrix
 * @param m A - m * k
 * @param n B - k * n
 * @param k C - m * n
 * @return cudaError_t
 */
cudaError_t cublas_mmul(
	cublasHandle_t &handle,
	const float *A,
	const float *B,
	float *C,
	const int m,
	const int n,
	const int k
);

/**
 * @brief modified from https://github.com/NVIDIA/cutlass/blob/master/examples/00_basic_gemm/basic_gemm.cu\n
	Matrix multiplication (C = A * B) using CUTLASS
 * 
 * @param A first matrix
 * @param B second matrix
 * @param C return matrix
 * @param m A - m * k
 * @param n B - k * n
 * @param k C - m * n
 * @return cudaError_t
 */
cudaError_t cutlass_mmul(
	const float *A,
	const float *B,
	float *C,
	const int m,
	const int n,
	const int k
);

/**
 * @brief modified from https://github.com/NVIDIA/cutlass/blob/master/examples/00_basic_gemm/basic_gemm.cu\n
	Matrix multiplication (C = A * B) using CUTLASS, split k dimension
 * 
 * @param workspace the workspace initialized by split_workspace
 * @param A first matrix
 * @param B second matrix
 * @param C return matrix
 * @param m A - m * k
 * @param n B - k * n
 * @param k C - m * n
 * @param split_size split every ... elements
 * @return cudaError_t
 */
cudaError_t cutlass_mmul_split(
	uint8_t *workspace,
	const float *A,
	const float *B,
	float *C,
	const int m,
	const int n,
	const int k,
	const int split_size
);

cudaError_t split_workspace(
	const float *A,
	const float *B,
	float *C,
	const int m,
	const int n,
	const int k,
	const int split_size,
	uint8_t **ptr
);

#endif /* _MM_FUNC */