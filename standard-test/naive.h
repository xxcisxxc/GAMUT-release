#include <cuda_runtime.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

/**
 * @brief Copied from https://github.com/nicolaswilde/cuda-sgemm.git
 * 
 */
__global__ void naive_kernel(
	float *a, float *b, float *c,
    const int M, const int N, const int K
)
{
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	int m = blockIdx.y * blockDim.y + threadIdx.y;
	if (m < M && n < N) {
		float psum = 0.0;
		#pragma unroll
		for (int k = 0; k < K; k++) {
		    psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
		}
		c[OFFSET(m, n, N)] = psum;
	}
}

cudaError_t naive(float *a, float *b, float *c,
	const int M, const int N, const int K,
	cudaStream_t stream = nullptr)
{
	const int BM = 32, BN = 32;
	dim3 blockDim(BN, BM);
	dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

	naive_kernel<<<gridDim, blockDim, 0, stream>>>(a, b, c, M, N, K);

	return cudaGetLastError();
}