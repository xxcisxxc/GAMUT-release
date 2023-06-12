#include "launch.h"
#include "m_alloc.h"
#include "gpu_timer.h"
#include <cmath>
#include <iostream>
#include <cuda.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

__global__ void naive_kernel(
	float *a, float *b, float *c, int *g,
    const int M, const int N, const int K
)
{
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	int m = blockIdx.y * blockDim.y + threadIdx.y;
	if (m < M && n < N) {
		float psum = 0.0;
		#pragma unroll
		for (int k = 0; k < K; k++) {
			psum += a[m + k * M] * b[n + k * N];
		}
		if (psum < 10000) {
			int off = atomicAdd(g, 1);
			c[off] = psum;
		}
	}
}

cudaError_t naive(float *a, float *b, float *c, int *g,
	const int M, const int N, const int K,
	cudaStream_t stream = nullptr)
{
	const int BM = 32, BN = 32;
	dim3 blockDim(BN, BM);
	dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

	naive_kernel<<<gridDim, blockDim, 0, stream>>>(a, b, c, g, M, N, K);

	return cudaGetLastError();
}

#define D 8192
#define M 8192
#define N 8192

int main()
{
	float *C, *A, *B, *Q;
	int *G;
	
	AllocateMatrix(&C, M, N, false);
	AllocateMatrix(&Q, M, N, false);
	AllocateMatrix((float **)&G, 1, 1, false);
	AllocateMatrix(&A, M, D, true);
	AllocateMatrix(&B, D, N, true);
	cudaDeviceSynchronize();

	*G = 0;
	GpuTimer timer;
	timer.start();
	for (int i = 0; i < 1; i++)
		standard::launch({C, nullptr, {A, B, G}, {M, N, D}, D, {{M, D}, {N, D}, {1, 1}, {M, N}}, 1});
	timer.stop_and_wait();
	std::cout << "Time: " << timer.duration(1) << " ms\n";
	std::cout << *G << std::endl;
			
	std::cout << "Start Naive\n";
	*G = 0;
	timer.start();
	naive(A, B, Q, G, M, N, D);
	timer.stop_and_wait();
	std::cout << "Time: " << timer.duration(1) << " ms\n";
	std::cout << *G << std::endl;

	std::cout << "Start Correctness\n";
	int g = *G;
	for (int i = 0; i < g; i++)
		if (std::abs(C[OFFSET(i, j, P1)] - Q[OFFSET(i, j, P1)]) > 1e-3)
			std::printf("(%d,)\ttarget: %.1f\tcompute: %.1f\n", i, Q[i], C[i]));
}
