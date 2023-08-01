#include "launch.h"
#include "m_alloc.h"
#include "gpu_timer.h"
#include <cmath>
#include <iostream>
#include <cuda.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

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

#define D 512
#define M 512
#define N 512

int main()
{
	float *C, *A, *B, *Q;
	
	AllocateMatrix(&C, M, N, false);
	AllocateMatrix(&Q, M, N, false);
	AllocateMatrix(&A, M, D, true);
	AllocateMatrix(&B, D, N, true);
	cudaDeviceSynchronize();

	GpuTimer timer;
	timer.start();
	for (int i = 0; i < 1; i++)
		standard::launch({C, nullptr, {A, B}, {M, N, D}, D, {{D, M}, {N, D}}, 1});
	timer.stop_and_wait();
	std::cout << "Time: " << timer.duration(1) << " ms\n";
			
	std::cout << "Start Naive\n";
	timer.start();
	naive(A, B, Q, M, N, D);
	timer.stop_and_wait();
	std::cout << "Time: " << timer.duration(1) << " ms\n";

	std::cout << "Start Correctness\n";
	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
			if (std::abs(Q[OFFSET(i, j, N)] - C[OFFSET(i, j, N)]) > 1e-3)
				std::printf("(%d, %d)\ttarget: %.1f\tcompute: %.1f\n", i, j, Q[OFFSET(i, j, N)], C[i * N + j]);
}
