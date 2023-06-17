#include "launch.h"
#include "m_alloc.h"
#include "gpu_timer.h"
#include <cmath>
#include <iostream>
#include <cuda.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

__global__ void naive_kernel(
	float *a, float *b, float *c, int *e, int *f,
    const int M, const int N, const int K, const int P0, const int P1
)
{
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	int m = blockIdx.y * blockDim.y + threadIdx.y;
	if (m < M && n < N) {
		float psum = 0.0;
		#pragma unroll
		for (int k = 0; k < K; k++) {
			psum += a[m + k * M] * b[k * N + n];
		}
		c += e[m] * P1 + f[n];
		if (e[m] == 125 && f[n] == 125)
			printf("%f\n", psum);
		atomicAdd(c, psum);
	}
}

cudaError_t naive(float *a, float *b, float *c, int *e, int *f,
	const int M, const int N, const int K, const int P0, const int P1,
	cudaStream_t stream = nullptr)
{
	const int BM = 32, BN = 32;
	dim3 blockDim(BN, BM);
	dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

	naive_kernel<<<gridDim, blockDim, 0, stream>>>(a, b, c, e, f, M, N, K, P0, P1);

	return cudaGetLastError();
}

const int P0 = 128, P1 = 128;

__global__ void setList_kernel(int *A, int *B, const int M, const int N)
{
	curandState_t state;
	curand_init(blockIdx.x, 0, 0, &state);
	if (blockIdx.x == 0) {
		int arr[P0] = {0};
		for (int i = 0; i < M; i++) {
			arr[curand(&state) % P0]++;
		}
		int k = 0;
		for (int i = 0; i < P0; i++) {
			for (int j = 0; j < arr[i]; j++) {
				A[k] = i;
				k += 1;
			}
		}
	} else {
		int arr[P1] = {0};
		for (int i = 0; i < N; i++) {
			arr[curand(&state) % P1]++;
		}
		int k = 0;
		for (int i = 0; i < P1; i++) {
			for (int j = 0; j < arr[i]; j++) {
				B[k] = i;
				k += 1;
			}
		}
	}
}

cudaError_t setList(int *A, int *B, const int M, const int N)
{
	setList_kernel<<<2, 1>>>(A, B, M, N);
	return cudaGetLastError();
}

#define D 512
#define M 512
#define N 512

int main()
{
	float *C, *A, *B, *Q;
	int *E, *F;
	
	AllocateMatrix(&C, P0, P1, false);
	AllocateMatrix(&Q, P0, P1, false);
	AllocateMatrix(&A, M, D, true);
	AllocateMatrix(&B, D, N, true);
	AllocateMatrix((float **)&E, 1, M, false);
	AllocateMatrix((float **)&F, 1, N, false);
	setList(E, F, M, N);
	cudaDeviceSynchronize();

	GpuTimer timer;
	timer.start();
	for (int i = 0; i < 1; i++)
		standard::launch({C, nullptr, {A, B, E, F}, {M, N, D}, D, {{D, M}, {N, D}, {M, 1}, {N, 1}, {P0, P1}}, 1});
	timer.stop_and_wait();
	std::cout << "Time: " << timer.duration(1) << " ms\n";
			
	std::cout << "Start Naive\n";
	timer.start();
	naive(A, B, Q, E, F, M, N, D, P0, P1);
	timer.stop_and_wait();
	std::cout << "Time: " << timer.duration(1) << " ms\n";

	std::cout << "Start Correctness\n";
	for (int i = 0; i < P0; i++)
		for (int j = 0; j < P1; j++)
			if (std::abs(C[OFFSET(i, j, P1)] - Q[OFFSET(i, j, P1)]) > 1e-3)
				std::printf("(%d, %d)\ttarget: %.1f\tcompute: %.1f\n", i, j, Q[OFFSET(i, j, P1)], C[OFFSET(i, j, P1)]);
}
