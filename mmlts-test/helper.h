#pragma once

#include <cuda_runtime.h>
#include <curand.h>

const int P0 = 128, P1 = 128;

__global__ void setList_kernel(float *A, float *B, const int M, const int N)
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

cudaError_t setList(float *A, float *B, const int M, const int N)
{
	setList_kernel<<<2, 1>>>(A, B, M, N);
	return cudaGetLastError();
}