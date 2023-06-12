#pragma once

#include <cuda_runtime.h>
#include "example6/heap.h"

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

/**
 * @brief Copied from https://github.com/nicolaswilde/cuda-sgemm.git
 * 
 */
namespace standard {
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
}

namespace query2 {
__global__ void naive_kernel(
	float *a, float *b, float *c, float *d,
    const int M, const int N, const int K
)
{
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	int m = blockIdx.y * blockDim.y + threadIdx.y;
	if (m < M && n < N) {
		float psum = 0.0;
		#pragma unroll
		for (int k = 0; k < K; k++) {
			float multi = a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
			float thres = d[n];
			psum += multi + float(multi > thres) * (multi - thres);
		}
		c[OFFSET(m, n, N)] = psum;
	}
}

cudaError_t naive(float *a, float *b, float *c, float *d,
	const int M, const int N, const int K,
	cudaStream_t stream = nullptr)
{
	const int BM = 32, BN = 32;
	dim3 blockDim(BN, BM);
	dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

	naive_kernel<<<gridDim, blockDim, 0, stream>>>(a, b, c, d, M, N, K);

	return cudaGetLastError();
}
}

namespace example1 {
__global__ void naive_kernel(
	float *a, float *b, float *c, float *e, float *f,
    const int M, const int N, const int K, const int P0, const int P1
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
		c += OFFSET(int(e[m]), int(f[n]), P1);
		atomicAdd(c, psum);
	}
}

cudaError_t naive(float *a, float *b, float *c, float *e, float *f,
	const int M, const int N, const int K, const int P0, const int P1,
	cudaStream_t stream = nullptr)
{
	const int BM = 32, BN = 32;
	dim3 blockDim(BN, BM);
	dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

	naive_kernel<<<gridDim, blockDim, 0, stream>>>(a, b, c, e, f, M, N, K, P0, P1);

	return cudaGetLastError();
}
}

namespace example2 {
__global__ void naive_kernel(
	float *a, float *b, float *c, float *e, float *f,
    const int M, const int N, const int K, const int P0, const int P1
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
		psum = float(psum > 1000.);
		c += OFFSET(int(e[m]), int(f[n]), P1);
		atomicAdd(c, psum);
	}
}

cudaError_t naive(float *a, float *b, float *c, float *e, float *f,
	const int M, const int N, const int K, const int P0, const int P1,
	cudaStream_t stream = nullptr)
{
	const int BM = 32, BN = 32;
	dim3 blockDim(BN, BM);
	dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

	naive_kernel<<<gridDim, blockDim, 0, stream>>>(a, b, c, e, f, M, N, K, P0, P1);

	return cudaGetLastError();
}
}

namespace example3 {
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
		atomicAdd(c, psum);
	}
}

cudaError_t naive(float *a, float *b, float *c, float *e, float *f,
	const int M, const int N, const int K, const int P0, const int P1,
	cudaStream_t stream = nullptr)
{
	int P = P0 < P1 ? P0 : P1;
	const int BM = 32, BN = 32;
	dim3 blockDim(BN, BM);
	dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

	cudaStream_t streams[P];
	int cur_E = 0, cur_F = 0;
	for (int i = 0; i < P; i++) {
		int last_E = cur_E, last_F = cur_F;
		while (cur_E < M && int(e[cur_E]) == i) {
			cur_E++;
		}
		while (cur_F < N && int(f[cur_F]) == i) {
			cur_F++;
		}
		int M = cur_E-last_E, N = cur_F-last_F;
		cudaStreamCreate(streams + i);
		naive_kernel<<<gridDim, blockDim, 0, streams[i]>>>(a+last_E, b+last_F, c+i, M, N, K);
	}

	return cudaGetLastError();
}
}

namespace example4 {
__global__ void naive_kernel(
	float *a, float *b, float *c, float *g,
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
		if (psum < 100) {
			int off = int(atomicAdd(g, 1.));
			c[off] = psum;
		}
	}
}

cudaError_t naive(float *a, float *b, float *c, float *g,
	const int M, const int N, const int K,
	cudaStream_t stream = nullptr)
{
	const int BM = 32, BN = 32;
	dim3 blockDim(BN, BM);
	dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

	naive_kernel<<<gridDim, blockDim, 0, stream>>>(a, b, c, g, M, N, K);

	return cudaGetLastError();
}
}

namespace example5 {
__global__ void naive_kernel(
	float *a, float *b, float *c, float *g,
    const int M, const int N, const int K
)
{
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	int m = blockIdx.y * blockDim.y + threadIdx.y;
	if (m < M && n < N) {
		float psum = 0.0;
		#pragma unroll
		for (int k = 0; k < K; k++) {
			float max_ij = max(a[OFFSET(m, k, K)], b[OFFSET(k, n, N)]);
			psum += max_ij * max_ij;
		}
		if (psum > 0.5) {
			int off = int(atomicAdd(g, 1.));
			c[off] = psum;
		}
	}
}

cudaError_t naive(float *a, float *b, float *c, float *g,
	const int M, const int N, const int K,
	cudaStream_t stream = nullptr)
{
	const int BM = 32, BN = 32;
	dim3 blockDim(BN, BM);
	dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

	naive_kernel<<<gridDim, blockDim, 0, stream>>>(a, b, c, g, M, N, K);

	return cudaGetLastError();
}
}

namespace example6 {
__global__ void naive_kernel(
	float *a, float *b, volatile float *c, float *g, int *mu,
    const int M, const int N, const int K
)
{
	__shared__ float local_heap[32 * 32];
	const int Heap_Size = 128;
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	int m = blockIdx.y * blockDim.y + threadIdx.y;
	if (m < M && n < N) {
		float psum = 0.0;
		#pragma unroll
		for (int k = 0; k < K; k++) {
			psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
		}
		int off = OFFSET(threadIdx.x, threadIdx.y, 32);
		local_heap[off] = psum;
		if (off != 0) {
			__syncthreads();
			return;
		}
		build_min_heap(local_heap, Heap_Size);
		#pragma unroll
		for (int i = Heap_Size; i < 32 * 32; i++) {
			if (local_heap[0] < local_heap[i]) {
				min_heap_push_pop(local_heap, psum, Heap_Size);
			}
		}
		while (atomicCAS(mu, 0, 1) != 0); // Lock();
		off = atomicAdd(g, 1.);
		if (off == 0) {
			#pragma unroll
			for (int i = 0; i < Heap_Size; i++) {
				c[i] = local_heap[i];
			}
		} else {
			#pragma unroll
			for (int i = 0; i < Heap_Size; i++) {
				float val = local_heap[i];
				if (c[0] < val) {
					min_heap_push_pop(local_heap, val, Heap_Size);
				}
			}
		}
		atomicExch(mu ,0); // Unlock();
	}
}

cudaError_t naive(float *a, float *b, float *c, float *g, int *mu,
	const int M, const int N, const int K,
	cudaStream_t stream = nullptr)
{
	const int BM = 32, BN = 32;
	dim3 blockDim(BN, BM);
	dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

	naive_kernel<<<gridDim, blockDim, 0, stream>>>(a, b, c, g, mu, M, N, K);

	return cudaGetLastError();
}
}