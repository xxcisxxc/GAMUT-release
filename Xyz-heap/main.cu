#include "launch.h"
#include "m_alloc.h"
#include "gpu_timer.h"
#include <cmath>
#include <iostream>
#include <cuda.h>
#include "heap.h"

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

__global__ void naive_kernel(
	float *a, float *b, volatile float *c, int *g, int *mu,
    const int M, const int N, const int K
)
{
	standard::heap_algo algo;
	standard::spin_lock lock = {mu};
	__shared__ float local_heap[32 * 32];
	const int Heap_Size = 128;
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	int m = blockIdx.y * blockDim.y + threadIdx.y;
	if (m < M && n < N) {
		float psum = 0.0;
		#pragma unroll
		for (int k = 0; k < K; k++) {
			psum += a[m + k * M] * b[n + k * N];
		}
		int off = OFFSET(threadIdx.x, threadIdx.y, 32);
		local_heap[off] = psum;
		if (off != 0) {
			goto end;
		}
		algo.build(local_heap, Heap_Size);
		#pragma unroll
		for (int i = Heap_Size; i < 32 * 32; i++) {
			if (local_heap[0] < local_heap[i]) {
				algo.put(local_heap, psum, Heap_Size);
			}
		}
		lock.lock();
		off = atomicAdd(g, 1);
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
					algo.put(c, val, Heap_Size);
				}
			}
		}
		lock.unlock();
	}
end:
	__syncthreads();
}

cudaError_t naive(float *a, float *b, float *c, int *g, int *mu,
	const int M, const int N, const int K,
	cudaStream_t stream = nullptr)
{
	const int BM = 32, BN = 32;
	dim3 blockDim(BN, BM);
	dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

	naive_kernel<<<gridDim, blockDim, 0, stream>>>(a, b, c, g, mu, M, N, K);

	return cudaGetLastError();
}

#define D 128
#define M 128
#define N 128

void min_heapify_cpu(float *arr, int ind, int heap_size)
{
	int left = 2 * ind + 1;
	int right = 2 * ind + 2;
	int smallest = 0;
	if (left < heap_size && arr[left] < arr[ind])
		smallest = left;
	else
		smallest = ind;
	if (right < heap_size && arr[right] < arr[smallest])
		smallest = right;
	if (smallest != ind) {
		float temp = arr[ind];
		arr[ind] = arr[smallest];
		arr[smallest] = temp;
		min_heapify_cpu(arr, smallest, heap_size);
	}
}
void build_min_heap_cpu(float *arr, int heap_size)
{
	for (int i = heap_size / 2 - 1; i >= 0; i--)
		min_heapify_cpu(arr, i, heap_size);
}
void min_heap_push_pop_cpu(float *arr, float val, int heap_size)
{
	arr[0] = val;
	min_heapify_cpu(arr, 0, heap_size);
}
float heappop(float *arr, int heap_size)
{
	float ret = arr[0];
	arr[0] = arr[heap_size];
	min_heapify_cpu(arr, 0, heap_size);
	return ret;
}

int main()
{
	float *C, *A, *B;
	int *G, *mu;
	
	AllocateMatrix(&C, M, N, false);
	AllocateMatrix((float **)&G, 1, 1, false);
	AllocateMatrix((float **)&mu, 1, 1, false);
	AllocateMatrix(&A, M, D, true);
	AllocateMatrix(&B, D, N, true);
	cudaDeviceSynchronize();

	*G = 0;
	*mu = 0;
	GpuTimer timer;
	timer.start();
	for (int i = 0; i < 1; i++)
		standard::launch({C, nullptr, {A, B, G, mu}, {M, N, D}, D, {{M, D}, {N, D}, {1, 1}, {1, 1}, {M, N}}, 1});
	timer.stop_and_wait();
	std::cout << "Time: " << timer.duration(1) << " ms\n";

	/* std::cout << "Start Naive\n";
	 * *G = 0;
	 * *mu = 0;
	 * timer.start();
	 * naive(A, B, C, G, mu, M, N, D);
	 * timer.stop_and_wait();
	 * std::cout << "Time: " << timer.duration(1) << " ms\n";
	 */

	std::cout << "Start Correctness\n";
	float heap[128];
	int cur = 0;
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			float c = 0.f;
			for (int k = 0; k < D; k++) {
				c += A[i + k * M] * B[k * N + j];
			}
			if (cur < 128) {
				heap[cur] = c;
			}
			cur++;
			if (cur == 128) {
				build_min_heap_cpu(heap, 128);
			} else if (c > heap[0]) {
				min_heap_push_pop_cpu(heap, c, 128);
			}
		}
	}
	for (int i = 128 - 1; i >= 0; i--) {
		float c0 = heappop(heap, i);
		float c1 = heappop(C, i);
		if (std::abs(c0 - c1) > 1e-3)
			std::printf("(%d, %d)\ttarget: %.3f\tcompute: %.3f\n", i, i, c0, c1);
	}
}
