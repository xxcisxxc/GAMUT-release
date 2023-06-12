#pragma once

namespace example6 {
__device__ void min_heapify(volatile float *arr, int ind, int heap_size);

__device__ void build_min_heap(volatile float *arr, int heap_size);

__device__ void min_heap_push_pop(volatile float *arr, float val, int heap_size);
}