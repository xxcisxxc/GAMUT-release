#pragma once

#include "utils.h"

namespace standard {

struct heap_algo {
__device__ void min_heapify(volatile float *arr, int ind, int heap_size);

__device__ void build(volatile float *arr, int heap_size);

__device__ void put(volatile float *arr, float val, int heap_size);
};
}