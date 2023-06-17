#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdint.h>
#include <stdio.h>

#include "macros.h"
#include "constants.h"

namespace standard {
struct size_problem {
	int kRow, kColumn, kDepth;
};

template <int N>
struct coord;

template <>
struct coord<1>
{
	int x;
};

template <>
struct coord<2>
{
	int x, y;
};

template <>
struct coord<3>
{
	int x, y, z;
};

struct Params {
	void *output;
	void *work;
	void *inputs[Num_Inputs];
	size_problem size_p;
	int size_K;
	coord<2> dim2[Num_Inputs+1];
	int n_partition;
};

struct idx {
	int block_m, block_n, block_k;
	int warp_m, warp_n;
	int thread_m, thread_n;
	int tot_thr;
};

template <typename T, int N>
struct Array
{
	T storage[N];

	MMLT_DEVICE
	void fill(T val)
	{
		MMLT_UNROLL
		for (int i = 0; i < N; i++) {
			storage[i] = val;
		}
	}

	MMLT_DEVICE
	int size()
	{
		return N;
	}
};

template <typename T, int N>
struct AlignedBuffer
{
	alignas(16) T storage[N];

	MMLT_DEVICE
	int size()
	{
		return N;
	}
};
}
