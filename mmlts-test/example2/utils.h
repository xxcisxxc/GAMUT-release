#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdint.h>
#include <stdio.h>

#include "macros.h"

#ifndef MMLT_DEVICE
	#define MMLT_DEVICE __forceinline__ __device__
#endif
#ifndef MMLT_UNROLL
	#define MMLT_UNROLL _Pragma("unroll")
#endif
#ifndef MMLT_LOOP
	#define MMLT_LOOP _Pragma("unroll 1")
#endif

namespace example2 {
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
	float *output;
	float *work;
	float *inputs[Num_Inputs];
	size_problem size_p;
	int size_K;
	coord<2> dim2[Num_Inputs + 1];
	int n_partition;
};

struct kernel_param {
	int problem_m, problem_n, problem_k;
	int size_K, n_partition;
};

struct idx {
	unsigned int block_m, block_n, block_k;
	int warp_m, warp_n;
	int thread_m, thread_n;
	int tot_thr;
};

template <int N>
struct Array
{
	float storage[N];

	MMLT_DEVICE
	void fill(float val)
	{
		MMLT_UNROLL
		for (int i = 0; i < N; i++) {
			storage[i] = val;
		}
	}
};

template <int N>
struct AlignedBuffer
{
	alignas(16) float storage[N];
};
}
/*class Tile
{
private:
	coord base;
	coord offset;
	coord size;
	bool row_major;
	int cur_buffer;

public:
	MMLT_DEVICE
	Tile(coord base_, coord offset_, coord size_, bool major) :
		base(base_), offset(offset_), size(size_), row_major(major), cur_buffer(1)
	{ }

	MMLT_DEVICE
	void add_offset(coord off_)
	{
		offset.x += off_.x;
		offset.y += off_.y;
	}

	MMLT_DEVICE
	void move_base(coord off_)
	{
		base.x += off_.x;
		base.y += off_.y;
	}

	MMLT_DEVICE
	void next_buffer()
	{
		move_base({cur_buffer * size.x, cur_buffer * size.y});
		cur_buffer = -cur_buffer;
	}

	MMLT_DEVICE
	int index()
	{
		return row_major ?
			(base.x + offset.x) * size.y + (base.y + offset.y) :
			(base.x + offset.x) + (base.y + offset.y) * size.x;
	}

	MMLT_DEVICE
	void parallelize()
	{

	}
};*/