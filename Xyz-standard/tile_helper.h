/**
 * @file tile_helper.h
 * @author Xincheng Xie (xie.xincheng@columbia.edu)
 * @brief Helper Library
 * @version 0.1
 * @date 2023-01-23
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#pragma once

#include "utils.h"
#include <float.h>
#include <algorithm>

namespace standard {

template <typename T, int thread_size, int warp_size, int block_size, int block_depth = Depth_Block>
struct SliceLoad {
	Array<T, thread_size> *rsmem;
	Array<T, thread_size> *reg;

	struct idx_p {
		int warp_x;
		int thread_x;
	};

	MMLT_DEVICE
	SliceLoad(Array<T, thread_size> *reg_, T *smem_, idx_p id) : reg(reg_)
	{
		rsmem = (Array<T, thread_size> *)(smem_ + warp_size * id.warp_x + thread_size * id.thread_x);
	}

	MMLT_DEVICE
	void load()
	{
		reg[0] = rsmem[0];
	}

	MMLT_DEVICE
	void next()
	{
		rsmem += block_size / thread_size;
	}

	MMLT_DEVICE
	void reset()
	{
		rsmem -= (block_size / thread_size) * block_depth;
	}
};

template <typename T, int thread_size, int warp_size>
struct SliceLoadVec
{
	struct idx_p {
		int block_x, warp_x, thread_x;
	};

	MMLT_DEVICE
	SliceLoadVec(Array<T, thread_size> *reg_, T *gmem_, coord<1> size_, idx_p id)
	{
		T *gmem = gmem_ + id.block_x * thread_size;
		Array<T, thread_size> *gmem_v = (Array<T, thread_size> *)(gmem + warp_size * id.warp_x + thread_size * id.thread_x);
		reg_[0] = gmem_v[0];
	}
};

template <typename T, int block_size, int block_depth = Depth_Block,
int thread_count = Count_Thr_Block_Tot,
int load_iter = block_size * block_depth / thread_count, int load_interval = thread_count / block_size>
struct TileLoad {
	T *gmem;
	T *gsmem;

	coord<2> size;

	//bool within_x;
	//int cur_y;

	struct shared_memory {
		AlignedBuffer<T, block_size * block_depth> tile;
	};

	struct idx_p {
		int block_x, block_k;
		int tot_thr;
	};

	MMLT_DEVICE
	TileLoad(T *gmem_, shared_memory &smem_, coord<3> size_, idx_p id)
	{
		size.x = size_.x;
		size.y = size_.y;

		coord<2> addr = {id.block_x * block_size + id.tot_thr % block_size, size_.z * id.block_k + id.tot_thr / block_size};

		gmem = gmem_ + addr.x + addr.y * size.x;
		gsmem = smem_.tile.storage + id.tot_thr;

		//within_x = addr_a.x < size.x;
		//cur_y = addr_a.y;
	}

	MMLT_DEVICE
	void load()
	{
		//float *frag_ptr = frag.storage;
		MMLT_UNROLL
		for (int i = 0; i < load_iter; i++) {
			//if (within_x && cur_y < size.y) {
				//frag_ptr[i] = gmem[i * load_interval * size.x];
				gsmem[i * load_interval * block_size] = gmem[i * load_interval * size.x];
			//} else {
			//	gsmem[i * load_interval * block_size] = 0.f;
			//}
			//cur_y += load_interval;
		}
	}

	MMLT_DEVICE
	void next()
	{
		gmem += block_depth * size.x;
	}
};

template <typename T, int block_size, int block_depth = Depth_Block, 
int thread_count = Count_Thr_Block_Tot,
int load_iter = block_size * block_depth / thread_count, int load_interval = thread_count / block_depth>
struct TileLoadX {
	T *gmem;
	float *gsmem;

	coord<2> size;

	//bool within_x;
	//int cur_y;

	struct shared_memory {
		AlignedBuffer<T, block_size * block_depth> tile;
	};

	struct idx_p {
		int block_x, block_k;
		int tot_thr;
	};

	MMLT_DEVICE
	TileLoadX(T *gmem_, shared_memory &smem_, coord<3> size_, idx_p id)
	{
		size.x = size_.x;
		size.y = size_.y;

		coord<2> addr = {id.block_x * block_size + id.tot_thr / block_depth, size_.z * id.block_k + id.tot_thr % block_depth};

		gmem = gmem_ + addr.x * size.x + addr.y;
		gsmem = smem_.tile.storage + (id.tot_thr / block_depth) + (id.tot_thr % block_depth) * block_size;

		//within_x = addr_a.x < size.x;
		//cur_y = addr_a.y;
	}

	MMLT_DEVICE
	void load()
	{
		MMLT_UNROLL
		for (int i = 0; i < load_iter; i++) {
			//if (within_x && cur_y < size.y) {
				gsmem[i * load_interval] = gmem[i * load_interval * size.x];
			//} else {
			//	gsmem[i * load_interval * block_size] = 0.f;
			//}
			//cur_y += load_interval;
		}
	}

	MMLT_DEVICE
	void next()
	{
		gmem += block_depth;
	}
};

template <typename T, int thread_size_x = Size_Thr_Row, int thread_size_y = Size_Thr_Col,
int block_count_y = Count_Thr_Block_Row,
int warp_count_x = Count_Thr_Warp_Row, int warp_count_y = Count_Thr_Warp_Col>
struct SliceStore
{
	Array<T, thread_size_y> *rsmem;
	Array<T, thread_size_y> *reg;

	MMLT_DEVICE
	SliceStore(T *smem_, Array<T, thread_size_y> *reg_, idx &id) : reg(reg_)
	{
		rsmem = (Array<T, thread_size_y> *)smem_ + (id.warp_m * warp_count_x + id.thread_m) * block_count_y + (id.warp_n * warp_count_y + id.thread_n);
	}

	MMLT_DEVICE
	void store()
	{
		rsmem[0] = reg[0];
	}

	MMLT_DEVICE
	void next()
	{
		++reg;
	}
};

template <typename T, int thread_size_x = Size_Thr_Row, int thread_size_y = Size_Thr_Col,
int block_size_x = Size_Block_Row, int block_size_y = Size_Block_Col,
int block_count_x = Count_Thr_Block_Row,
int thread_count = Count_Thr_Block_Tot,
int store_iter = (block_count_x * block_size_y / thread_count), int store_interval = thread_count / block_size_y>
struct TileStore
{
	T *gmem;
	T *gsmem;

	coord<2> size;

	struct shared_memory {
		AlignedBuffer<T, block_count_x * block_size_y> tile;
	};

	//bool exceed_y;
	//int cur_x;

	MMLT_DEVICE
	TileStore(T *gmem_, shared_memory &smem_, coord<2> size_, idx &id) : size(size_)
	{
		coord<2> addr = {block_size_x * id.block_m + (id.tot_thr / block_size_y) * thread_size_x, block_size_y * id.block_n + id.tot_thr % block_size_y};
		gmem = gmem_ + id.block_k * (size.x * size.y) + addr.x * size.y + addr.y;
		gsmem = smem_.tile.storage + id.tot_thr;

		//exceed_y = addr.y >= size.y;
		//cur_x = addr.x;
	}

	MMLT_DEVICE
	void store()
	{
		//int cur = cur_x;
		MMLT_UNROLL
		for (int i = 0; i < store_iter; i++) {
			//if (exceed_y || cur >= size.x) {
			//	break;
			//}
			gmem[i * store_interval * thread_size_x * size.y] = gsmem[i * store_interval * block_size_y];
			//cur += Store_Interval * thread_size_x;
		}
	}

	MMLT_DEVICE
	void next()
	{
		gmem += size.y;
		//cur_x += 1;
	}
};

template <typename T, int thread_size, int warp_size, int block_size>
struct TileLoadVec
{
	struct idx_p {
		int block_x, warp_x, thread_x;
	};

	MMLT_DEVICE
	TileLoadVec(T *gmem_, T *smem_, coord<1> size_, idx_p id)
	{
		T *gmem = gmem_ + id.block_x * block_size;
		Array<T, thread_size> *gmem_v = (Array<T, thread_size> *)(gmem + warp_size * id.warp_x + thread_size * id.thread_x);
		Array<T, thread_size> *smem_v = (Array<T, thread_size> *)(smem_ + warp_size * id.warp_x + thread_size * id.thread_x);
		smem_v[0] = gmem_v[0];
		__syncthreads();
	}
};

template <typename T>
using AtomicOp = T(T *, T);

template <typename T, AtomicOp<T> atomicOp, int thread_size_x = Size_Thr_Row, int thread_size_y = Size_Thr_Col,
int block_size_x = Size_Block_Row, int block_size_y = Size_Block_Col,
int block_count_x = Count_Thr_Block_Row,
int thread_count = Count_Thr_Block_Tot,
int store_iter = (block_count_x * block_size_y / thread_count), int store_interval = thread_count / block_size_y>
struct TileStoreAgg
{
	T *gmem;
	T *gsmem;
	int *ind_x, *ind_y;

	coord<2> size;
	//coord<2> size_p;

	struct shared_memory {
		AlignedBuffer<T, block_count_x * block_size_y> tile;
		Array<int, block_size_x> ind_x;
		Array<int, block_size_y> ind_y;
	};

	//bool exceed_y;
	//int cur_x;

	MMLT_DEVICE
	TileStoreAgg(T *gmem_, shared_memory &smem_, coord<2> size_p_, coord<2> size_, idx &id) : size(size_)
	{
		coord<2> addr = {(id.tot_thr / block_size_y) * thread_size_x, id.tot_thr % block_size_y};
		ind_x = smem_.ind_x.storage + addr.x;
		ind_y = smem_.ind_y.storage + addr.y;
		gmem = gmem_ + id.block_k * (size.x * size.y) + ind_y[0];
		gsmem = smem_.tile.storage + id.tot_thr;

		//exceed_y = addr.y >= size.y;
		//cur_x = (id.tot_thr / block_size_y) * thread_size_x;
	}

	MMLT_DEVICE
	void store()
	{
		//int cur = cur_x;
		MMLT_UNROLL
		for (int i = 0; i < store_iter; i++) {
			//if (exceed_y || cur >= size.x) {
			//	break;
			//}
			atomicOp(gmem + ind_x[i * store_interval * thread_size_x] * size.y, gsmem[i * store_interval * block_size_y]);
			//cur += store_interval * thread_size_x;
		}
	}

	MMLT_DEVICE
	void next()
	{
		ind_x += 1;
		//cur_x += 1;
	}
};

template <typename T>
using SelectOp = bool(T);

template <typename T, SelectOp<T> selectOp, int thread_size_y = Size_Thr_Col>
struct SliceStoreSel
{
	int *rsmem;
	Array<float, thread_size_y> *reg;

	MMLT_DEVICE
	SliceStoreSel(int *smem_, Array<T, thread_size_y> *reg_, int thread_idx) : reg(reg_)
	{
		rsmem = smem_ + thread_idx + 1;
	}

	MMLT_DEVICE
	void store()
	{
		int count = 0;
		MMLT_UNROLL
		for (int i = 0; i < thread_size_y; i++) {
			count += selectOp(reg[0].storage[i]);
		}
		rsmem[0] = count;
	}

	MMLT_DEVICE
	void next()
	{
		++reg;
	}
};

template <typename T, SelectOp<T> selectOp, int thread_size_y = Size_Thr_Col,
int block_count_x = Count_Thr_Block_Row, int block_count_y = Count_Thr_Block_Col>
struct TileStoreSel
{
	T *gmem;
	int *rsmem;
	Array<T, thread_size_y> *reg;

	int *gcounter, *lcounter;

	struct shared_memory {
		Array<int, block_count_x * block_count_y + 1> tile;
		int counter;
	};

	MMLT_DEVICE
	TileStoreSel(T *gmem_, shared_memory &smem_, int *gcounter_, Array<T, thread_size_y> *reg_, int thread_idx) :
		gmem(gmem_), gcounter(gcounter_), reg(reg_)
	{
		lcounter = &(smem_.counter);
		rsmem = smem_.tile.storage + thread_idx;
	}

	MMLT_DEVICE
	void store()
	{
		if (threadIdx.x == 0) {
			rsmem[0] = 0;
			MMLT_UNROLL
			for (int i = 1; i < block_count_x * block_count_y + 1; i++) {
				rsmem[i] += rsmem[i-1];
			}
			lcounter[0] = atomicAdd(gcounter, rsmem[block_count_x * block_count_y]);
		}
		__syncthreads();

		float *cur_gmem = gmem + lcounter[0] + rsmem[0];
		MMLT_UNROLL
		for (int i = 0; i < thread_size_y; i++) {
			if (selectOp(reg[0].storage[i])) {
				cur_gmem[0] = reg[0].storage[i];
				cur_gmem++;
			}
		}
	}

	MMLT_DEVICE
	void next()
	{}
};

struct spin_lock {
	int *mutex;

	MMLT_DEVICE
	void lock() {
		while (atomicCAS(mutex, 0, 1) != 0);
	}

	MMLT_DEVICE
	void unlock() {
		atomicExch(mutex ,0);
	}
};

template <typename T>
using CondOp = bool(T, T);

template<typename T, typename Algorithm, CondOp<T> condOp, int data_size, int thread_count = Count_Thr_Block_Tot,
int thread_size_x = Size_Thr_Row, int block_size_y = Size_Block_Col, int block_count_x = Count_Thr_Block_Row,
int local_size = std::min(data_size, block_count_x * block_size_y),
int global_size = ((data_size + local_size - 1) / local_size) * local_size>
struct TileStoreUnlinear
{
	bool sinit;
	int inters;
	T *smem, *data;
	volatile T *gmem;
	int *ginit;
	Algorithm algo;
	spin_lock mu;

	struct shared_memory {
		AlignedBuffer<T, block_count_x * block_size_y> tile;
		Array<T, local_size> data;
	};

	MMLT_DEVICE
	TileStoreUnlinear(shared_memory &smem_, T *gmem_, int *gmutex_, int *ginit_) : sinit(false), mu{gmutex_}, inters(0), ginit(ginit_), gmem(gmem_)
	{
		smem = smem_.tile.storage;
		data = smem_.data.storage;
	}

	MMLT_DEVICE
	void store()
	{
		if (threadIdx.x != 0) {
			return;
		}
		if (!sinit) {
			MMLT_UNROLL
			for (int i = 0; i < local_size; i++) {
				data[i] = smem[i];
			}
			sinit = true;
			algo.build(data, local_size);
			MMLT_UNROLL
			for (int i = local_size; i < block_count_x * block_size_y; i++) {
				T val = smem[i];
				if (condOp(data[0], val)) {
					algo.put(data, val, local_size);
				}
			}
		} else {
			MMLT_UNROLL
			for (int i = 0; i < block_count_x * block_size_y; i++) {
				T val = smem[i];
				if (condOp(data[0], val)) {
					algo.put(data, val, local_size);
				}
			}
		}
	}

	MMLT_DEVICE
	void next()
	{
		if (threadIdx.x != 0) {
			return;
		}
		inters++;
		if (inters == thread_size_x) {
			mu.lock();
			int init = int(atomicAdd(ginit, local_size));
			if (init == 0) {
				MMLT_UNROLL
				for (int i = 0; i < local_size; i++) {
					gmem[init + i] = data[i];
				}
				/*if (local_size < data_size && init + local_size >= data_size) {
					algo.build(gmem, data_size);
					MMLT_UNROLL
					for (int i = data_size; i < global_size; i++) {
						T val = gmem[i];
						if (condOp(gmem[0], val)) {
							algo.put(gmem, val, data_size);
						}
					}
				}*/
			} else {
				MMLT_UNROLL
				for (int i = 0; i < data_size; i++) {
					T val = data[i];
					if (condOp(gmem[0], val)) {
						algo.put(gmem, val, data_size);
					}
				}
			}
			mu.unlock();
		}
	}
};
}