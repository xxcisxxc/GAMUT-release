#pragma once

#include "utils.h"

namespace example3 {
struct InSharedMemory
{
	AlignedBuffer<Size_Block_Row * Block_Depth> smem_a;
	AlignedBuffer<Size_Block_Col * Block_Depth> smem_b;
};

class InIterator
{
private:
	float *gmem_a, *gsmem_a;

	float *gmem_b, *gsmem_b;

	Array<Size_Thr_Row> *rsmem_a;
	Array<Size_Thr_Col> *rsmem_b;

	Array<Size_Thr_Row> *reg_a;
	Array<Size_Thr_Col> *reg_b;

	int size_M, size_N, size_K;
	int size_pM, size_pN;

	bool within_x_a, within_y_a;
	bool within_x_b, within_y_b;
	int cur_a_y, cur_b_y;

public:
	MMLT_DEVICE
	InIterator(Array<Size_Thr_Row> *regs_row, Array<Size_Thr_Col> *regs_col, Array<Size_Thr_Tot> *regs_tot, InSharedMemory &smem, Params &param, idx &id)
	{
		size_M = param.dim2[1].x;
		size_N = param.dim2[2].y;
		size_K = param.size_p.kDepth;
		size_pM = param.size_p.kRow;
		size_pN = param.size_p.kColumn;

		coord<2> addr_a = {int(id.block_m) * Size_Block_Row + id.tot_thr % Size_Block_Row, param.size_K * int(id.block_k) + id.tot_thr / Size_Block_Row};
		gmem_a = param.inputs[0] + addr_a.x + addr_a.y * size_M;
		gsmem_a = smem.smem_a.storage + id.tot_thr;

		coord<2> addr_b = {int(id.block_n) * Size_Block_Col + id.tot_thr % Size_Block_Col, param.size_K * int(id.block_k) + id.tot_thr / Size_Block_Col};
		gmem_b = param.inputs[1] + addr_b.x + addr_b.y * size_N;
		gsmem_b = smem.smem_b.storage + id.tot_thr;

		within_x_a = addr_a.x < size_pM;
		within_x_b = addr_b.x < size_pN;
		cur_a_y = addr_a.y;
		cur_b_y = addr_b.y;

		rsmem_a = (Array<Size_Thr_Row> *)smem.smem_a.storage + (Size_Warp_Row / Size_Thr_Row) * id.warp_m + (Size_Thr_Row / Size_Thr_Row) * id.thread_m;
		reg_a = regs_row;

		rsmem_b = (Array<Size_Thr_Col> *)smem.smem_b.storage + (Size_Warp_Col / Size_Thr_Col) * id.warp_n + (Size_Thr_Col / Size_Thr_Col) * id.thread_n;
		reg_b = regs_col;
	}

	MMLT_DEVICE
	void reset_smem_offset()
	{
		rsmem_a -= (Size_Block_Row / Size_Thr_Row) * Block_Depth;
		rsmem_b -= (Size_Block_Col / Size_Thr_Col) * Block_Depth;
	}

	MMLT_DEVICE
	void load_next_smem()
	{
		MMLT_UNROLL
		for (int i = 0; i < Load_Num_Iter_A; i++) {
			if (within_x_a && cur_a_y < size_K) {
				gsmem_a[i * Load_Interval_A * Size_Block_Row] = gmem_a[i * Load_Interval_A * size_M];
			} else {
				gsmem_a[i * Load_Interval_A * Size_Block_Row] = 0.f;
			}
			cur_a_y += Load_Interval_A;
		}

		MMLT_UNROLL
		for (int i = 0; i < Load_Num_Iter_B; i++) {
			if (within_x_b && cur_b_y < size_K) {
				gsmem_b[i * Load_Interval_B * Size_Block_Col] = gmem_b[i * Load_Interval_B * size_N];
			} else {
				gsmem_b[i * Load_Interval_B * Size_Block_Col] = 0.f;
			}
			cur_b_y += Load_Interval_B;
		}

		gmem_a += Block_Depth * size_M;
		gmem_b += Block_Depth * size_N;
	}

	MMLT_DEVICE
	void load_next_reg()
	{
		reg_a[0] = rsmem_a[0];
		reg_b[0] = rsmem_b[0];

		rsmem_a += Size_Block_Row / Size_Thr_Row;
		rsmem_b += Size_Block_Col / Size_Thr_Col;
	}
};

struct OutSharedMemory
{ };

class OutIterator
{
private:

public:
	MMLT_DEVICE
	OutIterator(Array<Size_Thr_Tot> &out_reg, Array<Size_Thr_Row> *regs_row, Array<Size_Thr_Col> *regs_col, Array<Size_Thr_Tot> *regs_tot, OutSharedMemory &smem, Params &param, idx &id)
	{
		// Add together
		float accum = 0;
		for (int i = 0; i < Size_Thr_Row; i++) {
			for (int j = 0; j < Size_Thr_Col; j++) {
				accum += out_reg.storage[i * Size_Thr_Col + j];
			}
		}
		atomicAdd(param.work, accum);
	}

	MMLT_DEVICE
	void store_next_smem()
	{ }

	MMLT_DEVICE
	void store_next_reg()
	{ }
};
}