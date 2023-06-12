#pragma once

#include "utils.h"

namespace query2 {
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

	//bool within_x_a, within_y_a;
	//bool within_x_b, within_y_b;
	//int cur_a_y, cur_b_y;

public:
	MMLT_DEVICE
	InIterator(Array<Size_Thr_Row> *regs_row, Array<Size_Thr_Col> *regs_col, Array<Size_Thr_Tot> *regs_tot, InSharedMemory &smem, Params &param, idx &id)
	{
		size_M = param.size_p.kRow;
		size_N = param.size_p.kColumn;
		size_K = param.size_p.kDepth;

		coord<2> addr_a = {int(id.block_m) * Size_Block_Row + id.tot_thr % Size_Block_Row, param.size_K * int(id.block_k) + id.tot_thr / Size_Block_Row};
		gmem_a = param.inputs[0] + addr_a.x + addr_a.y * size_M;
		gsmem_a = smem.smem_a.storage + id.tot_thr;

		coord<2> addr_b = {int(id.block_n) * Size_Block_Col + id.tot_thr % Size_Block_Col, param.size_K * int(id.block_k) + id.tot_thr / Size_Block_Col};
		gmem_b = param.inputs[1] + addr_b.x + addr_b.y * size_N;
		gsmem_b = smem.smem_b.storage + id.tot_thr;

		//within_x_a = addr_a.x < size_M;
		//within_x_b = addr_b.x < size_N;
		//cur_a_y = addr_a.y;
		//cur_b_y = addr_b.y;

		rsmem_a = (Array<Size_Thr_Row> *)smem.smem_a.storage + (Size_Warp_Row / Size_Thr_Row) * id.warp_m + (Size_Thr_Row / Size_Thr_Row) * id.thread_m;
		reg_a = regs_row;

		rsmem_b = (Array<Size_Thr_Col> *)smem.smem_b.storage + (Size_Warp_Col / Size_Thr_Col) * id.warp_n + (Size_Thr_Col / Size_Thr_Col) * id.thread_n;
		reg_b = regs_col;

		float *gmem_r = param.inputs[2] + int(id.block_n) * Size_Block_Col;
		Array<Size_Thr_Col> *rsmem_r = (Array<Size_Thr_Col> *)gmem_r + (Size_Warp_Col / Size_Thr_Col) * id.warp_n + (Size_Thr_Col / Size_Thr_Col) * id.thread_n;
		regs_col[1] = rsmem_r[0];
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
			//if (within_x_a && cur_a_y < size_K) {
				gsmem_a[i * Load_Interval_A * Size_Block_Row] = gmem_a[i * Load_Interval_A * size_M];
			//} else {
			//	gsmem_a[i * Load_Interval_A * Size_Block_Row] = 0.f;
			//}
			//cur_a_y += Load_Interval_A;
		}

		MMLT_UNROLL
		for (int i = 0; i < Load_Num_Iter_B; i++) {
			//if (within_x_b && cur_b_y < size_K) {
				gsmem_b[i * Load_Interval_B * Size_Block_Col] = gmem_b[i * Load_Interval_B * size_N];
			//} else {
			//	gsmem_b[i * Load_Interval_B * Size_Block_Col] = 0.f;
			//}
			//cur_b_y += Load_Interval_B;
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
{
	AlignedBuffer<Count_Thr_Block_Row * Size_Block_Col> smem_c;
};

class OutIterator
{
private:
	float *gmem_c;
	float *gsmem_c;

	Array<Size_Thr_Col> *rsmem_c;
	Array<Size_Thr_Col> *reg_c;

	int size_M, size_N;

	//bool exceed_y;
	//int cur_c_x;

public:
	MMLT_DEVICE
	OutIterator(Array<Size_Thr_Tot> &out_reg, OutSharedMemory &smem, Params &param, idx &id)
	{
		size_M = param.size_p.kRow;
		size_N = param.size_p.kColumn;

		coord<2> addr_c = {Size_Block_Row * (int)id.block_m + (id.tot_thr / Size_Block_Col) * Size_Thr_Row, Size_Block_Col * (int)id.block_n + id.tot_thr % Size_Block_Col};
		gmem_c = param.work + id.block_k * (size_M * size_N) + addr_c.x * size_N + addr_c.y;
		//exceed_y = addr_c.y >= size_N;
		//cur_c_x = addr_c.x;

		gsmem_c = smem.smem_c.storage + id.tot_thr;
		rsmem_c = (Array<Size_Thr_Col> *)smem.smem_c.storage + (id.warp_m * Count_Thr_Warp_Row + id.thread_m) * Count_Thr_Block_Col + (id.warp_n * Count_Thr_Warp_Col + id.thread_n);
		reg_c = (Array<Size_Thr_Col> *)&out_reg;
	}

	MMLT_DEVICE
	void store_next_smem()
	{
		//int cur_x = cur_c_x;
		MMLT_UNROLL
		for (int i = 0; i < Store_Num_Iter; i++) {
			//if (exceed_y || cur_x >= size_M) {
			//	break;
			//}
			gmem_c[i * Store_Interval * Size_Thr_Row * size_N] = gsmem_c[i * Store_Interval * Size_Block_Col];
			//cur_x += Store_Interval * Size_Thr_Row;
		}
		gmem_c += size_N;
		//cur_c_x += 1;
	}

	MMLT_DEVICE
	void store_next_reg()
	{
		rsmem_c[0] = reg_c[0];
		++reg_c;
	}
};
}