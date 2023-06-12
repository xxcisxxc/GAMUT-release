#include "kernel.h"

__global__
__launch_bounds__(standard::Count_Thr_Block_Tot)
void standard::compute
(
	Params param
)
{
	__shared__ uint8_t SharedMemoryBase[sizeof(SharedMemory)];

	SharedMemory *smem = (SharedMemory *)SharedMemoryBase;

	int thread_idx = threadIdx.x;
	int warp_idx = thread_idx / Count_Thr_Warp_Tot;
	int lane_idx = thread_idx % Count_Thr_Warp_Tot;

	idx id = {
		blockIdx.x, blockIdx.y, blockIdx.z,
		warp_idx % Count_Warp_Block_Row, warp_idx / Count_Warp_Block_Row,
		lane_idx % Count_Thr_Warp_Row, lane_idx / Count_Thr_Warp_Row,
		thread_idx
	};

#if Num_Inputs_Row == 0
	Array<Size_Thr_Row> *in_regs_row = nullptr;
#else
	Array<Size_Thr_Row> in_regs_row[Num_Inputs_Row];
#endif /* Num_Inputs_Row == 0 */

#if Num_Inputs_Col == 0
	Array<Size_Thr_Col> *in_regs_col = nullptr;
#else
	Array<Size_Thr_Col> in_regs_col[Num_Inputs_Col];
#endif /* Num_Inputs_Col == 0 */

#if Num_Inputs_Tot == 0
	Array<Size_Thr_Tot> *in_regs_tot = nullptr;
#else
	Array<Size_Thr_Tot> in_regs_tot[Num_Inputs_Tot];
#endif /* Num_Inputs_Tot == 0 */

	InIterator in_iter(in_regs_row, in_regs_col, in_regs_tot, smem->in_smem, param, id);

	Array<Size_Thr_Tot> out_reg;
	out_reg.fill(0);

	int size_K_ = min(param.size_K, param.size_p.kDepth - id.block_k * param.size_K);
	int k_iterations = (size_K_ + Block_Depth - 1) / Block_Depth;

	MMLT_LOOP
	for (; k_iterations > 0; k_iterations--) {
		in_iter.load_next_smem();

		__syncthreads();

		MMLT_UNROLL
		for (int k = 0; k < Block_Depth; k++) {
			in_iter.load_next_reg();
			mmlt_op(out_reg, in_regs_row, in_regs_col, in_regs_tot);
		}
		in_iter.reset_smem_offset();

		__syncthreads();
	}

	/*if (param.n_partition == 1) {
		MMLT_UNROLL
		for (int i = 0; i < Size_Thr_Row; i++) {
			for (int j = 0; j < Size_Thr_Col; j++) {
				coda_op(out_reg.storage[i * Size_Thr_Col + j]);
			}
		}
	}*/

	OutIterator out_iter(out_reg, smem->out_smem, param, id);
	MMLT_UNROLL
	for (int i = 0; i < Size_Thr_Row; i++) {
		out_iter.store_next_reg();
		__syncthreads();

		out_iter.store_next_smem();
		__syncthreads();
	}
}

__global__
__launch_bounds__(standard::Reduce_Row * standard::Reduce_Col)
void standard::reduce
(
	Params param
)
{
	coord<2> offset = {int(blockIdx.x) * Reduce_Row + int(threadIdx.y), int(blockIdx.y) * Reduce_Col + int(threadIdx.x)};

	if (offset.x >= param.size_p.kRow || offset.y >= param.size_p.kColumn) {
		return;
	}

	float accum = 0;
	float thread_work[Reduce_Inner_Iter];

	int off = offset.x * param.size_p.kColumn + offset.y;
	float *work_base = param.work + off;
	float *output_base = param.output + off;

	off = param.size_p.kRow * param.size_p.kColumn;

	MMLT_LOOP
	for (int k = 0; k < param.n_partition; k += Reduce_Inner_Iter) {
		MMLT_UNROLL
		for (int i = 0; i < Reduce_Inner_Iter; i++) {
			if (k + i < param.n_partition) {
				thread_work[i] = *work_base;
				work_base += off;
			}
		}

		MMLT_UNROLL
		for (int i = 0; i < Reduce_Inner_Iter; i++) {
			if (k + i < param.n_partition) {
				reduce_op(accum, thread_work[i]);
			}
		}
	}

	coda_op(accum);

	*output_base = accum;
}