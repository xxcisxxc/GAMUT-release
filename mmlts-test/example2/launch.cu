#include "launch.h"
#include <iostream>

cudaError_t example2::launch
(
	Params param,
	cudaStream_t stream
)
{
	cudaError_t result;

	int grid_m = (param.size_p.kRow + Size_Block_Row - 1) / Size_Block_Row;
	int grid_n = (param.size_p.kColumn + Size_Block_Col - 1) / Size_Block_Col;
	int grid_k = (param.size_p.kDepth + param.size_K - 1) / param.size_K;

	param.n_partition = grid_k;

	if (grid_k == 1) {
		param.work = param.output;
	} else {
		result = cudaMalloc(&param.work, grid_k * param.size_p.kRow * param.size_p.kColumn * sizeof(float));
		if (result != cudaSuccess)
			return result;
	}

	dim3 grid(grid_m, grid_n, grid_k);
	dim3 block(Count_Thr_Block_Tot, 1, 1);

	compute
	<<<grid, block, 0, stream>>> (
		param
	);

	result = cudaGetLastError();
	if (result != cudaSuccess)
		return result;

	if (grid_k == 1) {
		return cudaSuccess;
	}

	grid.x = (param.size_p.kRow + Reduce_Row - 1) / Reduce_Row;
	grid.y = (param.size_p.kRow + Reduce_Col - 1) / Reduce_Col;
	grid.z = 1;

	block.x = Reduce_Col;
	block.y = Reduce_Row;

	reduce
	<<<grid, block, 0, stream>>> (
		param
	);

	result = cudaGetLastError();
	if (result != cudaSuccess)
		return result;

	return cudaSuccess;
}