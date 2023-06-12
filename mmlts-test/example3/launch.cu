#include "launch.h"
#include <iostream>

namespace example3 {
cudaError_t launch_internal
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
}

cudaError_t example3::launch
(
	Params param,
	cudaStream_t stream
)
{
	int P0 = param.dim2[0].x, P1 = param.dim2[0].y;
	int P = P0 < P1 ? P0 : P1;
	float *A = param.inputs[0], *B = param.inputs[1], *C = param.output;
	float *E = param.inputs[2], *F = param.inputs[3];
	int D = param.size_p.kDepth;

	Params p_arr[P];
	cudaStream_t streams[P];
	int cur_E = 0, cur_F = 0;
	for (int i = 0; i < P; i++) {
		int last_E = cur_E, last_F = cur_F;
		while (cur_E < param.size_p.kRow && int(E[cur_E]) == i) {
			cur_E++;
		}
		while (cur_F < param.size_p.kColumn && int(F[cur_F]) == i) {
			cur_F++;
		}
		int M = cur_E-last_E, N = cur_F-last_F;
		Params p = {C + i, nullptr, {A+last_E, B+last_F}, {M, N, D}, D, {{1, 1}, {param.size_p.kRow, D}, {D, param.size_p.kColumn}}, 1};
		p_arr[i] = p;
		cudaStreamCreate(streams + i);
	}

	for (int i = 0; i < P; i++)
		launch_internal(p_arr[i], streams[i]);

	return cudaGetLastError();
}