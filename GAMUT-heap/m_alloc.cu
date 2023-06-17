#include "m_alloc.h"

__global__ void InitializeMatrix_kernel(float *matrix, int rows, int columns, int seed)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (i < rows && j < columns) {
		int offset = i + j * rows;

		//curandState_t state;
		//curand_init(seed, offset, 0, &state);
		//matrix[offset] = curand_uniform(&state);
		matrix[offset] = i;
	}
}

cudaError_t InitializeMatrix(float *matrix, int rows, int columns)
{
	dim3 block(16, 16);
	dim3 grid(
		(rows + block.x - 1) / block.x,
		(columns + block.y - 1) / block.y
	);

	InitializeMatrix_kernel<<< grid, block >>>(matrix, rows, columns, std::rand());

	return cudaGetLastError();
}

cudaError_t AllocateMatrix(float **matrix, int rows, int columns, bool initialized)
{
	cudaError_t result;

	size_t sizeof_matrix = sizeof(float) * rows * columns;

	// Allocate device memory.
	result = cudaMallocManaged((void **)(matrix), sizeof_matrix);
	//result = cudaMalloc((void **)(matrix), sizeof_matrix);

	// Clear the allocation.
	result = cudaMemset(*matrix, 0, sizeof_matrix);

	if (!initialized)
		return result;

	// Initialize matrix elements to arbitrary small integers.
	result = InitializeMatrix(*matrix, rows, columns);

	return result;
}
