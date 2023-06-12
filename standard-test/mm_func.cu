#include "mm_func.h"

cudaError_t cublas_mmul(
	cublasHandle_t &handle,
	const float *A,
	const float *B,
	float *C,
	const int m,
	const int n,
	const int k)
{
	int lda = m, ldb = k, ldc = m;
	const float alpha = 1.;
	const float beta = 0.;

	cublasStatus_t status = cublasSgemm(
		handle,
		CUBLAS_OP_N, CUBLAS_OP_N,
		m, n, k,
		&alpha,
		A, lda,
		B, ldb,
		&beta,
		C, ldc
	);
	check_othret(
		status, CUBLAS_STATUS_SUCCESS,
		cudaSuccess, cudaErrorUnknown
	);
}

cudaError_t cutlass_mmul(
	const float *A,
	const float *B,
	float *C,
	const int m,
	const int n,
	const int k
)
{
	int lda = m, ldb = n, ldc = n;
	const float alpha = 1.;
	const float beta = 0.;

	using ColumnMajor = cutlass::layout::ColumnMajor;
	using RowMajor = cutlass::layout::RowMajor;
	using CutlassGemm = cutlass::gemm::device::Gemm<
		float, // Data-type of A matrix
		ColumnMajor, // Layout of A matrix
		float, // Data-type of B matrix
		RowMajor, // Layout of B matrix
		float, // Data-type of C matrix
		RowMajor // Layout of C matrix
	>;

	CutlassGemm gemm_operator;

	CutlassGemm::Arguments args(
		{m, n, k}, // Gemm Problem dimensions
		{A, lda}, // Tensor-ref for source matrix A
		{B, ldb}, // Tensor-ref for source matrix B
		{C, ldc}, // Tensor-ref for source matrix C
		{C, ldc}, // Tensor-ref for destination matrix D (may be different memory than source C matrix)
		{alpha, beta}
	);

	cutlass::Status status = gemm_operator(args);
	check_othret(
		status, cutlass::Status::kSuccess,
		cudaSuccess, cudaErrorUnknown
	);
}

cudaError_t cutlass_mmul_split(
	uint8_t *workspace,
	const float *A,
	const float *B,
	float *C,
	const int m,
	const int n,
	const int k,
	const int split_size
)
{
	int lda = m, ldb = n, ldc = n;
	const float alpha = 1.;
	const float beta = 0.;
	int split_k_slices = (k + split_size - 1) / split_size;

	using ColumnMajor = cutlass::layout::ColumnMajor;
	using RowMajor = cutlass::layout::RowMajor;
	using CutlassGemm = cutlass::gemm::device::GemmSplitKParallel<
		float, // Data-type of A matrix
		ColumnMajor, // Layout of A matrix
		float, // Data-type of B matrix
		RowMajor, // Layout of B matrix
		float, // Data-type of C matrix
		RowMajor // Layout of C matrix
	>;

	CutlassGemm gemm_operator;

	CutlassGemm::Arguments args(
		{m, n, k}, // Gemm Problem dimensions
		{A, lda}, // Tensor-ref for source matrix A
		{B, ldb}, // Tensor-ref for source matrix B
		{C, ldc}, // Tensor-ref for source matrix C
		{C, ldc}, // Tensor-ref for destination matrix D (may be different memory than source C matrix)
		{alpha, beta},
		split_k_slices
	);

	cutlass::Status status = gemm_operator.initialize(args, workspace);
	check_return(
		(status == cutlass::Status::kSuccess ? cudaSuccess : cudaErrorUnknown),
		cudaSuccess, "Fail to initialize split-k\n"
	);
	status = gemm_operator();
	check_othret(
		status, cutlass::Status::kSuccess,
		cudaSuccess, cudaErrorUnknown
	);
}

cudaError_t split_workspace(
	const float *A,
	const float *B,
	float *C,
	const int m,
	const int n,
	const int k,
	const int split_size,
	uint8_t **ptr
)
{
	int lda = m, ldb = n, ldc = n;
	const float alpha = 1.;
	const float beta = 0.;
	int split_k_slices = (k + split_size - 1) / split_size;

	using ColumnMajor = cutlass::layout::ColumnMajor;
	using RowMajor = cutlass::layout::RowMajor;
	using CutlassGemm = cutlass::gemm::device::GemmSplitKParallel<
		float, // Data-type of A matrix
		ColumnMajor, // Layout of A matrix
		float, // Data-type of B matrix
		RowMajor, // Layout of B matrix
		float, // Data-type of C matrix
		RowMajor // Layout of C matrix
	>;

	CutlassGemm gemm_operator;

	CutlassGemm::Arguments args(
		{m, n, k}, // Gemm Problem dimensions
		{A, lda}, // Tensor-ref for source matrix A
		{B, ldb}, // Tensor-ref for source matrix B
		{C, ldc}, // Tensor-ref for source matrix C
		{C, ldc}, // Tensor-ref for destination matrix D (may be different memory than source C matrix)
		{alpha, beta},
		split_k_slices
	);
	size_t workspace_size = CutlassGemm::get_workspace_size(args);
	size_t bytes = workspace_size * sizeof(uint8_t);
	if ((bytes / 1024) > MAX_MEM)
		return cudaSuccess;
	cudaError_t result = cudaMalloc(ptr, bytes);

	check_return(
		result, cudaSuccess,
		"Failed to allocate workspace (size %ld): %s\n",
		bytes, cudaGetErrorString(result)
	);

	return result;
}