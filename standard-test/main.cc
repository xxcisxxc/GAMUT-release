#include "common.h"
#include "m_alloc.h"
#include "mm_func.h"
#include "workkernel/launch.h"
#include "gpu_timer.h"
#include "naive.h"
#include <iostream>

#define N 10

int test();

int main(int argc, char *argv[])
{
	/*if (argc == 1) {
		return test();
	}*/

	check_return(argc, 4, "Usage: %s m n k deprecated[size]\n", argv[0]);
	const int m = std::atoi(argv[1]),
		n = std::atoi(argv[2]),
		k = std::atoi(argv[3])/*,
		split_size = std::atoi(argv[4])*/;
	float *A, *B, *C;
	Progress prog;
	GpuTimer timer;
	double avg_time = 0.;

	std::cout << "Allocate memory\n";
	prog.start(4, "Allocating matrices");

	cudaError_t result = AllocateMatrix(&A, m, k);
	cudaDeviceSynchronize();
	prog.inc();
	check_return(result, cudaSuccess, "Allocate A fails\n");

	result = AllocateMatrix(&B, k, n);
	cudaDeviceSynchronize();
	prog.inc();
	check_return(result, cudaSuccess, "Allocate B fails\n");

	result = AllocateMatrix(&C, m, n);
	cudaDeviceSynchronize();
	prog.inc();
	check_return(result, cudaSuccess, "Allocate C fails\n");

	cublasHandle_t handle;
	cublasCreate(&handle);

	/*uint8_t *work = nullptr;
	result = split_workspace(A, B, C, m, n, k, split_size, &work);
	cudaDeviceSynchronize();*/
	prog.end();


	std::cout << "Estimate time & Guarantee runable\n";
	prog.start(4, "Estimating time");

	timer.start();
	result = cublas_mmul(handle, A, B, C, m, n, k);
	timer.stop_and_wait();
	prog.inc();
	avg_time += timer.duration();
	check_return(result, cudaSuccess, "CUBLAS fails\n");

	timer.start();
	result = cutlass_mmul(A, B, C, m, n, k);
	timer.stop_and_wait();
	prog.inc();
	avg_time += timer.duration();
	check_return(result, cudaSuccess, "CUTLASS fails\n");

	timer.start();
	result = standard::launch({C, nullptr, {A, B}, {m, n, k}, k, {{m, k}, {k, n}}, 1});
	timer.stop_and_wait();
	prog.inc();
	avg_time += timer.duration();
	check_return(result, cudaSuccess, "MMLT fails\n");

	timer.start();
	result = naive(A, B, C, m, n, k);
	timer.stop_and_wait();
	prog.inc();
	check_return(result, cudaSuccess, "Naive fails\n");

	/*if (work) {
		timer.start();
		result = cutlass_mmul_split(work, A, B, C, m, n, k, split_size);
		timer.stop_and_wait();
		prog.inc();
		avg_time += timer.duration();
		check_return(result, cudaSuccess, "CUTLASS split k fails\n");
	}*/
	prog.end();

	//avg_time /= (work ? 3 : 2);
	avg_time /= 3;
	if (avg_time > 50.f)
		goto progbar;


	std::cout << "Test CUBLAS\n";

	timer.start();
	for (int i = 0; i < N; i++) {
		cublas_mmul(handle, A, B, C, m, n, k);
	}
	timer.stop_and_wait();

	cublasDestroy(handle);
	std::cout << "CUBLAS result is " << timer.duration(N) << " ms\n";


	std::cout << "Test CUTLASS\n";

	timer.start();
	for (int i = 0; i < N; i++) {
		cutlass_mmul(A, B, C, m, n, k);
	}
	timer.stop_and_wait();

	std::cout << "CUTLASS result is " << timer.duration(N) << " ms\n";


	std::cout << "Test MMLT\n";

	timer.start();
	for (int i = 0; i < N; i++) {
		standard::launch({C, nullptr, {A, B}, {m, n, k}, k, {{m, k}, {k, n}}, 1});
	}
	timer.stop_and_wait();

	std::cout << "MMLT result is " << timer.duration(N) << " ms\n";


	std::cout << "Test Naive\n";

	timer.start();
	for (int i = 0; i < N; i++) {
		naive(A, B, C, m, n, k);
	}
	timer.stop_and_wait();

	std::cout << "Naive result is " << timer.duration(N) << " ms\n";
	/*if (!work)
		return cudaSuccess;
	
	std::cout << "Test CUTLASS split k\n";

	timer.start();
	for (int i = 0; i < N; i++) {
		cutlass_mmul_split(work, A, B, C, m, n, k, split_size);
	}
	timer.stop_and_wait();

	cudaFree(work);
	std::cout << "CUTLASS split k result is " << timer.duration(N) << " ms\n";*/


	return cudaSuccess;


progbar:
	std::cout << "Test CUBLAS\n";
	prog.start(N, "Testing CUBLAS");

	avg_time = 0.;
	for (int i = 0; i < N; i++) {
		timer.start();
		cublas_mmul(handle, A, B, C, m, n, k);
		timer.stop_and_wait();
		prog.inc();
		avg_time += timer.duration();
	}
	prog.end();

	cublasDestroy(handle);
	std::cout << "CUBLAS result is " << avg_time / N << " ms\n";


	std::cout << "Test CUTLASS\n";
	prog.start(N, "Testing CUTLASS");

	avg_time = 0.;
	for (int i = 0; i < N; i++) {
		timer.start();
		cutlass_mmul(A, B, C, m, n, k);
		timer.stop_and_wait();
		prog.inc();
		avg_time += timer.duration();
	}
	prog.end();

	std::cout << "CUTLASS result is " << avg_time / N << " ms\n";


	std::cout << "Test MMLT\n";
	prog.start(N, "Testing MMLT");

	avg_time = 0.;
	for (int i = 0; i < N; i++) {
		timer.start();
		standard::launch({C, nullptr, {A, B}, {m, n, k}, k, {{m, k}, {k, n}}, 1});
		timer.stop_and_wait();
		prog.inc();
		avg_time += timer.duration();
	}
	prog.end();

	std::cout << "MMLT result is " << avg_time / N << " ms\n";


	std::cout << "Test Naive\n";
	prog.start(N, "Testing Naive");

	avg_time = 0.;
	for (int i = 0; i < N; i++) {
		timer.start();
		naive(A, B, C, m, n, k);
		timer.stop_and_wait();
		prog.inc();
		avg_time += timer.duration();
	}
	prog.end();

	std::cout << "Naive result is " << avg_time / N << " ms\n";
	/*if (!work)
		return cudaSuccess;

	std::cout << "Test CUTLASS split k\n";
	prog.start(N, "Testing CUTLASS split k");

	avg_time = 0.;
	for (int i = 0; i < N; i++) {
		timer.start();
		cutlass_mmul_split(work, A, B, C, m, n, k, split_size);
		timer.stop_and_wait();
		prog.inc();
		avg_time += timer.duration();
	}
	prog.end();

	cudaFree(work);
	std::cout << "CUTLASS split k result is " << avg_time / N << " ms\n";*/


	return cudaSuccess;
}

int test()
{
	// Run this test under unified memory access or it will be segmentfault
	float *A, *B, *C;
	const int m = 512, k = 512, n = 512, split_size = 128;
	cudaError_t result = AllocateMatrix(&A, m, k);
	check_return(result, cudaSuccess, "Allocate A fails\n");

	result = AllocateMatrix(&B, k, n);
	check_return(result, cudaSuccess, "Allocate B fails\n");

	result = AllocateMatrix(&C, m, n);
	check_return(result, cudaSuccess, "Allocate C fails\n");

	cublasHandle_t handle;
	cublasCreate(&handle);
	uint8_t *work;
	result = split_workspace(A, B, C, m, n, k, split_size, &work);
	//result = cutlass_mmul(A, B, C, m, n, k);
	result = cutlass_mmul_split(work, A, B, C, m, n, k, split_size);
	//result = cublas_mmul(handle, A, B, C, m, n, k);
	check_return(result, cudaSuccess, "Matrix multiplication fails\n");
	cudaDeviceSynchronize();

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			float c = 0.f;
			for (int l = 0; l < k; l++) {
				c += ((A[i+l*m] * B[l*n+j]) > 0.5);
			}
			if (std::abs(c - C[i*n+j]) > 1e-3)
				std::printf("(%d, %d)\ttarget: %f\tcompute: %f\n", i, j, c, C[i*n+j]);
		}
	}

	cublasDestroy(handle);
	cudaFree(work);

	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
	return cudaSuccess;
}