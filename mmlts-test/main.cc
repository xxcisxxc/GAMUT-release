#include "common.h"
#include "m_alloc.h"
#include "gpu_timer.h"
#include "naive.h"
#include "helper.h"
#include <iostream>

#include "workkernel/launch.h"
#include "query2/launch.h"
#include "example1/launch.h"
#include "example2/launch.h"
#include "example3/launch.h"
#include "example4/launch.h"
#include "example5/launch.h"
#include "example6/launch.h"

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
	float *A, *B, *C, *R, *E, *F, *G, *mu;
	Progress prog;
	GpuTimer timer;
	double avg_time = 0.;

	std::cout << "Allocate memory\n";
	prog.start(7, "Allocating matrices");

	cudaError_t result = AllocateMatrix(&A, m, k);
	result = NormalizeMatrix(A, m, k);
	cudaDeviceSynchronize();
	prog.inc();
	check_return(result, cudaSuccess, "Allocate A fails\n");

	result = AllocateMatrix(&B, k, n);
	cudaDeviceSynchronize();
	prog.inc();
	check_return(result, cudaSuccess, "Allocate B fails\n");

	result = AllocateMatrix(&C, m, n, false);
	cudaDeviceSynchronize();
	prog.inc();
	check_return(result, cudaSuccess, "Allocate C fails\n");

	result = AllocateMatrix(&R, 1, n);
	cudaDeviceSynchronize();
	prog.inc();
	check_return(result, cudaSuccess, "Allocate R fails\n");

	result = AllocateMatrix(&E, 1, m, false);
	cudaDeviceSynchronize();
	check_return(result, cudaSuccess, "Allocate E fails\n");

	result = AllocateMatrix(&F, 1, n, false);
	cudaDeviceSynchronize();
	check_return(result, cudaSuccess, "Allocate F fails\n");

	result = setList(E, F, m, n);
	cudaDeviceSynchronize();
	prog.inc();
	check_return(result, cudaSuccess, "Initialize E & F fails\n");

	result = AllocateMatrix(&G, 1, 1, false);
	cudaDeviceSynchronize();
	cudaMemset(G, 0, sizeof(float));
	prog.inc();
	check_return(result, cudaSuccess, "Allocate G fails\n");

	result = cudaMalloc(&mu, sizeof(int));
	cudaMemset(mu, 0, sizeof(int));
	prog.inc();
	check_return(result, cudaSuccess, "Allocate mu fails\n");

	prog.end();


	std::cout << "Estimate time & Guarantee runable\n";
	prog.start(16, "Estimating time");

	timer.start();
	result = standard::launch({C, nullptr, {A, B}, {m, n, k}, k, {{m, k}, {k, n}}, 1});
	timer.stop_and_wait();
	prog.inc();
	avg_time += timer.duration();
	check_return(result, cudaSuccess, "MMLT fails\n");

	timer.start();
	result = query2::launch({C, nullptr, {A, B, R}, {m, n, k}, k, {{m, k}, {k, n}, {1, n}}, 1});
	timer.stop_and_wait();
	prog.inc();
	avg_time += timer.duration();
	check_return(result, cudaSuccess, "Query2 fails\n");

	timer.start();
	result = example1::launch({C, nullptr, {A, B, E, F}, {m, n, k}, k, {{P0, P1}, {m, k}, {k, n}, {m, 1}, {1, n}}, 1});
	timer.stop_and_wait();
	prog.inc();
	avg_time += timer.duration();
	check_return(result, cudaSuccess, "Example1 fails\n");

	timer.start();
	result = example2::launch({C, nullptr, {A, B, E, F}, {m, n, k}, k, {{P0, P1}, {m, k}, {k, n}, {m, 1}, {1, n}}, 1});
	timer.stop_and_wait();
	prog.inc();
	avg_time += timer.duration();
	check_return(result, cudaSuccess, "Example2 fails\n");

	timer.start();
	result = example3::launch({C, nullptr, {A, B, E, F}, {m, n, k}, k, {{P0, P1}, {m, k}, {k, n}, {m, 1}, {1, n}}, 1});
	timer.stop_and_wait();
	prog.inc();
	avg_time += timer.duration();
	check_return(result, cudaSuccess, "Example3 fails\n");

	cudaMemset(G, 0, sizeof(float));
	timer.start();
	result = example4::launch({C, nullptr, {A, B, G}, {m, n, k}, k, {{m, n}, {m, k}, {k, n}, {1, 1}}, 1});
	timer.stop_and_wait();
	prog.inc();
	avg_time += timer.duration();
	check_return(result, cudaSuccess, "Example4 fails\n");

	cudaMemset(G, 0, sizeof(float));
	timer.start();
	result = example5::launch({C, nullptr, {A, A, G}, {m, m, k}, k, {{m, m}, {m, k}, {k, m}, {1, 1}}, 1});
	timer.stop_and_wait();
	prog.inc();
	avg_time += timer.duration();
	check_return(result, cudaSuccess, "Example5 fails\n");

	cudaMemset(G, 0, sizeof(float));
	cudaMemset(mu, 0, sizeof(int));
	timer.start();
	result = example6::launch({C, nullptr, {A, B, G, mu}, {m, n, k}, k, {{m, n}, {m, k}, {k, n}, {1, 1}, {1, 1}}, 1});
	timer.stop_and_wait();
	prog.inc();
	avg_time += timer.duration();
	check_return(result, cudaSuccess, "Example6 fails\n");

	timer.start();
	result = standard::naive(A, B, C, m, n, k);
	timer.stop_and_wait();
	prog.inc();
	check_return(result, cudaSuccess, "Naive standard fails\n");

	timer.start();
	result = query2::naive(A, B, C, R, m, n, k);
	timer.stop_and_wait();
	prog.inc();
	check_return(result, cudaSuccess, "Naive query 2 fails\n");

	timer.start();
	result = example1::naive(A, B, C, E, F, m, n, k, P0, P1);
	timer.stop_and_wait();
	prog.inc();
	check_return(result, cudaSuccess, "Naive example 1 fails\n");

	timer.start();
	result = example2::naive(A, B, C, E, F, m, n, k, P0, P1);
	timer.stop_and_wait();
	prog.inc();
	check_return(result, cudaSuccess, "Naive example 2 fails\n");

	timer.start();
	result = example3::naive(A, B, C, E, F, m, n, k, P0, P1);
	timer.stop_and_wait();
	prog.inc();
	check_return(result, cudaSuccess, "Naive example 3 fails\n");

	cudaMemset(G, 0, sizeof(float));
	timer.start();
	result = example4::naive(A, B, C, G, m, n, k);
	timer.stop_and_wait();
	prog.inc();
	check_return(result, cudaSuccess, "Naive example 4 fails\n");

	cudaMemset(G, 0, sizeof(float));
	timer.start();
	result = example5::naive(A, A, C, G, m, n, k);
	timer.stop_and_wait();
	prog.inc();
	check_return(result, cudaSuccess, "Naive example 5 fails\n");

	cudaMemset(G, 0, sizeof(float));
	cudaMemset(mu, 0, sizeof(int));
	timer.start();
	result = example6::naive(A, B, C, G, (int *)mu, m, n, k);
	timer.stop_and_wait();
	prog.inc();
	check_return(result, cudaSuccess, "Naive example 6 fails\n");

	prog.end();

	avg_time /= 8;
	int N;
	if (avg_time > 2000.f)
		N = 1;
	else
		N = 10;

	std::cout << "Test MMLT-standard\n";

	timer.start();
	for (int i = 0; i < N; i++) {
		standard::launch({C, nullptr, {A, B}, {m, n, k}, k, {{m, k}, {k, n}}, 1});
	}
	timer.stop_and_wait();

	std::cout << "MMLT-standard result is " << timer.duration(N) << " ms\n";


	std::cout << "Test MMLT-Query2\n";

	timer.start();
	for (int i = 0; i < N; i++) {
		query2::launch({C, nullptr, {A, B, R}, {m, n, k}, k, {{m, k}, {k, n}, {1, n}}, 1});
	}
	timer.stop_and_wait();

	std::cout << "MMLT-Query2 result is " << timer.duration(N) << " ms\n";


	std::cout << "Test MMLT-Example1\n";

	timer.start();
	for (int i = 0; i < N; i++) {
		example1::launch({C, nullptr, {A, B, E, F}, {m, n, k}, k, {{P0, P1}, {m, k}, {k, n}, {m, 1}, {1, n}}, 1});
	}
	timer.stop_and_wait();

	std::cout << "MMLT-Example1 result is " << timer.duration(N) << " ms\n";


	std::cout << "Test MMLT-Example2\n";

	timer.start();
	for (int i = 0; i < N; i++) {
		example2::launch({C, nullptr, {A, B, E, F}, {m, n, k}, k, {{P0, P1}, {m, k}, {k, n}, {m, 1}, {1, n}}, 1});
	}
	timer.stop_and_wait();

	std::cout << "MMLT-Example2 result is " << timer.duration(N) << " ms\n";


	std::cout << "Test MMLT-Example3\n";

	timer.start();
	for (int i = 0; i < N; i++) {
		example3::launch({C, nullptr, {A, B, E, F}, {m, n, k}, k, {{P0, P1}, {m, k}, {k, n}, {m, 1}, {1, n}}, 1});
	}
	timer.stop_and_wait();

	std::cout << "MMLT-Example3 result is " << timer.duration(N) << " ms\n";


	std::cout << "Test MMLT-Example4\n";

	timer.start();
	for (int i = 0; i < N; i++) {
		cudaMemset(G, 0, sizeof(float));
		example4::launch({C, nullptr, {A, B, G}, {m, n, k}, k, {{m, n}, {m, k}, {k, n}, {1, 1}}, 1});
	}
	timer.stop_and_wait();

	std::cout << "MMLT-Example4 result is " << timer.duration(N) << " ms\n";


	std::cout << "Test MMLT-Example5\n";

	timer.start();
	for (int i = 0; i < N; i++) {
		cudaMemset(G, 0, sizeof(float));
		example5::launch({C, nullptr, {A, A, G}, {m, m, k}, k, {{m, m}, {m, k}, {k, m}, {1, 1}}, 1});
	}
	timer.stop_and_wait();

	std::cout << "MMLT-Example5 result is " << timer.duration(N) << " ms\n";


	std::cout << "Test MMLT-Example6\n";

	timer.start();
	for (int i = 0; i < N; i++) {
		cudaMemset(G, 0, sizeof(float));
		cudaMemset(mu, 0, sizeof(int));
		example6::launch({C, nullptr, {A, B, G, mu}, {m, n, k}, k, {{m, n}, {m, k}, {k, n}, {1, 1}, {1, 1}}, 1});
	}
	timer.stop_and_wait();

	std::cout << "MMLT-Example6 result is " << timer.duration(N) << " ms\n";

	N = 1;
	std::cout << "Test Naive-standard\n";

	timer.start();
	for (int i = 0; i < N; i++) {
		standard::naive(A, B, C, m, n, k);
	}
	timer.stop_and_wait();

	std::cout << "Naive-standard result is " << timer.duration(N) << " ms\n";


	std::cout << "Test Naive-Query2\n";

	timer.start();
	for (int i = 0; i < N; i++) {
		query2::naive(A, B, C, R, m, n, k);
	}
	timer.stop_and_wait();

	std::cout << "Naive-Query2 result is " << timer.duration(N) << " ms\n";


	std::cout << "Test Naive-Example1\n";

	timer.start();
	for (int i = 0; i < N; i++) {
		example1::naive(A, B, C, E, F, m, n, k, P0, P1);
	}
	timer.stop_and_wait();

	std::cout << "Naive-Example1 result is " << timer.duration(N) << " ms\n";


	std::cout << "Test Naive-Example2\n";

	timer.start();
	for (int i = 0; i < N; i++) {
		example2::naive(A, B, C, E, F, m, n, k, P0, P1);
	}
	timer.stop_and_wait();

	std::cout << "Naive-Example2 result is " << timer.duration(N) << " ms\n";


	std::cout << "Test Naive-Example3\n";

	timer.start();
	for (int i = 0; i < N; i++) {
		example3::naive(A, B, C, E, F, m, n, k, P0, P1);
	}
	timer.stop_and_wait();

	std::cout << "Naive-Example3 result is " << timer.duration(N) << " ms\n";


	std::cout << "Test Naive-Example4\n";

	timer.start();
	for (int i = 0; i < N; i++) {
		cudaMemset(G, 0, sizeof(float));
		example4::naive(A, B, C, G, m, n, k);
	}
	timer.stop_and_wait();

	std::cout << "Naive-Example4 result is " << timer.duration(N) << " ms\n";


	std::cout << "Test Naive-Example5\n";

	timer.start();
	for (int i = 0; i < N; i++) {
		cudaMemset(G, 0, sizeof(float));
		example5::naive(A, A, C, G, m, n, k);
	}
	timer.stop_and_wait();

	std::cout << "Naive-Example5 result is " << timer.duration(N) << " ms\n";


	std::cout << "Test Naive-Example6\n";

	timer.start();
	for (int i = 0; i < N; i++) {
		cudaMemset(mu, 0, sizeof(int));
		cudaMemset(G, 0, sizeof(float));
		example6::naive(A, B, C, G, (int *)mu, m, n, k);
	}
	timer.stop_and_wait();

	std::cout << "Naive-Example6 result is " << timer.duration(N) << " ms\n";


	return cudaSuccess;
}
