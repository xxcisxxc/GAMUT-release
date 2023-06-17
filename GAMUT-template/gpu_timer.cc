#include <stdexcept>

#include "gpu_timer.h"

GpuTimer::GpuTimer()
{
	cudaError_t result;
	for (auto & event : events) {
		result = cudaEventCreate(&event);
		if (result != cudaSuccess) {
			throw std::runtime_error("Failed to create CUDA event");
		}
	}
}

GpuTimer::~GpuTimer()
{
	for (auto & event : events) {
		cudaEventDestroy(event);
	}
}

/// Records a start event in the stream
void GpuTimer::start(cudaStream_t stream)
{
	cudaError_t result = cudaEventRecord(events[0], stream);
	if (result != cudaSuccess) {
		throw std::runtime_error("Failed to record start event.");
	}
}

/// Records a stop event in the stream
void GpuTimer::stop(cudaStream_t stream)
{
	cudaError_t result = cudaEventRecord(events[1], stream);
	if (result != cudaSuccess) {
		throw std::runtime_error("Failed to record stop event.");
	}
}

/// Records a stop event in the stream and synchronizes on the stream
void GpuTimer::stop_and_wait(cudaStream_t stream)
{
	stop(stream);

	cudaError_t result;
	if (stream) {
		result = cudaStreamSynchronize(stream);
		if (result != cudaSuccess) {
			throw std::runtime_error("Failed to synchronize with non-null CUDA stream.");
		}
	}
	else {
		result = cudaDeviceSynchronize();
		if (result != cudaSuccess) {
			throw std::runtime_error("Failed to synchronize with CUDA device.");
		}
	}
}

/// Returns the duration in miliseconds
double GpuTimer::duration(int iterations) const
{
	float avg_ms;

	cudaError_t result = cudaEventElapsedTime(&avg_ms, events[0], events[1]);
	if (result != cudaSuccess) {
		throw std::runtime_error("Failed to query elapsed time from CUDA events.");
	}

	return double(avg_ms) / double(iterations);
}