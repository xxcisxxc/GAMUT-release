#ifndef GPU_TIMER_
#define GPU_TIMER_

#include <cuda_runtime.h>

/**
 * @brief record GPU compute time\n
	Copied from https://github.com/NVIDIA/cutlass/blob/master/tools/profiler/src/gpu_timer.h\n
	and https://github.com/NVIDIA/cutlass/blob/master/tools/profiler/src/gpu_timer.cpp
 * 
 */
class GpuTimer
{
private:
	cudaEvent_t events[2];

public:
	GpuTimer();
	~GpuTimer();

	/// Records a start event in the stream
	void start(cudaStream_t stream = nullptr);

	/// Records a stop event in the stream
	void stop(cudaStream_t stream = nullptr);

	/// Records a stop event in the stream and synchronizes on the stream
	void stop_and_wait(cudaStream_t stream = nullptr);

	/// Returns the duration in miliseconds
	double duration(int iterations = 1) const;
};

#endif /* GPU_TIMER_ */