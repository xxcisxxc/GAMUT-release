#pragma once

#include "utils.h"
#include "kernel.h"

namespace example5 {
cudaError_t launch
(
	Params param,
	cudaStream_t stream = nullptr
);
}