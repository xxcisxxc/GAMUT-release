#pragma once

#include "utils.h"
#include "kernel.h"

namespace standard {
cudaError_t launch
(
	Params param,
	cudaStream_t stream = nullptr
);
}