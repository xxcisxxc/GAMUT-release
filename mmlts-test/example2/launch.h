#pragma once

#include "utils.h"
#include "kernel.h"

namespace example2 {
cudaError_t launch
(
	Params param,
	cudaStream_t stream = nullptr
);
}