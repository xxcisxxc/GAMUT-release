#pragma once

#include "iterator.h"
#include "operator.h"
#include "utils.h"

namespace standard {
union SharedMemory {
  typename InIterator::InSharedMemory in_smem;
  typename OutIterator::OutSharedMemory out_smem;
};

__global__ void compute(Params param);

__global__ void reduce(Params param);
} // namespace standard