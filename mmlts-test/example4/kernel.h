#pragma once

#include "utils.h"
#include "iterator.h"
#include "operator.h"

namespace example4 {
union SharedMemory
{
	InSharedMemory in_smem;
	OutSharedMemory out_smem;
};


__global__
void compute
(
    Params param
);

__global__
void reduce
(
	Params param
);
}