cmake_minimum_required(VERSION 3.12.4 FATAL_ERROR)

set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.7/bin/nvcc)
project(MMLT_GPU LANGUAGES CUDA CXX C)

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

#add_subdirectory(workkernel)
#add_subdirectory(query2)

set_source_files_properties(main.cc PROPERTIES LANGUAGE CUDA)

add_executable(test m_alloc.cu gpu_timer.cc main.cc
	workkernel/launch.cu workkernel/kernel.cu
	query2/launch.cu query2/kernel.cu
	example1/launch.cu example1/kernel.cu
	example2/launch.cu example2/kernel.cu
	example3/launch.cu example3/kernel.cu
	example4/launch.cu example4/kernel.cu
	example5/launch.cu example5/kernel.cu
	example6/launch.cu example6/kernel.cu example6/heap.cu)
target_link_libraries(test curand)
set_target_properties(test PROPERTIES CUDA_ARCHITECTURES "75")