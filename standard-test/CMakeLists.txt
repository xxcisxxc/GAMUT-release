cmake_minimum_required(VERSION 3.12.4 FATAL_ERROR)

project(MMLT_GPU LANGUAGES CUDA CXX C)

find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

set(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "Enable only the header library" FORCE)
set(CUTLASS_NVCC_ARCHS "75" CACHE STRING "The SM architectures requested." FORCE)
add_subdirectory(cutlass)
add_subdirectory(workkernel)

set_source_files_properties(main.cc PROPERTIES LANGUAGE CUDA)

add_executable(test m_alloc.cu mm_func.cu gpu_timer.cc main.cc workkernel/launch.cu workkernel/kernel.cu)
target_link_libraries(test curand cublas CUTLASS)
set_target_properties(test PROPERTIES CUDA_ARCHITECTURES "75")
