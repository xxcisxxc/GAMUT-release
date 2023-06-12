cmake_minimum_required(VERSION 3.12.4 FATAL_ERROR)

set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.7/bin/nvcc)
project(MMLT_GPU LANGUAGES CUDA CXX C)

#set_source_files_properties(main.cc PROPERTIES LANGUAGE CUDA)

#add_executable(test launch.cu kernel.cu main.cc)
#set_target_properties(test PROPERTIES CUDA_ARCHITECTURES "75")